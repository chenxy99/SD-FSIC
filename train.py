from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F

import numpy as np

import time
import os
from six.moves import cPickle
import traceback
from collections import defaultdict

import opts
import models
from dataloader import *
import skimage.io
import eval_utils
import misc.utils as utils
from misc.rewards import init_scorer, get_self_critical_reward
from misc.loss_wrapper import LossWrapper
from misc.kdloss_wrapper import KDLossWrapper

from misc.tools import construct_triplet

from models.MultiModels import MultiModels
from models.SenEncodeModel import SenEncodeModel
from models.AttEnsemble import AttEnsemble

import nltk
from misc.vob_list import vob_transform_list

def add_summary_value(writer, key, value, iteration):
    if writer:
        writer.add_scalar(key, value, iteration)

def train(opt):

    ################################
    # Build dataloader
    ################################
    loader = DataLoader(opt)
    opt.vocab_size = loader.vocab_size
    opt.seq_length = loader.seq_length

    ##########################
    # Initialize infos
    ##########################
    infos = {
        'iter': 0,
        'epoch': 0,
        'loader_state_dict': None,
        'vocab': loader.get_vocab(),
    }
    # Load old infos(if there is) and check if models are compatible
    if opt.start_from is not None and os.path.isfile(os.path.join(opt.start_from, 'infos_'+opt.id+'.pkl')):
        with open(os.path.join(opt.start_from, 'infos_'+opt.id+'.pkl'), 'rb') as f:
            infos = utils.pickle_load(f)
            saved_model_opt = infos['opt']
            need_be_same=["caption_model", "rnn_type", "rnn_size", "num_layers"]
            for checkme in need_be_same:
                assert getattr(saved_model_opt, checkme) == getattr(opt, checkme), "Command line argument and saved model disagree on '%s' " % checkme
    infos['opt'] = opt

    #########################
    # Build logger
    #########################
    # naive dict logger
    histories = defaultdict(dict)
    if opt.start_from is not None and os.path.isfile(os.path.join(opt.start_from, 'histories_'+opt.id+'.pkl')):
        with open(os.path.join(opt.start_from, 'histories_'+opt.id+'.pkl'), 'rb') as f:
            histories.update(utils.pickle_load(f))

    # tensorboard logger
    tb_summary_writer = SummaryWriter(opt.checkpoint_path)

    ##########################
    # Build model
    ##########################
    opt.vocab = loader.get_vocab()
    multi_models_list = []
    for order in range(opt.number_of_models):
        multi_models_list.append(models.setup(opt).cuda())
    for order in range(opt.number_of_models):
        multi_models_list.append(models.setup(opt).cuda())
    for order in range(opt.number_of_models, 2*opt.number_of_models):
        for param in multi_models_list[order].parameters():
            param.detach_()
    for order in range(opt.number_of_models):
        for param, param_ema in zip(multi_models_list[order].parameters(), multi_models_list[order + opt.number_of_models].parameters()):
            param_ema.data = param.data.clone()
    # multi_models = MultiModels(multi_models_list)
    # multi_models_list.append(SenEncodeModel(opt).cuda())
    multi_models = nn.ModuleList(multi_models_list)
    del opt.vocab
    # Load pretrained weights:
    if opt.start_from is not None and os.path.isfile(os.path.join(opt.start_from, 'model.pth')):
        multi_models.load_state_dict(torch.load(os.path.join(opt.start_from, 'model.pth')))
    
    # Wrap generation model with loss function(used for training)
    # This allows loss function computed separately on each machine
    lw_models = nn.ModuleList([LossWrapper(multi_models[index], opt) for index in range(opt.number_of_models)])
    kdlw_models = nn.ModuleList([KDLossWrapper(multi_models[index], opt) for index in range(opt.number_of_models)])
    lw_models_ema = nn.ModuleList([LossWrapper(multi_models[opt.number_of_models + index], opt) for index in range(opt.number_of_models)])
    kdlw_models_ema = nn.ModuleList([KDLossWrapper(multi_models[opt.number_of_models + index], opt) for index in range(opt.number_of_models)])
    # Wrap with dataparallel
    dp_models = nn.ModuleList([torch.nn.DataParallel(multi_models[index]) for index in range(opt.number_of_models)])
    dp_lw_models = nn.ModuleList([torch.nn.DataParallel(lw_models[index]) for index in range(opt.number_of_models)])
    dp_kdlw_models = nn.ModuleList([torch.nn.DataParallel(kdlw_models[index]) for index in range(opt.number_of_models)])
    dp_models_ema = nn.ModuleList([torch.nn.DataParallel(multi_models[opt.number_of_models + index]) for index in range(opt.number_of_models)])
    dp_lw_models_ema = nn.ModuleList([torch.nn.DataParallel(lw_models_ema[index]) for index in range(opt.number_of_models)])
    dp_kdlw_models_ema = nn.ModuleList([torch.nn.DataParallel(kdlw_models_ema[index]) for index in range(opt.number_of_models)])

    ##########################
    #  Build optimizer
    ##########################
    if opt.noamopt:
        assert opt.caption_model in ['transformer', 'bert', 'm2transformer'], 'noamopt can only work with transformer'
        optimizer = utils.get_std_opt(multi_models, factor=opt.noamopt_factor, warmup=opt.noamopt_warmup)
    elif opt.reduce_on_plateau:
        optimizer = utils.build_optimizer(multi_models.parameters(), opt)
        optimizer = utils.ReduceLROnPlateau(optimizer, factor=0.5, patience=3)
    else:
        optimizer = utils.build_optimizer(multi_models.parameters(), opt)
    # Load the optimizer
    if opt.start_from is not None and os.path.isfile(os.path.join(opt.start_from,"optimizer.pth")):
        optimizer.load_state_dict(torch.load(os.path.join(opt.start_from, 'optimizer.pth')))

    ##########################
    #  Build loss
    ##########################
    # triplet_loss = nn.TripletMarginLoss()

    #########################
    # Get ready to start
    #########################
    iteration = infos['iter']
    epoch = infos['epoch']
    # For back compatibility
    if 'iterators' in infos:
        infos['loader_state_dict'] = {split: {'index_list': infos['split_ix'][split], 'iter_counter': infos['iterators'][split]}
                                      for split in ['paired_train', 'unpaired_images_train', 'unpaired_captions_train', 'train', 'val', 'test']}
    loader.load_state_dict(infos['loader_state_dict'])
    if opt.load_best_score == 1:
        best_val_score = infos.get('best_val_score', None)
    if opt.noamopt:
        optimizer._step = iteration
    # flag indicating finish of an epoch
    # Always set to True at the beginning to initialize the lr or etc.
    epoch_done = True
    # Assure in training mode
    dp_lw_models.train()
    dp_kdlw_models.train()
    dp_lw_models_ema.train()
    dp_kdlw_models_ema.train()

    # Build the ensemble model
    # # Setup the model
    model_ensemble = AttEnsemble(multi_models_list[opt.number_of_models:2*opt.number_of_models], weights=None)
    # model_ensemble.seq_length = 20
    model_ensemble.cuda()
    # model_ensemble.eval()
    kd_model_outs_list = []


    # Start training
    try:
        while True:
            # Stop if reaching max epochs
            if epoch >= opt.max_epochs and opt.max_epochs != -1:
                break

            if epoch_done:
                if not opt.noamopt and not opt.reduce_on_plateau:
                    # Assign the learning rate
                    if epoch > opt.learning_rate_decay_start and opt.learning_rate_decay_start >= 0:
                        frac = (epoch - opt.learning_rate_decay_start) // opt.learning_rate_decay_every
                        decay_factor = opt.learning_rate_decay_rate  ** frac
                        opt.current_lr = opt.learning_rate * decay_factor
                    else:
                        opt.current_lr = opt.learning_rate
                    utils.set_lr(optimizer, opt.current_lr) # set the decayed rate
                # Assign the scheduled sampling prob
                if epoch > opt.scheduled_sampling_start and opt.scheduled_sampling_start >= 0:
                    frac = (epoch - opt.scheduled_sampling_start) // opt.scheduled_sampling_increase_every
                    opt.ss_prob = min(opt.scheduled_sampling_increase_prob  * frac, opt.scheduled_sampling_max_prob)
                    for index in range(opt.number_of_models):
                        multi_models[index].ss_prob = opt.ss_prob

                # If start self critical training
                if opt.self_critical_after != -1 and epoch >= opt.self_critical_after:
                    sc_flag = True
                    init_scorer(opt.cached_tokens)
                else:
                    sc_flag = False
                
                # If start structure loss training
                if opt.structure_after != -1 and epoch >= opt.structure_after:
                    struc_flag = True
                    init_scorer(opt.cached_tokens)
                else:
                    struc_flag = False

                if epoch >= opt.paired_train_epoch:
                    opt.current_lambda_x = opt.hyper_parameter_lambda_x * \
                                         (epoch - (opt.paired_train_epoch - 1)) /\
                                         (opt.max_epochs - opt.paired_train_epoch)
                    opt.current_lambda_y = opt.hyper_parameter_lambda_y * \
                                           (epoch - (opt.paired_train_epoch - 1)) / \
                                           (opt.max_epochs - opt.paired_train_epoch)

                epoch_done = False
                    
            start = time.time()
            # Load data from train split (0)
            if epoch < opt.language_pretrain_epoch:
                data = loader.get_batch('unpaired_captions_train')
            elif epoch < opt.paired_train_epoch:
                data = loader.get_batch('paired_train')
            else:
                data = loader.get_batch('paired_train')
                unpaired_data = loader.get_batch('unpaired_images_train')
                unpaired_caption = loader.get_batch('unpaired_captions_train')
            print('Read data:', time.time() - start)

            torch.cuda.synchronize()
            start = time.time()
            if epoch < opt.language_pretrain_epoch:
                tmp = [data['fc_feats']*0, data['att_feats']*0, data['labels'], data['masks'], data['att_masks']]
            elif epoch < opt.paired_train_epoch:
                tmp = [data['fc_feats'], data['att_feats'], data['labels'], data['masks'], data['att_masks']]
            else:
                tmp = [data['fc_feats'], data['att_feats'], data['labels'], data['masks'], data['att_masks']]
                unpaired_tmp = [unpaired_data['fc_feats'], unpaired_data['att_feats'], unpaired_data['labels'],
                                unpaired_data['masks'], unpaired_data['att_masks']]
                unpaired_caption_tmp = [unpaired_caption['fc_feats']*0, unpaired_caption['att_feats']*0,
                                        unpaired_caption['labels'],
                                        unpaired_caption['masks'], unpaired_caption['att_masks']]

            tmp = [_ if _ is None else _.cuda() for _ in tmp]
            fc_feats, att_feats, labels, masks, att_masks = tmp

            if epoch >= opt.paired_train_epoch:
                unpaired_tmp = [_ if _ is None else _.cuda() for _ in unpaired_tmp]
                unpaired_fc_feats, unpaired_att_feats, unpaired_labels, unpaired_masks, unpaired_att_masks = unpaired_tmp
                unpaired_caption_tmp = [_ if _ is None else _.cuda() for _ in unpaired_caption_tmp]
                unpaired_caption_fc_feats, unpaired_caption_att_feats, unpaired_caption_labels, unpaired_caption_masks, unpaired_caption_att_masks = unpaired_caption_tmp
                unpaired_caption_fc_feats = unpaired_caption_fc_feats.repeat(5, 1)
                unpaired_caption_fc_feats = opt.std_pseudo_visual_feature * torch.randn_like(unpaired_caption_fc_feats)
                unpaired_caption_att_feats = unpaired_caption_att_feats.repeat(5, 1, 1)
                unpaired_caption_fc_feats.requires_grad = True
                unpaired_caption_att_feats.requires_grad = True
                unpaired_caption_labels = unpaired_caption_labels.reshape(unpaired_caption_fc_feats.shape[0], -1)
                unpaired_caption_masks = unpaired_caption_masks.reshape(unpaired_caption_fc_feats.shape[0], -1)

            optimizer.zero_grad()
            if epoch < opt.language_pretrain_epoch:
                language_loss = 0
                model_outs_list = []
                for index in range(opt.number_of_models):
                    model_out = dp_lw_models[index](fc_feats, att_feats, labels, masks, att_masks, data['gts'], torch.arange(0, len(data['gts'])), sc_flag, struc_flag)
                    model_outs_list.append(model_out)
                    language_loss += model_out['loss'].mean()

                loss = language_loss
            elif epoch < opt.paired_train_epoch:
                language_loss = 0
                model_outs_list = []
                for index in range(opt.number_of_models):
                    model_out = dp_lw_models[index](fc_feats, att_feats, labels, masks, att_masks, data['gts'],
                                                    torch.arange(0, len(data['gts'])), sc_flag, struc_flag)
                    model_outs_list.append(model_out)
                    language_loss += model_out['loss'].mean()

                loss = language_loss
            else:
                language_loss = 0
                model_outs_list = []
                for index in range(opt.number_of_models):
                    model_out = dp_lw_models[index](fc_feats, att_feats, labels, masks, att_masks, data['gts'],
                                                    torch.arange(0, len(data['gts'])), sc_flag, struc_flag)
                    model_outs_list.append(model_out)
                    language_loss += model_out['loss'].mean()
                loss = language_loss


                # else:
                # for unpaired image sentences
                # # Setup the model
                # model_ensemble = AttEnsemble(multi_models_list[:opt.number_of_models], weights=None)
                # model_ensemble.seq_length = 16
                # model_ensemble.cuda()
                # model_ensemble.eval()

                model_ensemble.eval()
                eval_kwargs = dict()
                eval_kwargs.update(vars(opt))

                with torch.no_grad():
                    seq, seq_logprobs = model_ensemble(unpaired_fc_feats, unpaired_att_feats, unpaired_att_masks, opt=eval_kwargs,
                                                            mode='sample')
                    # val_loss, predictions, lang_stats = eval_utils.eval_split(model_ensemble, lw_models[0].crit, loader,
                    #                                                           eval_kwargs)
                # print('\n'.join([utils.decode_sequence(loader.get_vocab(), _['seq'].unsqueeze(0))[0] for _ in
                #                  model_ensemble.done_beams[0]]))
                # print('++' * 10)
                # for ii in range(10):
                #     sents = utils.decode_sequence(loader.get_vocab(), seq[ii].unsqueeze(0))
                #     gt_sent = utils.decode_sequence(loader.get_vocab(), labels[ii,0].unsqueeze(0))
                #     a=1

                model_ensemble.train()

                model_ensemble_sudo_labels = labels.new_zeros(
                    (opt.batch_size, opt.beam_size, eval_kwargs['max_length'] + 2))
                model_ensemble_sudo_log_prob = masks.new_zeros(
                    (opt.batch_size, opt.beam_size, eval_kwargs['max_length'] + 2, len(loader.get_vocab()) + 1))
                model_ensemble_sum_log_prob = masks.new_zeros(
                    (opt.batch_size, opt.beam_size))

                for batch_index in range(opt.batch_size):
                    for beam_index in range(opt.beam_size):
                    # for beam_index in range(3):
                        pred = model_ensemble.done_beams[batch_index][beam_index]['seq']
                        log_prob = model_ensemble.done_beams[batch_index][beam_index]['logps']
                        model_ensemble_sudo_labels[batch_index, beam_index, 1:pred.shape[0] + 1] = pred
                        model_ensemble_sudo_log_prob[batch_index, beam_index, 1:pred.shape[0] + 1] = log_prob
                        model_ensemble_sum_log_prob[batch_index][beam_index] = model_ensemble.done_beams[batch_index][beam_index]['p']

                # model_ensemble_prob = F.softmax(model_ensemble_sum_log_prob)

                data_ensemble_sudo_gts = list()
                for data_ensemble_sudo_gts_index in range(model_ensemble_sudo_labels.shape[0]):
                    data_ensemble_sudo_gts.append(model_ensemble_sudo_labels[data_ensemble_sudo_gts_index, :, 1:-1].data.cpu().numpy())

                # generated_sentences = list()
                # for i in range(unpaired_fc_feats.shape[0]):
                #     generated_sentences.append(
                #         [utils.decode_sequence(loader.get_vocab(), _['seq'].unsqueeze(0))[0] for _ in
                #          model_ensemble.done_beams[i]])
                #
                # pos_tag_results = list()
                # for i in range(unpaired_fc_feats.shape[0]):
                #     generated_sentences_i = generated_sentences[i]
                #     pos_tag_results_i = []
                #     for text in generated_sentences_i:
                #         text_tokenize = nltk.word_tokenize(text)
                #         pos_tag_results_i_jbeam = []
                #         for vob, vob_type in nltk.pos_tag(text_tokenize):
                #             if vob_type == 'NN' or vob_type == 'NNS':
                #                 pos_tag_results_i_jbeam.append(vob)
                #         pos_tag_results_i.append(pos_tag_results_i_jbeam)
                #     pos_tag_results.append(pos_tag_results_i)

                # for i in range(fc_feats.shape[0]):
                #     print('\n'.join([utils.decode_sequence(loader.get_vocab(), _['seq'].unsqueeze(0))[0] for _ in
                #                      model_ensemble.done_beams[i]]))
                #     print('--' * 10)
                # dets = data['dets']
                #
                # promising_flag = labels.new_zeros(opt.batch_size, opt.beam_size)
                # for batch_index in range(opt.batch_size):
                #     dets_batch = dets[batch_index]
                #     for beam_index in range(opt.beam_size):
                #         indicator = [0] * len(dets_batch)
                #         pos_tag_batch_beam = pos_tag_results[batch_index][beam_index]
                #         for pos_tag_val in pos_tag_batch_beam:
                #             for ii in range(len(dets_batch)):
                #                 possible_list = vob_transform_list[dets_batch[ii]]
                #                 if pos_tag_val in possible_list:
                #                     indicator[ii] = 1
                #         if sum(indicator) == len(dets_batch) or sum(indicator) >= 2:
                #             promising_flag[batch_index, beam_index] = 1
                #
                # # model_ensemble_sudo_log_prob = model_ensemble_sudo_log_prob * promising_flag.unsqueeze(-1).unsqueeze(-1)
                # model_ensemble_sudo_labels = model_ensemble_sudo_labels * promising_flag.unsqueeze(-1)


                #sudo_masks_for_model = sudo_masks_for_model.detach()
                distilling_loss = 0
                # We use the random study machinism
                who_to_study = random.randint(0, opt.number_of_models - 1)

                # for index in range(opt.number_of_models):
                #     model_out = dp_kdlw_models[index](unpaired_fc_feats, unpaired_att_feats, model_ensemble_sudo_labels,
                #                                     model_ensemble_sudo_log_prob, att_masks, data_ensemble_sudo_gts,
                #                                     torch.arange(0, len(data_ensemble_sudo_gts)), sc_flag,
                #                                     struc_flag, model_ensemble_sum_log_prob)
                #     kd_model_outs_list.append(model_out)

                model_out = dp_kdlw_models[who_to_study](unpaired_fc_feats, unpaired_att_feats, model_ensemble_sudo_labels,
                                                model_ensemble_sudo_log_prob, att_masks, data_ensemble_sudo_gts,
                                                torch.arange(0, len(data_ensemble_sudo_gts)), sc_flag,
                                                struc_flag, model_ensemble_sum_log_prob)
                # kd_model_outs_list.append(model_out)
                distilling_loss += model_out['loss'].mean()
                loss += opt.number_of_models * opt.current_lambda_x * distilling_loss



                ###################################################################
                # use unlabelled captions
                # simple_sgd = utils.gradient_descent(unpaired_caption_fc_feats, stepsize=1e3)
                simple_sgd = utils.gradient_descent_adagrad(unpaired_caption_fc_feats, stepsize=1)
                gts_tmp = unpaired_caption['gts']
                new_gts = []
                for ii in range(len(data['gts'])):
                    for jj in range(gts_tmp[ii].shape[0]):
                        new_gts.append(gts_tmp[ii][jj])
                unpaired_caption['gts'] = new_gts
                for itr in range(opt.inner_iteration):
                    unlabelled_caption_model_out = dp_lw_models_ema[itr%opt.number_of_models](unpaired_caption_fc_feats, unpaired_caption_att_feats,
                                                                                unpaired_caption_labels, unpaired_caption_masks,
                                                                                unpaired_caption_att_masks, unpaired_caption['gts'],
                                                                                torch.arange(0, len(unpaired_caption['gts'])), sc_flag, struc_flag)
                    unlabelled_caption_loss = unlabelled_caption_model_out['loss'].mean()
                    unlabelled_caption_loss.backward()
                    # print(unlabelled_caption_loss)
                    simple_sgd.update(unpaired_caption_fc_feats)
                    # a=1

                unpaired_caption_fc_feats.requires_grad = False
                unpaired_caption_att_feats.requires_grad = False
                unlabelled_caption_model_out = dp_lw_models[who_to_study](unpaired_caption_fc_feats, unpaired_caption_att_feats, unpaired_caption_labels,
                                                                                            unpaired_caption_masks,
                                                                                            unpaired_caption_att_masks,
                                                                                            unpaired_caption['gts'],
                                                                                            torch.arange(0, len(
                                                                                                unpaired_caption[
                                                                                                    'gts'])), sc_flag,
                                                                                            struc_flag)
                unlabelled_caption_loss = unlabelled_caption_model_out['loss'].mean()
                loss += opt.number_of_models * opt.current_lambda_y * unlabelled_caption_loss


            loss.backward()
            if opt.grad_clip_value != 0:
                getattr(torch.nn.utils, 'clip_grad_%s_' %(opt.grad_clip_mode))(multi_models.parameters(), opt.grad_clip_value)
            optimizer.step()

            for order in range(opt.number_of_models):
                for param, param_ema in zip(multi_models_list[order].parameters(),
                                            multi_models_list[order + opt.number_of_models].parameters()):
                    param_ema.data = opt.alpha*param_ema.data + (1-opt.alpha)*param.data

            train_loss = loss.item()
            torch.cuda.synchronize()
            end = time.time()
            # if struc_flag:
            #     print("iter {} (epoch {}), train_loss = {:.3f}, lm_loss = {:.3f}, struc_loss = {:.3f}, time/batch = {:.3f}" \
            #         .format(iteration, epoch, train_loss, model_out['lm_loss'].mean().item(), model_out['struc_loss'].mean().item(), end - start))
            # elif not sc_flag:
            #     print("iter {} (epoch {}), train_loss = {:.3f}, time/batch = {:.3f}" \
            #         .format(iteration, epoch, train_loss, end - start))
            # else:
            #     print("iter {} (epoch {}), avg_reward = {:.3f}, time/batch = {:.3f}" \
            #         .format(iteration, epoch, model_out['reward'].mean(), end - start))
            if struc_flag:
                print("iter {} (epoch {}), train_loss = {:.3f}, lm_loss = {:.3f}, struc_loss = {:.3f}, time/batch = {:.3f}" \
                    .format(iteration, epoch, train_loss/opt.number_of_models, sum([model_outs_list[index]['lm_loss'].mean().item() for index in range(opt.number_of_models)])/opt.number_of_models,
                            sum([model_outs_list[index]['struc_loss'].mean().item() for index in range(opt.number_of_models)])/opt.number_of_models,
                            end - start))
            elif not sc_flag:
                print("iter {} (epoch {}), train_loss = {:.3f}, time/batch = {:.3f}" \
                    .format(iteration, epoch, language_loss.item()/opt.number_of_models, end - start))
            else:
                print("iter {} (epoch {}), avg_reward = {:.3f}, time/batch = {:.3f}" \
                    .format(iteration, epoch, sum([model_outs_list[index]['reward'].mean().item() for index in range(opt.number_of_models)])/opt.number_of_models, end - start))

            # Update the iteration and epoch
            iteration += 1
            if epoch < opt.paired_train_epoch:
                if data['bounds']['wrapped']:
                    epoch += 1
                    epoch_done = True
            else:
                if data['bounds']['wrapped']:
                    epoch += 1
                    epoch_done = True

            # Write the training loss summary
            if (iteration % opt.losses_log_every == 0):
                # tb_summary_writer.add_scalar('train_loss', train_loss, iteration)
                for index in range(opt.number_of_models):
                    model_id = 'model_{}'.format(index)
                    tb_summary_writer.add_scalars('language_loss', {model_id: model_outs_list[index]['loss'].mean().item()}, iteration)
                if epoch >= opt.paired_train_epoch:
                    # for index in range(opt.number_of_models):
                    #     model_id = 'model_{}'.format(index)
                    #     kd_model_outs_val = 0 if len(kd_model_outs_list) == 0 else kd_model_outs_list[index]['loss'].mean().item()
                    #     tb_summary_writer.add_scalars('distilling_loss',
                    #                                   {model_id: kd_model_outs_val},
                    #                                   iteration)
                    tb_summary_writer.add_scalar('distilling_loss', distilling_loss.item(), iteration)
                    tb_summary_writer.add_scalar('unlabelled_caption_loss', unlabelled_caption_loss.item(), iteration)
                    tb_summary_writer.add_scalar('hyper_parameter_lambda_x', opt.current_lambda_x, iteration)
                    tb_summary_writer.add_scalar('hyper_parameter_lambda_y', opt.current_lambda_y, iteration)
                # tb_summary_writer.add_scalar('triplet_loss', triplet_loss_val.item(), iteration)
                if opt.noamopt:
                    opt.current_lr = optimizer.rate()
                elif opt.reduce_on_plateau:
                    opt.current_lr = optimizer.current_lr
                tb_summary_writer.add_scalar('learning_rate', opt.current_lr, iteration)
                tb_summary_writer.add_scalar('scheduled_sampling_prob', multi_models[0].ss_prob, iteration)
                if sc_flag:
                    for index in range(opt.number_of_models):
                        # tb_summary_writer.add_scalar('avg_reward', model_out['reward'].mean(), iteration)
                        model_id = 'model_{}'.format(index)
                        tb_summary_writer.add_scalars('avg_reward', {model_id: model_outs_list[index]['reward'].mean().item()}, iteration)
                elif struc_flag:
                    # tb_summary_writer.add_scalar('lm_loss', model_out['lm_loss'].mean().item(), iteration)
                    # tb_summary_writer.add_scalar('struc_loss', model_out['struc_loss'].mean().item(), iteration)
                    # tb_summary_writer.add_scalar('reward', model_out['reward'].mean().item(), iteration)
                    # tb_summary_writer.add_scalar('reward_var', model_out['reward'].var(1).mean(), iteration)
                    model_id = 'model_{}'.format(index)
                    for index in range(opt.number_of_models):
                        tb_summary_writer.add_scalars('lm_loss', {model_id: model_outs_list[index]['lm_loss'].mean().item()}, iteration)
                        tb_summary_writer.add_scalars('struc_loss', {model_id: model_outs_list[index]['struc_loss'].mean().item()}, iteration)
                        tb_summary_writer.add_scalars('reward', {model_id: model_outs_list[index]['reward'].mean().item()}, iteration)
                        tb_summary_writer.add_scalars('reward_var', {model_id: model_outs_list[index]['reward'].var(1).mean()}, iteration)

                histories['loss_history'][iteration] = train_loss if not sc_flag else sum([model_outs_list[index]['reward'].mean().item() for index in range(opt.number_of_models)])/opt.number_of_models
                histories['lr_history'][iteration] = opt.current_lr
                histories['ss_prob_history'][iteration] = multi_models[0].ss_prob

            # update infos
            infos['iter'] = iteration
            infos['epoch'] = epoch
            infos['loader_state_dict'] = loader.state_dict()
            
            # make evaluation on validation set, and save model
            if (iteration % opt.save_checkpoint_every == 0 and not opt.save_every_epoch and epoch >= opt.paired_train_epoch) or \
                (epoch_done and opt.save_every_epoch and epoch >= opt.paired_train_epoch):
                # load ensemble
                # Setup the model
                model = AttEnsemble(multi_models_list[opt.number_of_models:2*opt.number_of_models], weights=None)
                model.seq_length = opt.max_length
                model.cuda()
                model.eval()
                # eval model
                eval_kwargs = {'split': 'val',
                               'dataset': opt.input_json}
                eval_kwargs.update(vars(opt))
                # eval_kwargs['beam_size'] = 5
                # eval_kwargs['verbose_beam'] = 1
                # eval_kwargs['verbose_loss'] = 1
                # val_loss, predictions, lang_stats = eval_utils.eval_split(
                #     dp_model, lw_model.crit, loader, eval_kwargs)
                with torch.no_grad():
                    val_loss, predictions, lang_stats = eval_utils.eval_split(model, lw_models[0].crit, loader, eval_kwargs)
                model.train()

                if opt.reduce_on_plateau:
                    if 'CIDEr' in lang_stats:
                        optimizer.scheduler_step(-lang_stats['CIDEr'])
                    else:
                        optimizer.scheduler_step(val_loss)
                # Write validation result into summary
                tb_summary_writer.add_scalar('validation loss', val_loss, iteration)
                if lang_stats is not None:
                    for k,v in lang_stats.items():
                        tb_summary_writer.add_scalar(k, v, iteration)
                histories['val_result_history'][iteration] = {'loss': val_loss, 'lang_stats': lang_stats, 'predictions': predictions}

                # Save model if is improving on validation result
                if opt.language_eval == 1:
                    current_score = lang_stats['CIDEr']
                else:
                    current_score = - val_loss

                best_flag = False

                if best_val_score is None or current_score > best_val_score:
                    best_val_score = current_score
                    best_flag = True

                # Dump miscalleous informations
                infos['best_val_score'] = best_val_score

                utils.save_checkpoint(opt, multi_models, infos, optimizer, histories)
                if opt.save_history_ckpt:
                    utils.save_checkpoint(opt, multi_models, infos, optimizer,
                        append=str(epoch) if opt.save_every_epoch else str(iteration))

                if best_flag:
                    utils.save_checkpoint(opt, multi_models, infos, optimizer, append='best')

            # if epoch_done and epoch == opt.paired_train_epoch:
            #     utils.save_checkpoint(opt, multi_models, infos, optimizer, histories)
            #     if opt.save_history_ckpt:
            #         utils.save_checkpoint(opt, multi_models, infos, optimizer,
            #                               append=str(epoch) if opt.save_every_epoch else str(iteration))
            #     cmd = 'cp -r ' + 'log_' + opt.id + ' ' + 'log_' + opt.id + '_backup'
            #     os.system(cmd)

    except (RuntimeError, KeyboardInterrupt):
        print('Save ckpt on exception ...')
        utils.save_checkpoint(opt, multi_models, infos, optimizer)
        print('Save ckpt done.')
        stack_trace = traceback.format_exc()
        print(stack_trace)


opt = opts.parse_opt()
train(opt)
