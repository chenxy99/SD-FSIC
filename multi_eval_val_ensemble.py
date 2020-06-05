from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import numpy as np

import time
import os
from six.moves import cPickle

import opts
import models
from dataloader import *
from dataloaderraw import *
import eval_utils
import argparse
import misc.utils as utils
import torch
import torch.nn as nn

# Input arguments and options
parser = argparse.ArgumentParser()
# Input paths
parser.add_argument('--weights', nargs='+', required=False, default=None, help='id of the models to ensemble')
parser.add_argument('--model', type=str, default='',
                help='path to model to evaluate')
parser.add_argument('--infos_path', type=str, default='',
                help='path to infos to evaluate')
parser.add_argument('--number_of_models', type=int, default=1,
                    help='The number of multi-models.')
# parser.add_argument('--infos_paths', nargs='+', required=True, help='path to infos to evaluate')
opts.add_eval_options(parser)
opts.add_diversity_opts(parser)

opt = parser.parse_args()

# model_infos = []
# model_paths = []
# for id in opt.ids:
#     if '-' in id:
#         id, app = id.split('-')
#         app = '-'+app
#     else:
#         app = ''
#     model_infos.append(utils.pickle_load(open('log_%s/infos_%s%s.pkl' %(id, id, app), 'rb')))
#     model_paths.append('log_%s/model%s.pth' %(id,app))

# Load infos
with open(opt.infos_path, 'rb') as f:
    infos = utils.pickle_load(f)

# override and collect parameters
replace = ['input_fc_dir', 'input_att_dir', 'input_box_dir', 'input_label_h5', 'input_json', 'batch_size', 'id']
ignore = ['start_from']

for k in vars(infos['opt']).keys():
    if k in replace:
        setattr(opt, k, getattr(opt, k) or getattr(infos['opt'], k, ''))
    elif k not in ignore:
        if not k in vars(opt):
            vars(opt).update({k: vars(infos['opt'])[k]}) # copy over options from model

vocab = infos['vocab'] # ix -> word mapping

opt.split = 'val'
pred_fn = os.path.join('eval_results/', '.saved_pred_'+ opt.id + '_' + opt.split + '.pth')
result_fn = os.path.join('eval_results/', opt.id + '_' + opt.split + '.json')

# Setup the model
from models.AttEnsemble import AttEnsemble

multi_models_list = []
# Setup the model
opt.vocab = vocab
for order in range(2*opt.number_of_models):
    multi_models_list.append(models.setup(opt).cuda())
del opt.vocab
# multi_models = MultiModels(multi_models_list)
multi_models = nn.ModuleList(multi_models_list)
multi_models.load_state_dict(torch.load(opt.model))

if opt.weights is not None:
    opt.weights = [float(_) for _ in opt.weights]
model = AttEnsemble(multi_models_list[opt.number_of_models:2*opt.number_of_models], weights=opt.weights)
model.seq_length = opt.max_length
model.cuda()
model.eval()
crit = utils.LanguageModelCriterion()

# Create the Data Loader instance
if len(opt.image_folder) == 0:
  loader = DataLoader(opt)
else:
  loader = DataLoaderRaw({'folder_path': opt.image_folder, 
                            'coco_json': opt.coco_json,
                            'batch_size': opt.batch_size,
                            'cnn_model': opt.cnn_model})
# When eval using provided pretrained model, the vocab may be different from what you have in your cocotalk.json
# So make sure to use the vocab in infos file.
loader.ix_to_word = infos['vocab']

#opt.id = '+'.join([_+str(__) for _,__ in zip(opt.ids, opt.weights)])

# calculate the std of feature
loader.reset_iterator('val')
nonzero_num = 0
total_num = 0
sum_square = 0
sum_val = 0
for counter in range(100):
    data = loader.get_batch('val')
    fc_feats = data['fc_feats']
    sum_square += (fc_feats**2).sum()
    sum_val +=fc_feats.sum()
    total_num += (fc_feats>=0).sum()
    nonzero_num += (fc_feats>0).sum()
nonzero_std = np.sqrt(sum_square/nonzero_num-(sum_val/nonzero_num)**2)
total_std = np.sqrt(sum_square/total_num-(sum_val/total_num)**2)




# Set sample options
loss, split_predictions, lang_stats = eval_utils.eval_split(model, crit, loader, 
    vars(opt))

print('loss: ', loss)
if lang_stats:
  print(lang_stats)

if opt.dump_json == 1:
    # dump the json
    json.dump(split_predictions, open('vis/vis.json', 'w'))
