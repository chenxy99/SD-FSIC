# Self-Distillation for Few-Shot Image Captioning (SD-FSIC)

This code implements the Self-Distillation for Few-Shot Image Captioning.

Reference
------------------
If you use our code or data, please cite our paper:
```text
Anonymous submission for NeurIPS 2020, paper ID 696.
```

Disclaimer
------------------
We adopt the pytorch implementation for Self-critical Sequence Training for Image Captioning [`self-critical.pytorch`](https://github.com/ruotianluo/self-critical.pytorch) as a baseline model for few-shot image captioner. We use the features provided in this repository. Please refer to these links for further README information.

Requirements
------------------
- Python 2 or 3 ([coco-caption](https://github.com/ruotianluo/coco-caption) supports python 3)
- PyTorch 1.3 (along with torchvision)
- cider ([cider](https://github.com/ruotianluo/cider/tree/e9b736d038d39395fa2259e39342bb876f1cc877)) (Download it in current folder `SD-FSIC/`)
- coco-caption ([coco-caption](https://github.com/ruotianluo/coco-caption/tree/ea20010419a955fed9882f9dcc53f2dc1ac65092)) (**Remember to follow initialization steps in coco-caption/README.md**) (Download it in current folder `SD-FSIC/`)
- yacs
- I also provide the conda enviroment [sc_rtl.yml](https://github.com/chenxy99/SD-FSIC/blob/master/sc_rtl.yml), you can directly run
```bash
$ conda env create -f sc_rtl.yml
```
to create the same enviroment where I succesfully run my code.

Datasets
------------------
One can follow the instructions in [data/README.md](data/README.md) to create the corresponding data. More specifically, we download the preprocessed file or preextracted features from [link](https://drive.google.com/drive/folders/1eCdz62FAVCGogOuNhy87Nmlo5_I0sH2J).
You need to download as least the following files, unzip them and put them in the `data` folder.
- coco-train-idxs.p
- coco-train-words.p
- cocotalk_label.h5
- cocotalk.json
- dataset_coco.json
- cocotalk_fc.zip
- cocotalk_att.zip

Train your own network on few-shot semi-supervised COCO dataset
------------------
### Start training
```bash
$ python train.py --cfg configs/fc.yml --id sd-fsic
```
The train script will dump checkpoints into the folder specified by `--checkpoint_path` (default = `log_$id/`). You can set the corresponding hyper-parameters in `configs/fc.yml`.

- `--paired_percentage` The percentage of the training set, where the images and sentences are paired.
- `--language_pretrain_epoch` The number of epochs used to pretrain the model.
- `--paired_train_epoch` The number of epochs used to train the model with image-caption pairs.
- `--random_seed` The seed used to select the image-caption pairs.
- `--alpha` The smoothing coefficient for Mean Teacher.
- `--hyper_parameter_lambda_x` The hyper-parameter to balance the unsupervised items for total loss.
- `--hyper_parameter_lambda_y` The hyper-parameter to balance the unsupervised items for total loss.
- `--std_pseudo_visual_feature` The hyper-parameter for the standard deviation of pseudo visual feature.
- `--number_of_models` The number of base models for model ensemble.
- `--inner_iteration` The number of the total iteration of the inner optimization to generate pseudo latent feature.

We also provide the corresponding results of the COCO test set in [`sd-fsic.json`](https://drive.google.com/open?id=1wgfJ8cVVxOmmWAA23an7l_X7Sq-ovpRV).

The corresponding results are listed below
<table>
  <tr>
    <th>BLEU-1</th><th>BLEU-2</th>
    <th>BLEU-3</th><th>BLEU-4</th>
    <th>METEOR</th><th>ROUGE_L</th><th>CIDEr</th><th>SPICE</th><th>WMD</th>
  </tr>
  <tr>
    <td>64.5</td><td>45.9</td>
    <td>32.1</td><td>22.5</td>
    <td>20.0</td><td>46.7</td><td>62.4</td><td>12.7</td><td>14.7</td>
  </tr>
</table>















Download the extra nocaps [dataset](https://drive.google.com/file/d/1puVmZN_UbDYas9m2c1cbBx7m9SMvgfTG/view?usp=sharing) that is not provided by [`nocaps`](https://github.com/nocaps-org/updown-baseline) and unzip it. (Remenber to download other documents by the [instruction](https://nocaps.org/updown-baseline/setup_dependencies.html))

This extra human saliency data for `COCO` and `nocaps` dataset is extracted by [Saliency Attentive Model](https://arxiv.org/pdf/1611.09571.pdf) and the detection results for `COCO` dataset are extracted by the [open image detector](https://github.com/nocaps-org/image-feature-extractors).












This repository includes the unofficial implementation [Self-critical Sequence Training for Image Captioning](https://arxiv.org/abs/1612.00563) and [Bottom-Up and Top-Down Attention for Image Captioning and Visual Question Answering](https://arxiv.org/abs/1707.07998).

The author of SCST helped me a lot when I tried to replicate the result. Great thanks. The att2in2 model can achieve more than 1.20 Cider score on Karpathy's test split (with self-critical training, bottom-up feature, large rnn hidden size, without ensemble)

This is based on my [ImageCaptioning.pytorch](https://github.com/ruotianluo/ImageCaptioning.pytorch) repository. The modifications is:
- Self critical training.
- Bottom up feature support from [ref](https://arxiv.org/abs/1707.07998). (Evaluation on arbitrary images is not supported.)
- Ensemble
- Multi-GPU training
- Add transformer (merged from [Transformer_captioning](https://github.com/ruotianluo/Transformer_Captioning))

## Requirements
Python 2 or 3 (My [coco-caption](https://github.com/ruotianluo/coco-caption) supports python 3)

PyTorch 1.3 (along with torchvision)

cider (already been added as a submodule)

coco-caption (already been added as a submodule) (**Remember to follow initialization steps in coco-caption/README.md**)

yacs

(**Skip if you are using bottom-up feature**): If you want to use resnet to extract image features, you need to download pretrained resnet model for both training and evaluation. The models can be downloaded from [here](https://drive.google.com/open?id=0B7fNdx_jAqhtbVYzOURMdDNHSGM), and should be placed in `data/imagenet_weights`.

## Pretrained models

Checkout [MODEL_ZOO.md](MODEL_ZOO.md).

If you want to do evaluation only, you can then follow [this section](#generate-image-captions) after downloading the pretrained models (and also the pretrained resnet101 or precomputed bottomup features).

## Train your own network on COCO/Flickr30k

### Prepare data.

We now support both flickr30k and COCO. See details in [data/README.md](data/README.md). (Note: the later sections assume COCO dataset; it should be trivial to use flickr30k.)

### Start training

```bash
$ python train.py --id fc --caption_model newfc --input_json data/cocotalk.json --input_fc_dir data/cocotalk_fc --input_att_dir data/cocotalk_att --input_label_h5 data/cocotalk_label.h5 --batch_size 10 --learning_rate 5e-4 --learning_rate_decay_start 0 --scheduled_sampling_start 0 --checkpoint_path log_fc --save_checkpoint_every 6000 --val_images_use 5000 --max_epochs 30
```

or 

```bash
$ python train.py --cfg configs/fc.yml --id fc
```

The train script will dump checkpoints into the folder specified by `--checkpoint_path` (default = `log_$id/`). By default only save the best-performing checkpoint on validation and the latest checkpoint to save disk space. You can also set `--save_history_ckpt` to 1 to save every checkpoint.

To resume training, you can specify `--start_from` option to be the path saving `infos.pkl` and `model.pth` (usually you could just set `--start_from` and `--checkpoint_path` to be the same).

To checkout the training curve or validation curve, you can use tensorboard. The loss histories are automatically dumped into `--checkpoint_path`.

The current command use scheduled sampling, you can also set `--scheduled_sampling_start` to -1 to turn off scheduled sampling.

If you'd like to evaluate BLEU/METEOR/CIDEr scores during training in addition to validation cross entropy loss, use `--language_eval 1` option, but don't forget to pull the submodule `coco-caption`.

For all the arguments, you can specify them in a yaml file and use `--cfg` to use the configurations in that yaml file. The configurations in command line will overwrite cfg file if there are conflicts.  

For more options, see `opts.py`. 

<!-- **A few notes on training.** To give you an idea, with the default settings one epoch of MS COCO images is about 11000 iterations. After 1 epoch of training results in validation loss ~2.5 and CIDEr score of ~0.68. By iteration 60,000 CIDEr climbs up to about ~0.84 (validation loss at about 2.4 (under scheduled sampling)). -->

### Train using self critical

First you should preprocess the dataset and get the cache for calculating cider score:
```
$ python scripts/prepro_ngrams.py --input_json data/dataset_coco.json --dict_json data/cocotalk.json --output_pkl data/coco-train --split train
```

Then, copy the model from the pretrained model using cross entropy. (It's not mandatory to copy the model, just for back-up)
```
$ bash scripts/copy_model.sh fc fc_rl
```

Then
```bash
$ python train.py --id fc_rl --caption_model newfc --input_json data/cocotalk.json --input_fc_dir data/cocotalk_fc --input_att_dir data/cocotalk_att --input_label_h5 data/cocotalk_label.h5 --batch_size 10 --learning_rate 5e-5 --start_from log_fc_rl --checkpoint_path log_fc_rl --save_checkpoint_every 6000 --language_eval 1 --val_images_use 5000 --self_critical_after 30 --cached_tokens coco-train-idxs --max_epoch 50 --train_sample_n 5
```

or 
```bash
$ python train.py --cfg configs/fc_rl.yml --id fc_rl
```


You will see a huge boost on Cider score, : ).

**A few notes on training.** Starting self-critical training after 30 epochs, the CIDEr score goes up to 1.05 after 600k iterations (including the 30 epochs pertraining).

## Generate image captions

### Evaluate on raw images

**Note**: this doesn't work for models trained with bottomup feature.
Now place all your images of interest into a folder, e.g. `blah`, and run
the eval script:

```bash
$ python eval.py --model model.pth --infos_path infos.pkl --image_folder blah --num_images 10
```

This tells the `eval` script to run up to 10 images from the given folder. If you have a big GPU you can speed up the evaluation by increasing `batch_size`. Use `--num_images -1` to process all images. The eval script will create an `vis.json` file inside the `vis` folder, which can then be visualized with the provided HTML interface:

```bash
$ cd vis
$ python -m SimpleHTTPServer
```

Now visit `localhost:8000` in your browser and you should see your predicted captions.

### Evaluate on Karpathy's test split

```bash
$ python eval.py --dump_images 0 --num_images 5000 --model model.pth --infos_path infos.pkl --language_eval 1 
```

The defualt split to evaluate is test. The default inference method is greedy decoding (`--sample_method greedy`), to sample from the posterior, set `--sample_method sample`.

**Beam Search**. Beam search can increase the performance of the search for greedy decoding sequence by ~5%. However, this is a little more expensive. To turn on the beam search, use `--beam_size N`, N should be greater than 1.

### Evaluate on COCO test set

```bash
$ python eval.py --input_json cocotest.json --input_fc_dir data/cocotest_bu_fc --input_att_dir data/cocotest_bu_att --input_label_h5 none --num_images -1 --model model.pth --infos_path infos.pkl --language_eval 0
```

You can download the preprocessed file `cocotest.json`, `cocotest_bu_att` and `cocotest_bu_fc` from [link](https://drive.google.com/open?id=1eCdz62FAVCGogOuNhy87Nmlo5_I0sH2J).

## Miscellanea
**Using cpu**. The code is currently defaultly using gpu; there is even no option for switching. If someone highly needs a cpu model, please open an issue; I can potentially create a cpu checkpoint and modify the eval.py to run the model on cpu. However, there's no point using cpus to train the model.

**Train on other dataset**. It should be trivial to port if you can create a file like `dataset_coco.json` for your own dataset.

**Live demo**. Not supported now. Welcome pull request.

## For more advanced features:

Checkout [ADVANCED.md](ADVANCED.md).

## Reference

If you find this repo useful, please consider citing (no obligation at all):

```
@article{luo2018discriminability,
  title={Discriminability objective for training descriptive captions},
  author={Luo, Ruotian and Price, Brian and Cohen, Scott and Shakhnarovich, Gregory},
  journal={arXiv preprint arXiv:1803.04376},
  year={2018}
}
```

Of course, please cite the original paper of models you are using (You can find references in the model files).

## Acknowledgements

Thanks the original [neuraltalk2](https://github.com/karpathy/neuraltalk2) and awesome PyTorch team.
