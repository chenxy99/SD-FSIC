# Self-Distillation for Few-Shot Image Captioning (SD-FSIC)

This code implements the Self-Distillation for Few-Shot Image Captioning.

Reference
------------------
If you use our code or data, please cite our paper:
```text
Anonymous submission for WACV 2021, paper ID 971.
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

Evaluate on Karpathy's test split
------------------
We provide the corresponding results of the COCO test set in [`sd-fsic.json`](https://drive.google.com/file/d/1UG9QsPraJepTgC-fj9lsROefXmMqj9CT/view?usp=sharing).

Furthermore, we also provide the pretrained model for the above results. You can download this pretrained model [`log_sd-fsic.zip`](https://drive.google.com/file/d/1aOQyj6SF2uJLUaKOco3wdgHRvYtGba1K/view?usp=sharing) and unzip it to current folder `SD-FSIC/`. Then you can run the following script to evaluate our model on Karpathy's test split.

```bash
$ python multi_eval_ensemble.py \
--dump_images 0 \
--num_images 5000 \
--beam_size 5 \
--language_eval 1 \
--model log_sd-fsic/model-best.pth \
--infos_path log_sd-fsic/infos_sd-fsic-best.pkl
```

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

