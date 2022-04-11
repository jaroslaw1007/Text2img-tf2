# Text2img-tf2

A Tensorflow-2.0 version implementation of Stackgan++. <br />
Han Zhang, Tao Xu, Hongsheng Li, Shaoting Zhang, Xiaogang Wang, Xiaolei Huang, Dimitris N. Metaxas: StackGAN++: Realistic Image Synthesis with Stacked Generative Adversarial Networks in TPAMI 2019. [arxiv](https://arxiv.org/abs/1710.10916)

## Architecture

![](https://github.com/jaroslaw1007/Text2img-tf2/blob/main/Architecture.png)

## Dependencies

* [Tensorflow2](https://www.tensorflow.org) >= 2.0.0 

## Datasets

We use images in [Oxford 102 flower](https://www.robots.ox.ac.uk/~vgg/data/flowers/102/) as our training and testing data.
For captions, you can download with this [URL](https://drive.google.com/file/d/0B0ywwgffWnLLcms2WWJQRFNSWXM/view).
To turn the text captions into embedding, we utilize skip-thoughts model. You can find more details in the [repo](https://github.com/ryankiros/skip-thoughts.git).

## Installation

```
git clone https://github.com/jaroslaw1007/Text2img-tf2.git
```

## Training

First you need to go config.py to fill in the directories you place the images, text captions, dictionaries, lookup table.

```
cfg.DATASET.IMAGE_PATH = ...
cfg.DATASET.CAPTION_PATH = ...
cfg.DATASET.LOOK_UP = ...
cfg.DATASET.DICTIONARY_PATH = ...
```

Then

```
python main.py
```

## Demo

![](https://github.com/jaroslaw1007/Text2img-tf2/blob/main/demo.png)

## Citing

```
@article{DBLP:journals/pami/ZhangXLZWHM19,
  author    = {Han Zhang and
               Tao Xu and
               Hongsheng Li and
               Shaoting Zhang and
               Xiaogang Wang and
               Xiaolei Huang and
               Dimitris N. Metaxas},
  title     = {StackGAN++: Realistic Image Synthesis with Stacked Generative Adversarial
               Networks},
  journal   = {{IEEE} Trans. Pattern Anal. Mach. Intell.},
  volume    = {41},
  number    = {8},
  pages     = {1947--1962},
  year      = {2019},
  url       = {https://doi.org/10.1109/TPAMI.2018.2856256},
  doi       = {10.1109/TPAMI.2018.2856256},
  timestamp = {Fri, 26 Feb 2021 08:54:53 +0100},
  biburl    = {https://dblp.org/rec/journals/pami/ZhangXLZWHM19.bib},
  bibsource = {dblp computer science bibliography, https://dblp.org}
}
```

