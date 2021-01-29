# Frame Difference-Based Temporal Loss
Jianjin Xu, Zheyang Xiong, Xiaolin Hu, 2021

## Installation

1. Download ![DAVIS](https://davischallenge.org/) dataset and ![MSCOCO-train2014](https://cocodataset.org/) dataset and place it under the `data` directory. Make sure the folder contains

```
data/DAVIS/train/JPEGImages/480p
data/mscoco/train2014
```

2. Folder organization is as the following:

```
data
`- DAVIS
`- mscoco
`- styles
`- testin
`- testout
download
exprs
pretrained
`- vgg16.weight
```

3. Install DeepFlow and estimate optic flow following the instructions from ![https://github.com/manuelruder/artistic-videos](https://github.com/manuelruder/artistic-videos)


## 

TODO
1. vgg download method
4. give credit to fast-neural-style
5. give credit to deepflow

## Acknowledgement

Part of this project is based on a pytorch implementation of fast neural style: ![https://github.com/abhiskk/fast-neural-style](https://github.com/abhiskk/fast-neural-style).