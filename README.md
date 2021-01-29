# Frame Difference-Based Temporal Loss
Jianjin Xu, Zheyang Xiong, Xiaolin Hu, 2021

## Installation

1. Environment: Pytorch == 1.1.0

2. Download [DAVIS](https://davischallenge.org/) dataset and [MSCOCO-train2014](https://cocodataset.org/) dataset and place it under the `data` directory. Download the test data used in the paper from [TODO] []() or you can use your own data. Make sure the folder contains

```
data/DAVIS/train/JPEGImages/480p
data/mscoco/train2014
data/testin
```

3. Make sure the folder organization is as the following:

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

4. Install DeepFlow and estimate optic flow following the instructions from [https://github.com/manuelruder/artistic-videos](https://github.com/manuelruder/artistic-videos)


## Usage

Using the `run.py` script, the experiments in the paper can be easily reproduced. For its detailed usage, please refer to the arguments specification of `run.py`.

Here are the steps to reproduce the experiments in the paper:

1. Train the SFN without temporal loss. These models are baselines themselves, and will be finetuned by different temporal losses in the next few steps.

```bash
python run.py train --temp-loss none
```

2. Train SFN with different temporal losses.

```bash
# SFN trained with the P-FDB loss
python run.py train --temp-loss p-fdb
# SFN trained with the C-FDB loss
python run.py train --temp-loss c-fdb
# SFN trained with the OFB loss
python run.py train --temp-loss ofb
```

3. Train RNN with different temporal losses.

```bash
# SFN trained with the P-FDB loss
python run.py train --temp-loss p-fdb --model rnn
# SFN trained with the C-FDB loss
python run.py train --temp-loss c-fdb --model rnn
# SFN trained with the OFB loss
python run.py train --temp-loss ofb --model rnn
```

4. Stylize videos. Using the `run.py` script, the models will be evaluated using the test data in `data/testin` and put the raw images in `data/testout`. The generated videos are properly re-named and put in the `download` folder.

```bash
# evaluation of SFN models
python run.py eval --temp-loss p-fdb
python run.py eval --temp-loss c-fdb
python run.py eval --temp-loss ofb
python run.py eval --temp-loss none
# evaluation of RNN models
python run.py eval --temp-loss p-fdb --model rnn
python run.py eval --temp-loss c-fdb --model rnn
python run.py eval --temp-loss ofb --model rnn
```

## Acknowledgement

Part of this project is based on a pytorch implementation of fast neural style: [https://github.com/abhiskk/fast-neural-style](https://github.com/abhiskk/fast-neural-style).

The optic flow visualization is adapted from [https://github.com/tomrunia/OpticalFlow_Visualization](https://github.com/tomrunia/OpticalFlow_Visualization).