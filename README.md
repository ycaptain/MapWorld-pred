# MapWorld-pred

<!-- depthFrom=1 depthTo=6 orderedList=false -->

- [Introduction](#Introduction)
- [Initialize Develop Environments](#Initialize-Develop-Environments)
  - [Preparation](#Preparation)
    - [Mandatory](#Mandatory)
    - [Strongly Recommendation](#Strongly-Recommendation)
  - [Installation](#Installation)
  - [Run The Code](#Run-The-Code)
  - [Training](#Training)
  - [Config file format](#Config-file-format)
  - [Test the trained model](#Test-the-trained-model)
  - [Demo](#Demo)
  - [Run Thrift Server](#Run-Thrift-Server)
- [Initialize Workspace](#Initialize-Workspace)
- [Push Your Code](#Push-Your-Code)
  - [Commit Message](#Commit-Message)

<!-- /TOC -->

## Introduction

WIP

## Initialize Develop Environments

- install [conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/), which is an open-source package management system and environment management system.
- [Python](https://www.python.org/downloads/) >= 3.6
- [PyTorch](https://pytorch.org/get-started/locally/) >= 1.2

### Preparation

Create the conda environment.

```bash
conda env create -f env.yaml
conda activate MapWorld
```

Run the following command to regenerate Thrift binding code.

```bash
cd src
rm -r gen-py
thrift --gen py predict.thrift
thrift --gen py ../../app/main.thrift # you need to clone the frontend
```

### Mandatory

WIP

### Strongly-Recommendation

WIP

### Installation

WIP

## Run The Code

### Training

```
usage: train.py [-h] [-c CONFIG] [-r RESUME] [-d DEVICE] [--lr LR] [--bs BS]

PyTorch Template

optional arguments:
  -h, --help            show this help message and exit
  -c CONFIG, --config CONFIG
                        config file path (default: None)
  -r RESUME, --resume RESUME
                        path to latest checkpoint (default: None)
  -d DEVICE, --device DEVICE
                        indices of GPUs to enable (default: all)
  --lr LR, --learning_rate LR
  --bs BS, --batch_size BS

```

Run the following command to start the train process conformed to `configs/building-deeplab.json`.

```bash
python train.py -c configs/building-deeplab.json
```

### Config file format

Config files are in `.json` format:

```javascript
{
    "name": "Building-Deeplab",  // training session name
    "n_gpu": 1,  // number of GPUs to use for training.
    "log_config": "src/logger/logger_config.json",  // path to logger config
    "save_dir": "saved/",  // path to training save directory
    "arch": {
         "type": "DeepLab",  // name of model architecture to train
         "args": {  // required args for the model, see model description
             "num_classes": 2,
             "backbone": "mobilenet"
         }
    },
    "data_loader": {
        "type": "SpaceNetDataLoader",  // selecting data loader
        "args":{
            "data_type": "building",  // data type (building or road)
            "data_name": [  // a set of directories contains the data
                "AOI_2_Vegas_Train",
                "AOI_3_Paris_Train",
                "AOI_4_Shanghai_Train",
                "AOI_5_Khartoum_Train"
            ],
            "data_dir": "data/FYPData/spacenet/buildings",  // dataset path
            "batch_size": 25,  
            "shuffle": true,  // shuffle training data before splitting
            "validation_split": 0.1,  // size of validation dataset. float(portion) or int(number of samples)
            "num_workers": 4,  // number of cpu processes to be used for data loading
            "scales": [0.5, 0.75, 1.0, 1.25, 1.5],  // random select a scale rate during the data loading
            "crop_size": 325  // final image size to the model
        }
    },
    "loss": {  // loss
        "type": "SegmentationLosses",
        "args": {
            "mode": "ce",
            "ignore_index": 255
        }
    },
    "metrics": [  // list of metrics to evaluate
        {
            "type": "SegEvaluator",
            "args": {
                "num_class": 2
            }
        }
    ],
    "optimizer": {
        "type": "SGD",
        "weight_decay": 5.0e-4,
        "momentum": 0.9,
        "lr": 0.001
    },
    "lr_scheduler": {
        "type": "PolynomialLR",  // learning rate scheduler
        "args": {
            "step_size": 10,
            "iter_max": 20000,
            "power": 0.9
        }
    },
    "trainer": {
        "type": "DeeplabTrainer",   // select the trainer for the model
        "args": {},
        "epochs": 300,  // number of training epochs
        "save_period": 20,  // save checkpoints every save_freq epochs
        "verbosity": 2,  // 0: quiet, 1: per epoch, 2: full
        "monitor": "max val_mIoU",  // mode and metric for model performance monitoring. set 'off' to disable.
        "early_stop": 20,  // number of epochs to wait before early stop. set 0 to disable.
        "tensorboard": false  // enable tensorboard visualization
    },
    "tester": {  // options during the test
        "postprocessor": {  // set a post processor for the result
            "type": "DenseCRF",
            "args": {
                "iter_max": 10,
                "pos_w": 3,
                "pos_xy_std": 1,
                "bi_w": 4,
                "bi_xy_std": 67,
                "bi_rgb_std": 3,
                "n_jobs": 8
            }
        },
        "save_logits": true,  // save the middle result
        "crop_size": 325  // final size to the model
    }
}
```

### Using config files

Modify the configurations in `.json` config files, then run:

```bash
python train.py --config configs/building-deeplab.json
```

### Resuming from checkpoints

You can resume from a previously saved checkpoint by:

```bash
python train.py --resume path/to/checkpoint
```

### Using Multiple GPU

You can enable multi-GPU training by setting `n_gpu` argument of the config file to larger number.
If configured to use smaller number of gpu than available, first n devices will be used by default.
Specify indices of available GPUs by cuda environmental variable.

```bash
python train.py --device 2,3 -c config.json
```

This is equivalent to

```bash
CUDA_VISIBLE_DEVICES=2,3 python train.py -c config.py
```

### Test the trained model

```
usage: test.py [-h] [-c CONFIG] [-r RESUME] [-d DEVICE]

PyTorch Test

optional arguments:
  -h, --help            show this help message and exit
  -c CONFIG, --config CONFIG
                        config file path (default: None)
  -r RESUME, --resume RESUME
                        path to latest checkpoint (default: None)
  -d DEVICE, --device DEVICE
                        indices of GPUs to enable (default: all)

```

Use the following command to test on the full dataset.

```bash
python test.py --resume path/to/checkpoint
```

Example result can be:
```
Loading checkpoint: saved/models/Road-UNet/0422_121200/model_best.pth ...
Checkpoint loaded. Resume from epoch 43
{
 'loss': 0.009059005480767381,
 'val_Acc': 0.966295790573112,
 'val_Acc_class': 0.9498229888003393,
 'val_mIoU': 0.9130075975808832,
 'val_FWIoU': 0.9352345624057612
}
```

### Demo
```
usage: demo.py [-h] [-c CONFIG] [-r RESUME] [-d DEVICE] -i IMAGE [IMAGE ...]

PyTorch Demo

optional arguments:
  -h, --help            show this help message and exit
  -c CONFIG, --config CONFIG
                        config file path (default: None)
  -r RESUME, --resume RESUME
                        path to latest checkpoint (default: None)
  -d DEVICE, --device DEVICE
                        indices of GPUs to enable (default: all)
  -i IMAGE [IMAGE ...], --image IMAGE [IMAGE ...]
                        path to image to be processed

```

Use the following code to show demos:

```bash
python test.py -r ...model_best.pth -i [path to images]
```

Then check the results from model saving directory.

### Run Thrift Server

Use the following code to run Thrift server:

```bash
python server_main.py
```

After you see the "Server started." from prompt, you can run unit test:

```bash
python test/SrvTest.py
```

## Initialize Workspace

## Push Your Code

[SourceTree](https://www.sourcetreeapp.com/) must be installed, before committing your code, you should review your code first, compare the current state with the last steady state in history in SourceTree (such as latest master status), make sure all of changes in the commits are your expectations

### Commit Message

All commit messages conform to [Conventional Commits Specification](https://www.conventionalcommits.org/)（[type](https://github.com/conventional-changelog/commitlint/tree/master/%40commitlint/config-conventional#type-enum)）

Run the following command, and follow the tips to finish commit:

```bash
git cz
```
