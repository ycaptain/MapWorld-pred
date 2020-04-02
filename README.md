# MapWorld-pred

<!-- depthFrom=1 depthTo=6 orderedList=false -->

- [Introduction](#Introduction)
- [Initialize Develop Environments](#Initialize-Develop-Environments)
  - [Preparation](#Preparation)
    - [Mandatory](#Mandatory)
    - [Strongly Recommendation](#Strongly-Recommendation)
  - [Installation](#Installation)
  - [Run The Code](#Run-The-Code)
  - [Config file format](#Config-file-format)
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

WIP

### Mandatory

WIP

### Strongly-Recommendation

WIP

### Installation

WIP

## Run The Code

Run the following command to start the train process conformed to `config.json`.

```bash
python train.py -c config.json
```

Run the following command to generate Thrift binding code.

```bash
cd src
thrift --gen py service.thrift
```

### Config file format

Config files are in `.json` format:

```javascript
{
  "name": "Mnist_LeNet",        // training session name
  "n_gpu": 1,                   // number of GPUs to use for training.
  
  "arch": {
    "type": "MnistModel",       // name of model architecture to train
    "args": {

    }                
  },
  "data_loader": {
    "type": "MnistDataLoader",         // selecting data loader
    "args":{
      "data_dir": "data/",             // dataset path
      "batch_size": 64,                // batch size
      "shuffle": true,                 // shuffle training data before splitting
      "validation_split": 0.1          // size of validation dataset. float(portion) or int(number of samples)
      "num_workers": 2,                // number of cpu processes to be used for data loading
    }
  },
  "optimizer": {
    "type": "Adam",
    "args":{
      "lr": 0.001,                     // learning rate
      "weight_decay": 0,               // (optional) weight decay
      "amsgrad": true
    }
  },
  "loss": "nll_loss",                  // loss
  "metrics": [
    "accuracy", "top_k_acc"            // list of metrics to evaluate
  ],                         
  "lr_scheduler": {
    "type": "StepLR",                  // learning rate scheduler
    "args":{
      "step_size": 50,          
      "gamma": 0.1
    }
  },
  "trainer": {
    "epochs": 100,                     // number of training epochs
    "save_dir": "saved/",              // checkpoints are saved in save_dir/models/name
    "save_freq": 1,                    // save checkpoints every save_freq epochs
    "verbosity": 2,                    // 0: quiet, 1: per epoch, 2: full
  
    "monitor": "min val_loss"          // mode and metric for model performance monitoring. set 'off' to disable.
    "early_stop": 10	                 // number of epochs to wait before early stop. set 0 to disable.
  
    "tensorboard": true,               // enable tensorboard visualization
  }
}
```

### Using config files

Modify the configurations in `.json` config files, then run:

```bash
python train.py --config config.json
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

### Initialize Workspace

### Push Your Code

[SourceTree](https://www.sourcetreeapp.com/) must be installed, before committing your code, you should review your code first, compare the current state with the last steady state in history in SourceTree (such as latest master status), make sure all of changes in the commits are your expectations

#### Commit Message

All commit messages conform to [Conventional Commits Specification](https://www.conventionalcommits.org/)（[type](https://github.com/conventional-changelog/commitlint/tree/master/%40commitlint/config-conventional#type-enum)）

Run the following command, and follow the tips to finish commit:

```bash
git cz
```
