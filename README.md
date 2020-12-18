# Change Detection Laboratory

Yet another project for benchmarking deep learning-based remote sensing change detection methods.

## Prerequisites

> opencv-python==4.1.1  
  pytorch==1.3.1  
  torchvision==0.4.2  
  pyyaml==5.1.2  
  scikit-image==0.15.0  
  scikit-learn==0.21.3  
  scipy==1.3.1  
  tqdm==4.35.0

Tested using Python 3.7.4 on Ubuntu 16.04.

## Get Started

First clone this repo from github.

```bash
git clone git@github.com:Bobholamovic/CDLab.git
cd CDLab
mkdir exp
cd src
```

In `src/constants.py`, change the dataset locations to your own.

### Model Training

To train a model from scratch, use

```bash
python train.py train --exp_config PATH_TO_CONFIG_FILE
```

A few configuration files regarding different datasets and models are provided in the `configs/` folder for ease of use. 

As soon as the program starts and prints out the configurations, there will be a prompt asking you to write some notes. What you write will be recorded into the log file to help you remember what you did, or you can simply skip this step by pressing `Enter`.

To resume training from some checkpoint, run the code with the `--resume` argument.

```bash
python train.py train --exp_config PATH_TO_CONFIG_FILE --resume PATH_TO_CHECKPOINT
```

Some other optional arguments include:

- `anew`: Add it if the checkpoint is just used to initialize model weights. Note that loading an incompatible checkpoint is supported as a feature, which is useful when you are trying to utilize a well pretrained model for finetuning.
- `save_on`: By default, an epoch-based trainer is used for training. At the end of each training epoch, the trainer evaluates the model on the validation dataset. If you want to save the model output during validation, add this argument.
- `log_off`: Disable logging.
- `tb_on`: Enable tensorboard summaries.
- `debug_on`: Useful when you are debugging your own code. In debugging mode, no checkpoint or model output will be written to disk. In addition, a breakpoint will be set before the program exits if some unhandled exception occurs, which allows you to check variables in the stack or do some cleanup jobs.

During or after the training process, you can check the model weight files in `exp/DATASET_NAME/weights/`, the log files in `exp/DATASET_NAME/logs`, and the output change maps in `exp/DATASET_NAME/out`.

### Model Evaluation

To evaluate a model, use

```bash
python train.py eval --exp_config PATH_TO_CONFIG_FILE --resume PATH_TO_CHECKPOINT --save_on
```

## Supported Architectures

Architecture | Name | Link
:-:|:-:|:-:
CDNet | `CDNet` | [paper](https://doi.org/10.1007/s10514-018-9734-5)
FC-EF | `UNet` | [paper](https://ieeexplore.ieee.org/abstract/document/8451652)
FC-Siam-conc | `SiamUNet_conc` | [paper](https://ieeexplore.ieee.org/abstract/document/8451652)
FC-Siam-diff | `SiamUNet_diff` | [paper](https://ieeexplore.ieee.org/abstract/document/8451652)

## Supported Datasets

Dataset | Name | Link
:-:|:-:|:-:
SZTAKI AirChange Benchmark set: Szada set | `AC_Szada` | [source](http://web.eee.sztaki.hu/remotesensing/airchange_benchmark.html)
SZTAKI AirChange Benchmark set: Tiszadob set | `AC_Tiszadob` | [source](http://web.eee.sztaki.hu/remotesensing/airchange_benchmark.html)
Onera Satellite Change Detection dataset | `OSCD` | [source](https://rcdaudt.github.io/oscd/)
Synthetic images and real season-varying remote sensing images | `Lebedev` | [source](https://drive.google.com/file/d/1GX656JqqOyBi_Ef0w65kDGVto-nHrNs9)

## License

This project is released under the [Unlicense](/LICENSE).