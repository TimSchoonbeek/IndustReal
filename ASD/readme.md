# Assembly state detection

## Installation 

First pip install the ultralytics package including all [requirements](https://github.com/ultralytics/ultralytics/blob/main/requirements.txt) in a [**Python>=3.8**](https://www.python.org/) environment with [**PyTorch>=1.8**](https://pytorch.org/get-started/locally/). Please check the ultralytics [repository](https://github.com/ultralytics/ultralytics/tree/main) for more detailed instructions.
```
pip install ultralytics
```
Then download the scripts and install additional requirements.
```
$ git clone https://github.com/TimSchoonbeek/IndustReal/ASD
$ pip install pylabel
```

## Usage

Three scripts are provided to prepare the synthetic and/or real data for ASD training:
* *create_dataset_synthetic.py*: Creates a synthetic dataset in proper YOLO format from the synthetic data from Unity. Effects as motion blur, occlusions and image mixup can be anabled as data augmentation methods.
* *create_dataset_real.py*: Creates a dataset in proper YOLO format from the IndustReal dataset.
* *create_dataset_hybrid.py*: Creates a new dataset, by combinding two YOLO formatted datasets. This is used to make a hybrid dataset of synthetic and real images.

Two scripts are provided for training and validation purposes:
* *train.py*: Runs the training with the correct hyperparameters. If the parameter "name" is not set, the network name is automatically generated.
* *validate.py*: Runs the validation script with the correct hyperparameters and outputs metrics as requested.

To perform ASD and reproduce the results outlined in the paper, follow these steps:
1. 
2. 
3. 
