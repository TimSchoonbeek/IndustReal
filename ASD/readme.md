# Assembly state detection

## Installation 

First pip install the ultralytics package including all [requirements](https://github.com/ultralytics/ultralytics/blob/main/requirements.txt) in a [**Python>=3.8**](https://www.python.org/) environment with [**PyTorch>=1.8**](https://pytorch.org/get-started/locally/). Please check the ultralytics [repository](https://github.com/ultralytics/ultralytics/tree/main) for more detailed instructions.
```
$ pip install ultralytics
```
Then download the scripts and install additional requirements.
```
$ git clone https://github.com/TimSchoonbeek/IndustReal
$ pip install pylabel
```
Please download the pre-trained COCO model from the ultralytics [repository](https://github.com/ultralytics/ultralytics). The VOC2012 dataset used for mixup data augmentation can be found on [this](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/index.html) page.

## Usage

Three scripts are provided to prepare the synthetic and/or real data for ASD training:
* *create_dataset_synthetic.py*: Creates a synthetic dataset in proper YOLO format from the synthetic data from Unity. Effects as motion blur, occlusions and image mixup can be enabled as data augmentation methods.
* *create_dataset_real.py*: Creates a dataset in proper YOLO format from the IndustReal dataset.
* *create_dataset_hybrid.py*: Creates a new dataset, by combining two YOLO formatted datasets. This is used to make a hybrid dataset of synthetic and real images.

Two scripts are provided for training and validation purposes:
* *train.py*: Runs the training with the correct hyperparameters. If the parameter "name" is not set, the network name is automatically generated.
* *validate.py*: Runs the validation script with the correct hyperparameters and outputs metrics as requested.

To perform ASD and reproduce the results outlined in the paper, follow these steps:
1. Create the datasets
```
python create_dataset_synthetic.py --input_image_folder 'data/synthetic/images/' --data_folder 'datasets/' --name 'synthetic_mixup_occluded' --val_percentage 20 --test_percentage 0 --motion_blur --mixup --occlusions --randomize
python create_dataset_real.py --name 'real_full'
python create_dataset_real.py --name 'real_full_bbpresent' --only_with_bb_present
python create_dataset_hybrid.py --synth_dataset 'synthetic_mixup_occluded' --real_dataset 'real_full' --name 'hybrid_mixup_occluded' --randomize
```

2. Train the models
```
# Train with synthetic dataset, pre-trained on COCO
python train.py --model_name 'pretrain_industreal_synth' --data_folder 'datasets/' --dataset_name 'synthetic_mixup_occluded' --lr 0.0005 --epochs 30 --patience 5 --pretrain_model 'yolov8m.pt'

# Train with real dataset, pre-trained on COCO
python train.py --model_name 'real_full_from_coco' --data_folder 'datasets/' --dataset_name 'real_full' --lr 0.0005 --epochs 50 --patience 5 --pretrain_model 'yolov8m.pt'

# Train with real dataset, pre-trained on industreal synthetic
python train.py --model_name 'real_full_from_industreal_synthetic' --data_folder 'datasets/' --dataset_name 'real_full' --lr 0.0005 --epochs 50 --patience 5 --pretrain_model 'pretrain_industreal_synth.pt'

# Train with hybrid dataset, pre-trained on COCO
python train.py --model_name 'hybrid_full_from_coco' --data_folder 'datasets/' --dataset_name 'synthetic_mixup_occluded' --lr 0.0005 --epochs 50 --patience 5 --pretrain_model 'yolov8m.pt'

```

3. Validate the models on the test sets
```
# Trained on synthetic dataset, pre-trained on COCO
python validate.py --model_path 'runs/detect/pretrain_industreal_synth/weights/best.pt' --data_path 'datasets/real_full.yaml' --phase 'test' --get_metrics
python validate.py --model_path 'runs/detect/pretrain_industreal_synth/weights/best.pt' --data_path 'datasets/real_full_bbpresent.yaml' --phase 'test' --get_metrics

# Trained on real dataset, pre-trained on COCO
python validate.py --model_path 'runs/detect/real_full_from_coco/weights/best.pt' --data_path 'datasets/real_full.yaml' --phase 'test' --get_metrics
python validate.py --model_path 'runs/detect/real_full_from_coco/weights/best.pt' --data_path 'datasets/real_full_bbpresent.yaml' --phase 'test' --get_metrics

# Trained on real dataset, pre-trained on industreal synthetic
python validate.py --model_path 'runs/detect/real_full_from_industreal_synthetic/weights/best.pt' --data_path 'datasets/real_full.yaml' --phase 'test' --get_metrics
python validate.py --model_path 'runs/detect/real_full_from_industreal_synthetic/weights/best.pt' --data_path 'datasets/real_full_bbpresent.yaml' --phase 'test' --get_metrics

# Trained on hybrid dataset, pre-trained on COCO
python validate.py --model_path 'runs/detect/hybrid_full_from_coco/weights/best.pt' --data_path 'datasets/real_full.yaml' --phase 'test' --get_metrics
python validate.py --model_path 'runs/detect/hybrid_full_from_coco/weights/best.pt' --data_path 'datasets/real_full_bbpresent.yaml' --phase 'test' --get_metrics
```
