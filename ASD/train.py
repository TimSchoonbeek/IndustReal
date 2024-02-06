"""
Runs the training with the correct hyperparameters.

If the parameter "name" is not set, the network name is automatically generated.

author: Tim Houben
email: timhouben@gmail.com
date: 25/06/2023
"""

import argparse
import os
from ultralytics import YOLO

def setup_options():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pretrain_model_path', default='pretrained/', type=str, help='Path to the weights of the model')
    parser.add_argument('--model_pretrain_name', default='coco', type=str, help='Model prefix name')
    parser.add_argument('--data_folder', default='datasets/', type=str, help='Input folder for the images to make qualitative results of')
    parser.add_argument('--dataset_name', default='synthetic_vanilla', type=str, help='Dataset Name')
    parser.add_argument('--model_name', default='run_1', type=str, help='Dataset Name')
    parser.add_argument('--continue_train', action='store_true', help='When set the is resumed from last epoch.')
    parser.add_argument('--pretrain_model', default='yolov8n.pt', type=str, help='Filename of the pretrained model to use')
    parser.add_argument('--lr', type=float, default=0.01, help='learning rate initial')
    parser.add_argument('--batchsize', type=int, default=-1, help='batchsize, -1 for auto')
    parser.add_argument('--optimizer', default='Adam', type=str, help='SGD or Adam')
    parser.add_argument('--epochs', type=int, default=100, help='Nr of epochs')
    parser.add_argument('--patience', type=int, default=10, help='Nr of epochs to wait before stopping early')
    parser.add_argument('--warmup_epochs', type=float, default=0.5, help='Nr of epochs to wait before stopping early')
    parser.add_argument('--dropout', type=float, default=0.0, help='Dropout rate')
    parser.add_argument('--save_period', type=int, default=1, help='Save Interval')
    parser.add_argument('--image_size', type=int, default=640, help='Save Interval')
    parser.add_argument('--translate', type=float, default=0.0, help='Translate augmentation')
    parser.add_argument('--scale', type=float, default=0.2, help='Scale augmentation')
    parser.add_argument('--fliplr', type=float, default=0.0, help='Flip left/right augmentation')
    parser.add_argument('--mosaic', type=float, default=0.0, help='Mosaic augmentation')
    parser.add_argument('--cosine_scheduler', action='store_true', help='When set the cosine scheduler is active.')
    parser.add_argument('--resume', action='store_true', help='When set the training is resumed from last checkpoint.')
    return parser.parse_args()


if __name__ == "__main__":
    args = setup_options()
    
    if not args.model_name == '':
        name = args.model_name
    else:
        name = "run_ds_" + str(args.dataset_name) + "_pt_" + args.model_pretrain_name + "_" + args.pretrain_model[:-3] + "_lr_" + str(learning_rate) + "_epochs_" + str(num_epochs)

    print("Starting to train: " + name)
        
    model = YOLO(os.path.join(args.pretrain_model_path, args.pretrain_model))
    model.train(data=args.data_folder + args.dataset_name + ".yaml", lr0=args.lr, batch=args.batchsize, optimizer=args.optimizer, epochs=args.epochs, name=name, warmup_epochs=args.warmup_epochs, patience=args.patience, resume=args.resume, dropout=args.dropout, save_period=args.save_period, translate=args.translate, scale=args.scale, fliplr=args.fliplr, mosaic=args.mosaic, imgsz=args.image_size)
