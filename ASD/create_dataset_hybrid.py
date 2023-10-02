"""
Creates a new dataset, by combinding two YOLO formatted datasets.
This is used to make a hybrid dataset of synthetic and real images.

author: Tim Houben
email: timhouben@gmail.com
date: 25/06/2023
"""

import argparse
import os
import numpy as np
import glob
from tqdm import tqdm
import shutil

def setup_options():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_folder', default='datasets/', type=str, help='Output folder, folder of the location of the dataset')
    parser.add_argument('--synth_dataset', default='synthetic_rgb_mb_16', type=str, help='Synthetic Dataset name')
    parser.add_argument('--real_dataset', default='exp_full_rgb', type=str, help='Real dataset name')
    parser.add_argument('--name', default='combined_rgb', type=str, help='Dataset name')
    parser.add_argument('--randomize', action='store_true', help='Randomize the order of the dataset items when set.')
    parser.add_argument('--seed', default=1, type=int, help='Seed value for data randomization')
    parser.add_argument('--exclude_synthetic_val', action='store_true', help='Exclude synthetic data from validation set.')
    parser.add_argument('--exclude_synthetic_test', action='store_true', help='Exclude synthetic data from test set.')
    parser.add_argument('--only_labels', action='store_true', help='When set only the labels are recreated.')
    return parser.parse_args()

def create_dir(dir: str):
    """Create a directory if one does not exist.

    Args:
        dir (str): Directory to be created.
    """
    if not os.path.exists(dir):
        os.makedirs(dir)


def create_dataset(args: argparse.Namespace, phase: str):
    """Creates a dataset in a certain folder.

    Args:
        args (argparse.Namespace): Command line arguments.
        phase (str): Part of the dataset that needs to be constructed, usually train, val or test.
        
    Raises:
        ValueError: If unsupported simulation mode is specified in the command line arguments.
    """
    dataset_items = []
    label_items = []
    if not ( (phase == 'val' and args.exclude_synthetic_val) or (phase == 'test' and args.exclude_synthetic_test) ):
        synth_image_path = os.path.join(args.data_folder, args.synth_dataset, 'images', phase)
        synth_labels_path = os.path.join(args.data_folder, args.synth_dataset, 'labels', phase)
        for filename in glob.glob(synth_image_path + '/*.png'):
            dataset_items.append(filename)
            labelfile = os.path.join(synth_labels_path, os.path.basename(filename)[:-4] + ".txt") #make path for corresponding label file
            label_items.append(labelfile)
    
    exp_image_path = os.path.join(args.data_folder, args.real_dataset, 'images', phase)
    exp_labels_path = os.path.join(args.data_folder, args.real_dataset, 'labels', phase)
    for filename in glob.glob(exp_image_path + '/*.png'):
        dataset_items.append(filename)
        labelfile = os.path.join(exp_labels_path, os.path.basename(filename)[:-4] + ".txt") #make path for corresponding label file
        label_items.append(labelfile)
    
    dataset_ids = np.arange(0, len(dataset_items), 1).astype(int)
    if args.randomize:
        dataset_ids = dataset_ids[np.random.RandomState(seed=seed).permutation(len(dataset_items))]

    dataset_folder = os.path.join(args.data_folder, args.name) 
    image_folder = os.path.join(args.data_folder, args.name, "images", phase)
    label_folder = os.path.join(args.data_folder, args.name, "labels", phase)
    create_dir(image_folder)
    create_dir(label_folder)

    filenamelist = []
    for entry_no in tqdm(dataset_ids):
        if not args.only_labels:
            shutil.copyfile(dataset_items[entry_no], os.path.join(image_folder, os.path.basename(dataset_items[entry_no])))
        shutil.copyfile(label_items[entry_no], os.path.join(label_folder, os.path.basename(label_items[entry_no])))

        filenamelist.append('./images/' + phase + '/' + os.path.basename(dataset_items[entry_no]))
        

    pathfile = open(os.path.join(dataset_folder, phase + '.txt'),"w") #write mode 
    for entry in filenamelist:
        pathfile.write(entry + '\n')
    pathfile.close()
    

if __name__ == "__main__":
    args = setup_options()
    seed = args.seed

    print("Creating Synthetic Dataset")
    create_dir(os.path.join(args.data_folder, args.name))
    create_dir(os.path.join(args.data_folder, args.name, "images"))
    create_dir(os.path.join(args.data_folder, args.name, "labels"))
    create_dir(os.path.join(args.data_folder, args.name, "annotations"))

    with open(os.path.join(args.data_folder, "template.yaml"), 'r') as file :
        filedata = file.read()
    filedata = filedata.replace('template', args.name)
    with open(os.path.join(args.data_folder, args.name + ".yaml"), 'w') as file:
        file.write(filedata)

    print("Adding to Train set...")
    create_dataset(args, 'train')

    print("Adding to Validation set...")
    create_dataset(args, 'val')

    print("Adding to Test set...")
    create_dataset(args, 'test')

    print("Done!")  


