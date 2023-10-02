"""
Creates a synthetic dataset in proper YOLO format from the synthetic data from Unity.
Effects as motion blur, occlusions and image mixup can be anabled as data augmentation methods.

author: Tim Houben
email: timhouben@gmail.com
date: 25/06/2023
"""

import argparse
import os
import glob
import numpy as np
from PIL import Image
from tqdm import tqdm
import cv2
import random

def setup_options():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_image_folder', default='data/synthetic/images/', type=str, help='Input folder for the images to make a dataset from')
    parser.add_argument('--input_labels', default='data/synthetic/training/labels/', type=str, help='Input folder for the labels labels to make a dataset from')
    parser.add_argument('--data_folder', default='datasets/', type=str, help='Output folder, folder of the location of the dataset')
    parser.add_argument('--name', default='synthetic_01', type=str, help='Dataset name')
    parser.add_argument('--randomize', action='store_true', help='Randomize the order of the dataset items when set.')
    parser.add_argument('--seed', default=1, type=int, help='Seed value for data randomization')
    parser.add_argument('--val_percentage', default=20, type=float, help='Percentage of the data that needs to be in the validation set.')
    parser.add_argument('--test_percentage', default=0, type=float, help='Percentage of the data that needs to be in the test set.')
    parser.add_argument('--motion_blur', action='store_true', help='When set motion blur is applied.')
    parser.add_argument('--motion_blur_max_kernelsize', type=int, default=16, help='sharpen value to apply in predefined kernel array')
    parser.add_argument('--occlusions', action='store_true', help='When set, occlusions are added to the dataset.')
    parser.add_argument('--mixup', action='store_true', help='When set, mixup is performed with specific image folder.')
    parser.add_argument('--mixup_images_folder', default='datasets/VOCdevkit/VOC2012/JPEGImages/', type=str, help='Input folder for the mixup images')
    parser.add_argument('--how_many', type=int, default=float("inf"), help='How many test images to use in the dataset')
    parser.add_argument('--how_many_offset', type=int, default=0, help='Offset of the section to be made, can be used in conjuction with how_many')
    return parser.parse_args()

# Convert IDs to IndustReal format
mapdict = {
    0: 0,
    1: 15,
    2: 12,
    3: 2,
    4: 19,
    5: 22,
    6: 14,
    7: 20,
    8: 3,
    9: 11,
    10: 8,
    11: 4,
    12: 10,
    13: 18,
    14: 1,
    15: 7,
    16: 9,
    17: 21,
    18: 13,
    19: 5,
    20: 6,
    21: 17,
    22: 16,
    23: 23
}

def create_dir(dir: str):
    """Create a directory if one does not exist.

    Args:
        dir (str): Directory to be created.
    """
    if not os.path.exists(dir):
        os.makedirs(dir)


def create_dataset(args: argparse.Namespace, phase: str, dataset_items, label_items, dataset_ids):
    """Creates a dataset in a certain folder given certain indexes.

    Args:
        args (argparse.Namespace): Command line arguments.
        phase (str): Part of the dataset that needs to be constructed, usually train, val or test
        dataset_items (list): List of all the possible items to make the dataset from
        label_items (list): List of all the possible labels to make the dataset from
        dataset_ids (numpy.array): Array of indexes.
    Raises:
        ValueError: If unsupported simulation mode is specified in the command line arguments.
    """
    dataset_folder = os.path.join(args.data_folder, args.name) 
    image_folder = os.path.join(args.data_folder, args.name, "images", phase)
    label_folder = os.path.join(args.data_folder, args.name, "labels", phase)
    create_dir(image_folder)
    create_dir(label_folder)

    if args.mixup:
        mixup_items = []
        for filename in glob.glob(args.mixup_images_folder + '*.jpg'):
            mixup_items.append(filename)

    filenamelist = []
    flipcoin_occ = 0
    for newindex, index in enumerate(tqdm(dataset_ids)):
        index = int(index)

        basename = str(newindex).zfill(10)
        filename = basename + ".png"

        if not os.path.isfile(os.path.join(image_folder, filename)):

            cv2.imwrite(os.path.join(image_folder, filename), np.zeros((5,5))) #Dummpy image to enable parallel procesing

            img = cv2.imread(dataset_items[index])

            if args.occlusions:
                flipcoin_occ = np.random.randint(0, high=3) # Apply on 33% of the images
                if flipcoin_occ == 1:
                    gtfile = open(label_items[index], 'r')
                    lines = gtfile.readlines()
                    for line in lines: # Read the bounding box coordinates of the image
                        if not line in ['\n', '\r\n']:
                            values = line.strip().split(" ")
                            # Convert from yolo format to absolute coordinates
                            w_img = 1280
                            h_img = 720
                            w = float(values[3]) * w_img
                            h = float(values[4]) * h_img
                            # Shift the coordinates randomly, but at least 50% of the original bounding box area should be covered
                            trans_x = -1*np.random.uniform(w/4,w/2) if np.random.randint(0, high=2) == 1 else np.random.uniform(w/4,w/2)
                            trans_y = -1*np.random.uniform(h/4,h/2) if np.random.randint(0, high=2) == 1 else np.random.uniform(h/4,h/2)
                            xmin = int(float(values[1]) * w_img - w/2 + trans_x)
                            xmax = int(float(values[1]) * w_img + w/2 + trans_x)
                            ymin = int(float(values[2]) * h_img - h/2 + trans_y)
                            ymax = int(float(values[2]) * h_img + h/2 + trans_y)
                            # Pick random color and apply the rectangle to the image
                            r = int(np.random.uniform(0,225))
                            g = int(np.random.uniform(0,225))
                            b = int(np.random.uniform(0,225))
                            cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (r,g,b), -1)

            flipcoin = np.random.randint(0, high=2)
            if args.motion_blur and flipcoin == 1: # Apply on 50% of the images
                kernel_size = np.random.randint(3, high=args.motion_blur_max_kernelsize+1)
                direction = np.random.randint(0, high=4)

                # Pick random direction and kernel size
                if direction == 0:
                    kernel = np.zeros((kernel_size, kernel_size))
                    kernel[:, int((kernel_size - 1)/2)] = np.ones(kernel_size)
                    kernel /= kernel_size
                elif direction == 1:
                    kernel = np.zeros((kernel_size, kernel_size))
                    kernel[int((kernel_size - 1)/2), :] = np.ones(kernel_size)
                    kernel /= kernel_size
                elif direction == 2:
                    kernel = np.eye(kernel_size)
                    kernel /= kernel_size
                elif direction == 3:
                    kernel = np.eye(kernel_size)
                    kernel = np.fliplr(kernel)
                    kernel /= kernel_size
                img = cv2.filter2D(img, -1, kernel)

            if args.mixup:
                mixup_image = random.choice(mixup_items)
                img_mix = cv2.imread(mixup_image)
                img_mix = cv2.resize(img_mix, (img.shape[1], img.shape[0]))
                weight = random.uniform(0.0, 0.3) # Mix percentage between 0 and 30
                img = cv2.addWeighted(img, 1 - weight, img_mix, weight, 1.0)

            cv2.imwrite(os.path.join(image_folder, filename), img) # Store the image

        filenamelist.append('./images/' + phase + '/' + filename)

        if not os.path.isfile(os.path.join(label_folder, basename + ".txt")):
            gtfile = open(label_items[index], 'r')
            lines = gtfile.readlines()
            linecntr = 0
            linestostore = []
            for line in lines:
                values = line.strip().split(" ")
                linecntr = linecntr + 1
                if flipcoin_occ == 1:
                    linestostore.append("\n")
                else:
                    values[0] = str(mapdict.get(int(values[0])))
                    newline = ' '.join(values)
                    linestostore.append(newline)
            pfile = open(os.path.join(label_folder, basename + ".txt"),"w") #write mode 
            for line in linestostore:
                pfile.write(line) #Store the label file
            pfile.close()

    pathfile = open(os.path.join(dataset_folder, phase + '.txt'),"w") 
    for entry in filenamelist:
        pathfile.write(entry + '\n')
    pathfile.close()
    

if __name__ == "__main__":
    args = setup_options()
    seed = args.seed

    dataset_items = []
    label_items = []
    for filename in glob.glob(args.input_image_folder + '*.png'):
        dataset_items.append(filename)
        labelfile = os.path.join(args.input_labels, os.path.basename(filename)[:-4] + ".txt") # Make path for corresponding label file
        label_items.append(labelfile)
    if type(args.how_many) == int:
        endvalue = min(len(dataset_items)-1,args.how_many_offset+args.how_many-1)
        dataset_items = dataset_items[args.how_many_offset:endvalue] # Select subset of data if parameter is set
        label_items = label_items[args.how_many_offset:endvalue]

    total_rows = len(dataset_items)
    test_rows = int(total_rows*(args.test_percentage/100))
    val_rows = int(total_rows*(args.val_percentage/100))
    train_rows = int(total_rows - test_rows - val_rows)

    dataset_ids = np.arange(0, total_rows, 1).astype(int)
    if args.randomize:
        shuffled_array = dataset_ids[np.random.RandomState(seed=seed).permutation(total_rows)]
    else:
        shuffled_array = dataset_ids
    train_dataset = np.round(shuffled_array[:train_rows], 1)
    validation_dataset = np.round(shuffled_array[train_rows+1:train_rows+val_rows])
    test_dataset = np.round(shuffled_array[train_rows+val_rows+1:], 1)

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
    create_dataset(args, 'train', dataset_items, label_items, train_dataset)

    if validation_dataset.size > 0:
        print("Adding to Validation set...")
        create_dataset(args, 'val', dataset_items, label_items, validation_dataset)

    if test_dataset.size > 0:
        print("Adding to Test set...")
        create_dataset(args, 'test', dataset_items, label_items, test_dataset)

    print("Done!")  