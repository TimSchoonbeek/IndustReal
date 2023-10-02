"""
Creates a dataset in proper YOLO format from the IndustReal dataset.

author: Tim Houben
email: timhouben@gmail.com
date: 25/06/2023
"""

import argparse
import os
import numpy as np
import glob
from tqdm import tqdm
import cv2
from pylabel import importer

def setup_options():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', default=1, type=int, help='Seed value for data randomization')
    parser.add_argument('--dataset_input_folder', default='data/IndustReal/recordings/', type=str, help='Location of the root folder of the dataset')
    parser.add_argument('--dataset_output_folder', default='datasets/', type=str, help='Location of the output folder')
    parser.add_argument('--name', default='real_dataset', type=str, help='Dataset name')
    parser.add_argument('--only_with_bb_present', action='store_true', help='When enabled, only samples with a bounding box are present in the dataset.')
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
    dataset_folder = os.path.join(args.dataset_output_folder, args.name) 

    image_folder = os.path.join(args.dataset_output_folder, args.name, "images", phase)
    label_folder = os.path.join(args.dataset_output_folder, args.name, "labels", phase)
    create_dir(image_folder)
    create_dir(label_folder)

    dataset_items = []
    folder_items = []
    label_items = []

    # Do for every episode in the dataset
    experiment_folders = os.listdir(os.path.join(args.dataset_input_folder, phase))
    for cfolder in experiment_folders:
        expfolder = os.path.join(args.dataset_input_folder, phase, cfolder)
        labelfolder = os.path.join('templabels', args.name, phase, cfolder, 'labels')
        # Convert Coco Labels to 
        importer.ImportCoco(os.path.join(expfolder, "OD_labels.json")).export.ExportToYoloV5(output_path=labelfolder)
    
        for filename in glob.glob(os.path.join(expfolder, 'rgb', '*.jpg')):
            dataset_items.append(filename)
            folder_items.append(cfolder)
            labelfile = os.path.join(labelfolder, os.path.basename(filename)[:-4] + ".txt") #Make path for corresponding label file
            label_items.append(labelfile)

    filenamelist = []

    for index, ditem in enumerate(tqdm(dataset_items)):
        basename = folder_items[index] + "_" + os.path.basename(ditem)[:-4]

        # copy labelfile first (currently only one bounding box supported)
        clsnr = ""
        if os.path.isfile(label_items[index]):
            gtfile = open(label_items[index], 'r')
            lines = gtfile.readlines()
            linecntr = 0
            linestostore = []
            for line in lines:
                values = line.strip().split(" ")
                clsnr = clsnr + "_" + str(int(values[0])).zfill(2)
                linecntr = linecntr + 1
                if phase == 'train' and values[0] == 23: # Remove label from error states in the training phase
                    linestostore.append('')
                else:
                    linestostore.append(line)

            pfile = open(os.path.join(label_folder, basename + clsnr + ".txt"),"w")
            for line in linestostore:
                pfile.write(line + '\n')
            pfile.close()
            if linecntr == 0:
                clsnr = "_00"
        elif not args.only_with_bb_present:
            clsnr = "_00"
            pfile = open(os.path.join(label_folder, basename + clsnr + ".txt"),"w")
            pfile.write('')
            pfile.close()

        filename = basename + clsnr + ".png"
        
        if not args.only_labels:

            if not ((not os.path.isfile(label_items[index])) and args.only_with_bb_present):

                if not os.path.isfile(os.path.join(image_folder, filename)):

                    cv2.imwrite(os.path.join(image_folder, filename), np.zeros((5,5)))

                    img = cv2.imread(dataset_items[index])

                    cv2.imwrite(os.path.join(image_folder, filename), img)

        filenamelist.append('./images/' + phase + '/' + filename)

    if not os.path.isfile(os.path.join(dataset_folder, phase + '.txt')):
        pathfile = open(os.path.join(dataset_folder, phase + '.txt'),"w") #write mode 
        for entry in filenamelist:
            pathfile.write(entry + '\n')
        pathfile.close()
        

if __name__ == "__main__":
    args = setup_options()
    seed = args.seed

    print("Creating Dataset")
    create_dir(os.path.join(args.dataset_output_folder, args.name))
    create_dir(os.path.join(args.dataset_output_folder, args.name, "images"))
    create_dir(os.path.join(args.dataset_output_folder, args.name, "labels"))
    create_dir(os.path.join(args.dataset_output_folder, args.name, "annotations"))

    with open(os.path.join(args.dataset_output_folder, "template.yaml"), 'r') as file :
        filedata = file.read()
    filedata = filedata.replace('template', args.name)
    with open(os.path.join(args.dataset_output_folder, args.name + ".yaml"), 'w') as file:
        file.write(filedata)

    print("Adding to Train set...")
    create_dataset(args, 'train')

    print("Adding to Validation set...")
    create_dataset(args, 'val')

    print("Adding to Test set...")
    create_dataset(args, 'test')

    print("Done!")