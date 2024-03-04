# Welcome to the IndustReal dataset!




https://github.com/TimSchoonbeek/IndustReal/assets/54444435/ec67b3f4-b2c5-4457-af66-be7beb680f21


### Check out our [project page](https://timschoonbeek.github.io/industreal.html)!

The IndustReal dataset contains 84 videos, demonstrating how 27 participants perform 
maintenance and assembly procedures on a construction-toy assembly set. 
The dataset, along with the 3D part geometries and trained model weights for the best performing models as reported
in the paper *"IndustReal: A Dataset for Procedure Step Recognition Handling Execution Errors 
in Egocentric Videos in an Industrial-Like Setting"*, is available [here](https://data.4tu.nl/datasets/b008dd74-020d-4ea4-a8ba-7bb60769d224).

## News

**Please follow this page, as updates are expected soon!**

[01 Nov  2023] [Data](https://data.4tu.nl/datasets/b008dd74-020d-4ea4-a8ba-7bb60769d224) released! \
[30 Oct  2023] Paper pre-print is available [here](https://arxiv.org/pdf/2310.17323.pdf) \
[03 Oct  2023] PSR, AR, and ASD code released \
[13 Sept 2023] GitHub landing page and PSR videos released


## Procedure step recognition (PSR)
The instructions on reproducing the PSR baseline, along with the required code, can be found [here](PSR/readme.md).

Three videos of the PSR B3 baseline on the IndustReal test set can be found here:
* [03_assy_0_1](https://youtu.be/S-o6MHxvY5c): desired behaviour from the model
* [03_assy_1_3](https://youtu.be/q24UoHaHyck): model does not see error in front chassis and fails to recognize anything after a missing wheel
* [08_assy_2_3](https://youtu.be/sN1uL-F4J4w): model mistakenly classifies final state as correct, whilst a wrong pin is used


## Action recognition (AR)
The instructions on reproducing the AR baseline, along with the required code, can be found [here](AR/readme.md).


## Assembly state detection (ASD)
The instructions on reproducing the ASD baseline, along with the required code, can be found [here](ASD/readme.md).


## Data information

The dataset can be downloaded [here](https://data.4tu.nl/datasets/b008dd74-020d-4ea4-a8ba-7bb60769d224). IndustReal should be structured as follows:
```md
IndustReal
├── part_geometries
│   ├── fbx_files
│   ├── 3mf_files
│   ├── overview_of_states.pdf
├── recordings
│   ├── train
│   │   ├── recording_x
│   │   │   ├── rgb
│   │   │   ├── stereo_left
│   │   │   ├── stereo_right
│   │   │   ├── depth
│   │   │   ├── ambient_light
│   │   │   ├── gaze.csv
│   │   │   ├── hands.csv
│   │   │   ├── pose.csv
│   │   │   ├── AR_labels.csv
│   │   │   ├── OD_labels.json
│   │   │   ├── PSR_labels.csv
│   │   │   ├── PSR_labels_with_errors.csv
│   │   │   └── PSR_labels_raw.csv
│   │   └── ...
│   ├── val
│   │   └── ...
│   ├── test
│   │   └── ...
├── train.csv
├── val.csv
└── test.csv
```

### Sensor and tracking information
All data is recorded using the Microsoft HoloLens 2 and the [HL2SS library](https://github.com/jdibenes/hl2ss). Every recording (e.g. 01_assy_1_1) contains 5 folders with the image data, named:

* **rgb (PV Camera)**. RGB images in .jpg format. Resolution is 1080x720 pixels, recorded at 10 FPS.
* **stereo_left and stereo_right (Visible Light Cameras)**. Grayscale images in .jpg format of the left-front and right-front cameras. Resolution is 640x480 pixels, recorded at 10 FPS.
* **depth (Long Throw)**. 3-channel images in .jpg format, 320x288 pixels at 5 FPS. The depth readings are clipped at 75cm, after which the matplotlib *turbo* colormap is applied, and values are normalized to uint8 integers. Unfortunately, software limitations prohibited us from using the AHAT (Short Throw) depth images, it can not be used simultaneously with the PV Camera (at the time of recording). This reduces depth quality and limits FPS to 5. We invite depth estimations based on the provided stereo images.
* **ambient_light**. Grayscale images in .jpg format, 320x288 pixels at 5 FPS, acquired together with the depth measurements. The images are obtained by dividing sensor readings by its maximum value and normalizing to the uint8 range.

Becuase depth and ambient_light are recorded at half FPS compared to rgb and stereo images, their frames are duplicated to ensure an equal number of frames in each folder.

Each folder also contains the three tracking readings:
* **gaze.csv**. Gaze tracking from the HL2, recorded at 10 FPS. Each row denotes the corresponding image name and the x and y coordinates (in image space).
* **hands.csv**. Hand tracking from the HL2, recorded at 10 FPS. Each row denotes the corresponding image name, 52 coordinates for the left hand joints, and 52 coordinates for the right hand joints.
* **pose.csv**. Head pose tracking from the HL2, recorded at 10 FPS. Each row denotes the corresponding image name, forward pose position (xyz), position (xyz), and up (xyz).

### Annotation information
In each folder, you will also find the annotations:
* Action recognition: **AR_labels.csv**. Each row indicates the recording name, action id, action description, start frame and end frame. These labels are not actually used by the action recognition algorithm, as we created our train-val-test splits according to the [PySlowFast](https://github.com/facebookresearch/SlowFast) library and published these (*train.csv*, *val.csv*, and *test.csv*). However, we keep the annotations in each recording folder in case you want to use a another action recognition framework.
* Assembly state detection: **OD_labels.json**. ASD labels provided in COCO format.
* Procedure step recognition: **PSR_labels.csv**, **PSR_labels_with_errors.csv**. These files contain the PSR labels, with at each row the name of the image where the action is deemed completed, the completed step id, and the description for this step. Note that *PSR_labels_with_errors.csv* also contains entries for wrongly executed steps, e.g. *incorrectly installed rear wheel assy*. Finally, **PSR_labels_raw.csv** contains raw labels, where the state of each component in IndustReal is indicated with a -1 (incorrectly completed), 0 (not (yet) completed), or 1 (correctly completed). These are provided to provide flexibility to different implementations.


## 3D printing your own model
We publish the part geometries together with the video data and ground-truth labels. To print your own model, you can follow these steps:
1. Use the files included in our dataset or download the latest version from the official [STEMFIE website](https://stemfie.org/sps-000001). Note: if you download the files from the website, you might have a slightly modified version!
2. Calibrate your 3D printer for STEMFIE files using [this tutorial](https://stemfie.org/scf).
3. Scale the files by 200% in all directions to obtain the size used in IndustReal.
4. Print settings: we used an Ultimaker 5S with 0.3mm layer height, 15% triangle infill, 50 mm/s print speed, *0.5 mm hole horizontal expansion*, and raft build plate adhesion.
5. Print materials: The braces and beams are printed in white PLA, the pins in silver metallic PLA, the nuts, washers, wing, and pulley in magenta PLA and the wheels in black tough PLA.

We are thankful to the [STEMFIE project](https://stemfie.org/sps-000001) for open-sourcing their 3D printed toy construction sets, enabling our research!

## Recording your own data
To record your own data, you should have a HoloLens 2 (HL2). Then, you can record your data in the same format as IndustReal following these steps:
1. Clone the [HL2SS library](https://github.com/jdibenes/hl2ss) and install all required dependencies. 
2. Open the [hl2ss_recording_script.py](hl2ss_recording_script.py). Set your host IP (ask "what is my IP" to the HoloLens you are trying to record on) and the directory where the data should be saved.
3. Start the HL2SS app on the HL2.
4. (optional) Enable the PV camera field-of-view limit indication line with the HL2SS provided script. This prevents users from aiming the HL2 too high, a commonly encountered issue.
5. Start the [hl2ss_recording_script.py](hl2ss_recording_script.py). You are now recording!


## Contributions 

All contributions are welcomed.


## Citing IndustReal
Please use the following BibTeX entry for citation:
```BibTeX
@inproceedings{schoonbeek2024industreal,
  title={IndustReal: A Dataset for Procedure Step Recognition Handling Execution Errors in Egocentric Videos in an Industrial-Like Setting},
  author={Schoonbeek, Tim J and Houben, Tim and Onvlee, Hans and van der Sommen, Fons and others},
  booktitle={Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision},
  pages={4365--4374},
  year={2024}
}
```


## License

IndustReal is released under the Apache 2.0 license.
