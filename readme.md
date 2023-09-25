# Welcome to the IndustReal dataset!


https://github.com/TimSchoonbeek/IndustReal/assets/54444435/02f63bb7-fa40-477a-b6f8-80d66da85656.mp4


The IndustReal dataset contains 84 videos, demonstrating how 27 participants perform 
maintenance and assembly procedures on a construction-toy assembly set. 
The dataset, along with the 3D part geometries and trained model weights for the best performing models as reported
in the paper *"IndustReal: A Dataset for Procedure Step Recognition Handling Execution Errors 
in Egocentric Videos in an Industrial-Like Setting"*, will soon be available.

## News

**Please follow this page, as updates are expected soon!**

[13 Sept 2023] GitHub landing page and PSR videos released


## 3D printing your own model
We publish the part geometries together with the video data and ground-truth labels. To print your own model, you can follow these steps:
1. Use the files included in our dataset or download the latest version from the official [STEMFIE website](https://stemfie.org/sps-000001). Note: if you download the files from the website, you might have a slightly modified version!
2. Calibrate your 3D printer for STEMFIE files using [this tutorial](https://stemfie.org/scf).
3. Scale the files by 200% in all directions to obtain the size used in IndustReal.
4. Print settings: we used an Ultimaker 5S with 0.3mm layer height, 15% triangle infill, 50 mm/s print speed, *0.5 mm hole horizontal expansion*, and raft build plate adhesion.
5. Print materials: The braces and beams are printed in white PLA, the pins in silver metallic PLA, the nuts, washers, wing, and pulley in magenta PLA and the wheels in black tough PLA.


## Procedure step recognition (PSR)
The instructions on reproducing the PSR baseline, along with the required code, will soon be available!

Three videos of the PSR B3 baseline on the IndustReal test set can be found here:
* [03_assy_0_1](https://youtu.be/S-o6MHxvY5c): desired behaviour from the model
* [03_assy_1_3](https://youtu.be/q24UoHaHyck): model does not see error in front chassis and fails to recognize anything after a missing wheel
* [08_assy_2_3](https://youtu.be/sN1uL-F4J4w): model mistakenly classifies final state as correct, whilst a wrong pin is used


## Action recognition (AR)
The instructions on reproducing the AR baseline, along with the required code, will soon be available!


## Assembly state detection (ASD)
The instructions on reproducing the ASD baseline, along with the required code, will soon be available!


## Contributions 

All contributions are welcomed.


## Citing IndustReal
Please use the following BibTeX entry for citation:
```BibTeX
@misc{schoonbeek2024industreal,
title={{IndustReal}: A Dataset for Procedure Step Recognition Handling Execution Errors in Egocentric Videos in an Industrial-Like Setting},
author={Tim Schoonbeek and Tim Houben and Hans Onvlee and Peter de With and Fons van der Sommen},
year={2024},
}
```



## License

IndustReal is released under the Apache 2.0 license.
