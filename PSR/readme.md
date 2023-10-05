# Procedure step recognition 

## Installation 

```
$ git clone https://github.com/TimSchoonbeek/IndustReal
$ cd IndustReal
$ conda create -n IndustReal python=3.9 -y
$ conda activate IndustReal
$ pip install -r requirements.txt
$ pip install git+https://github.com/infoscout/weighted-levenshtein.git#egg=weighted_levenshtein
```
> The weighted-levenshtein gives an error when installing requirements.txt file, therefore it is installed manually from their [GitHub repo](https://github.com/infoscout/weighted-levenshtein).

## Usage

The object state detections predictions are provided together with the dataset. Therefore, one can run the PSR code on CPU, since there are no new predictions from computationally expensive operations. The PSR code consists of three files:
* *psr_baseline.py*: executes the procedure step recognition algorithm.
* *psr_utils.py*: contains all utilization functions for procedure step recognition.
* *procedure_info.json*: holds the procedural information (***P***), as described in the IndustReal paper.

To perform PSR and reproduce the results outlined in the paper, follow these steps:

1. Set the paths in *psr_baseline.py* to the IndustReal data on your device. The *video_dir* path can be set to *None* if you do not desire visualized outputs, otherwise it must be set to the RGB video recordings (in all_rgb_videos.zip)
2. Determine which baselines you want to reproduce by setting *implementations*. 'naive', 'confidence', and 'expected' correspond to B1, B2, and B3 from the paper. 
3. Set your PSR settings using *psr_config*.
    1. 'asd_dir' should point to the directory with the assembly state detection predictions (i.e., the files in ASD_results_IndustRealplusSynthetic_test.zip or ASD_results_SyntheticOnly_test.zip).
    2. 'cum_conf_threshold' sets the cumulative threshold for determining an observation 'completed' in B2 and B3. Default = 8.
    3. 'cum_decay' sets the multiplication factor to decay non-observations in B2 and B3. Default = 0.75.
    4. 'conf_threshold' sets the confidence threshold for B1. Default = 0.5
5. Execute the script
