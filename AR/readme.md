# Action recognition

1. Clone the [PySlowFast](https://github.com/facebookresearch/SlowFast) repository and install all dependencies
2. In slowfast/configs, create an IndustReal folder and place our config files. You can create your own configurations and place them here too, of course. Do not forget to update the config paths to the IndustReal dataset on your device!
3. In slowfast/slowfast/datasets, place our industreal.py file.
4. In slowfast/slowfast/datasets/\_\_init\_\_.py, add the line ```from .industreal import Industreal```
5. (optional) Place kinetics and meccano pre-trained weights in slowfast/checkpoints

Now, you can train your network using 
```
python tools/run_net.py --cfg configs/IndustReal/name_of_your_config.yaml
```

Note that you can set MODALITY to rgb, depth, ambient_light, stereo_left, stereo_right, or stereo (combines stereo_left and stereo_right images). For further clarification on configurations, please refer to the slowfast documentation. 

The weights for the MViT and SlowFast architectures, trained on RGB, are published together with the dataset.
