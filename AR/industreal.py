"""
This file is largely identical to meccano.py at https://github.com/fpv-iplab/MECCANO, as IndustReal is structured
similarly. Please refer to their repo and cite their work if you find it fitting.
"""
# !/usr/bin/env python3

import os
import random
import torch
import torch.utils.data
from fvcore.common.file_io import PathManager

import slowfast.utils.logging as logging

from . import decoder as decoder
from . import transform as transform
from . import utils as utils
from . import video_container as container
from .build import DATASET_REGISTRY
from PIL import Image
import torchvision.transforms as transforms
import numpy as np
from . import sampling

logger = logging.get_logger(__name__)


@DATASET_REGISTRY.register()
class Industreal(torch.utils.data.Dataset):
    """
    IndustReal video loader. Construct the IndustReal video loader, then sample
    clips from the videos. For training and validation, a single clip is
    randomly sampled from every video with random cropping, scaling, and
    flipping. For testing, multiple clips are uniformaly sampled from every
    video with uniform cropping. For uniform cropping, we take the left, center,
    and right crop if the width is larger than height, or take top, center, and
    bottom crop if the height is larger than the width.
    """

    def __init__(self, cfg, mode, num_retries=10):
        """
        Construct the IndustReal images loader with a given csv file. The format of
        the csv file is:
        '''
        video_id_1, action_id_1, action_name_1, frame_start_1, frame_end_1
        video_id_2, action_id_2, action_name_2, frame_start_2, frame_end_2
        ...
        video_id_N, action_id_N, action_name_N, frame_start_N, frame_end_N
        '''
        Args:
            cfg (CfgNode): configs.
            mode (string): Options includes `train`, `val`, or `test` mode.
                For the train and val mode, the data loader will take data
                from the train or val set, and sample one clip per video.
                For the test mode, the data loader will take data from test set,
                and sample multiple clips per video.
            num_retries (int): number of retries.
        """
        # Only support train, val, and test mode.
        assert mode in [
            "train",
            "val",
            "test",
        ], "Split '{}' not supported for IndustReal".format(mode)
        self.mode = mode
        self.cfg = cfg

        self._video_meta = {}
        self._num_retries = num_retries
        # For training or validation mode, one single clip is sampled from every
        # video. For testing, NUM_ENSEMBLE_VIEWS clips are sampled from every
        # video. For every clip, NUM_SPATIAL_CROPS is cropped spatially from
        # the frames.
        if self.mode in ["train", "val"]:
            self._num_clips = 1
        elif self.mode in ["test"]:
            self._num_clips = (
                    cfg.TEST.NUM_ENSEMBLE_VIEWS * cfg.TEST.NUM_SPATIAL_CROPS
            )

        logger.info("Constructing IndustReal {}...".format(mode))
        self._construct_loader()
        self.num_videos = len(self._path_to_videos)

    def _construct_loader(self):
        """
        Construct the data loader.
        """
        path_to_file = os.path.join(
            self.cfg.DATA.PATH_TO_DATA_DIR, "{}.csv".format(self.mode)
        )
        assert PathManager.exists(path_to_file), "{} dir not found".format(
            path_to_file
        )

        self._path_to_videos = []
        self._labels = []
        self._spatial_temporal_idx = []
        self._frame_start = []
        self._frame_end = []
        with PathManager.open(path_to_file, "r") as f:
            for clip_idx, path_label in enumerate(f.read().splitlines()):
                assert len(path_label.split(',')) == 5
                video_path, action_label, action_noun, frame_start, frame_end = path_label.split(',')
                for idx in range(self._num_clips):
                    self._path_to_videos.append(
                        os.path.join(self.cfg.DATA.PATH_PREFIX, video_path)
                    )
                    self._frame_start.append(frame_start)
                    self._frame_end.append(frame_end)
                    self._labels.append(int(action_label))
                    self._spatial_temporal_idx.append(idx)
                    self._video_meta[clip_idx * self._num_clips + idx] = {}
        assert (
                len(self._path_to_videos) > 0
        ), "Failed to load IndustReal split {} from {}" + path_to_file
        logger.info(
            "Constructing IndustReal dataloader (size: {}) from {}".format(
                len(self._path_to_videos), path_to_file
            )
        )

    def __getitem__(self, index):
        """
        Given the video index, return the list of frames, label, and video
        index.
        Args:
            index (int): the video index provided by the pytorch sampler.
        Returns:
            frames (tensor): the frames of sampled from the video. The dimension
                is `channel` x `num frames` x `height` x `width`.
            label (int): the label of the current video.
            index (int): if the video provided by pytorch sampler.
        """
        if self.mode in ["train", "val"]:
            # -1 indicates random sampling.
            temporal_sample_index = -1
            spatial_sample_index = -1
            min_scale = self.cfg.DATA.TRAIN_JITTER_SCALES[0]
            max_scale = self.cfg.DATA.TRAIN_JITTER_SCALES[1]
            crop_size = self.cfg.DATA.TRAIN_CROP_SIZE
        elif self.mode in ["test"]:
            temporal_sample_index = (
                    self._spatial_temporal_idx[index]
                    // self.cfg.TEST.NUM_SPATIAL_CROPS
            )
            # spatial_sample_index is in [0, 1, 2]. Corresponding to left,
            # center, or right if width is larger than height, and top, middle,
            # or bottom if height is larger than width.
            spatial_sample_index = (
                    self._spatial_temporal_idx[index]
                    % self.cfg.TEST.NUM_SPATIAL_CROPS
            )
            min_scale, max_scale, crop_size = [self.cfg.DATA.TEST_CROP_SIZE] * 3
            # The testing is deterministic and no jitter should be performed.
            # min_scale, max_scale, and crop_size are expect to be the same.
            assert len({min_scale, max_scale, crop_size}) == 1
        else:
            raise NotImplementedError(
                "Does not support {} mode".format(self.mode)
            )

        # Recover frames
        frames = []
        frame_count = int(self._frame_start[index][:-4])  # to% obtain the number of the frame
        indexes = self.temporal_sampling(int(self._frame_start[index][:-4]), int(self._frame_end[index][:-4]),
                                         self.cfg.DATA.NUM_FRAMES)
        prev_frame_to_load = -1
        prev_loaded_frame = None
        n_loaded_new = 0
        for i, idx in enumerate(indexes):
            frame_to_load = frame_count + idx
            if frame_to_load == prev_frame_to_load:
                image = prev_loaded_frame
            else:
                name_frame = str(frame_to_load).zfill(6)
                if self.cfg.DATA.MODALITY == "stereo":
                    img_left = Image.open(
                        self.cfg.DATA.PATH_TO_DATA_DIR + f"recordings/{self.mode}/" + self._path_to_videos[
                            index] + "/stereo_left/" + name_frame + ".jpg")
                    img_right = Image.open(
                        self.cfg.DATA.PATH_TO_DATA_DIR + f"recordings/{self.mode}/" + self._path_to_videos[
                            index] + "/stereo_right/" + name_frame + ".jpg")
                    img_left = np.array(img_left)
                    img_right = np.array(img_right)
                    image = np.concatenate((img_left, img_right), axis=1)
                else:
                    image = Image.open(self.cfg.DATA.PATH_TO_DATA_DIR + f"recordings/{self.mode}/" + self._path_to_videos[
                        index] + "/" + self.cfg.DATA.MODALITY + "/" + name_frame + ".jpg").resize((1280, 720))
                    image = np.array(image)
                if len(image.shape) == 2:
                    # if we have grayscale image, duplicate grayscale into R, G, B channels for consistency
                    image = np.stack((image,) * 3, axis=-1)
                image = torch.from_numpy(image).float()
                prev_loaded_frame = image.clone()
                n_loaded_new += 1
            frames.append(image)
            prev_frame_to_load = frame_to_load

        frames = torch.stack(frames)
        frames = frames / 255.0
        frames -= torch.tensor(self.cfg.DATA.MEAN)
        frames /= torch.tensor(self.cfg.DATA.STD)
        frames = frames.permute(3, 0, 1, 2)

        # Perform data augmentation.
        frames = self.spatial_sampling(
            frames,
            spatial_idx=spatial_sample_index,
            min_scale=min_scale,
            max_scale=max_scale,
            crop_size=crop_size,
            random_horizontal_flip=self.cfg.DATA.RANDOM_FLIP,
        )
        label = self._labels[index]
        frames = utils.pack_pathway_output(self.cfg, frames)
        return frames, label, index, torch.zeros(1), {}

    def __len__(self):
        """
        Returns:
            (int): the number of videos in the dataset.
        """
        return len(self._path_to_videos)

    def spatial_sampling(self, frames, spatial_idx=-1, min_scale=256, max_scale=320, crop_size=224,
                         random_horizontal_flip=True):
        """
        Perform spatial sampling on the given video frames. If spatial_idx is
        -1, perform random scale, random crop, and random flip on the given
        frames. If spatial_idx is 0, 1, or 2, perform spatial uniform sampling
        with the given spatial_idx.
        Args:
            frames (tensor): frames of images sampled from the video. The
                dimension is `num frames` x `height` x `width` x `channel`.
            spatial_idx (int): if -1, perform random spatial sampling. If 0, 1,
                or 2, perform left, center, right crop if width is larger than
                height, and perform top, center, buttom crop if height is larger
                than width.
            min_scale (int): the minimal size of scaling.
            max_scale (int): the maximal size of scaling.
            crop_size (int): the size of height and width used to crop the
                frames.
        Returns:
            frames (tensor): spatially sampled frames.
        """
        assert spatial_idx in [-1, 0, 1, 2]
        if spatial_idx == -1:
            frames, _ = transform.random_short_side_scale_jitter(
                images=frames,
                min_size=min_scale,
                max_size=max_scale,
                inverse_uniform_sampling=self.cfg.DATA.INV_UNIFORM_SAMPLE,
            )
            frames, _ = transform.random_crop(frames, crop_size)
            if random_horizontal_flip:
                frames, _ = transform.horizontal_flip(0.5, frames)
        else:
            # The testing is deterministic and no jitter should be performed.
            # min_scale, max_scale, and crop_size are expect to be the same.
            assert len({min_scale, max_scale, crop_size}) == 1
            frames, _ = transform.random_short_side_scale_jitter(
                frames, min_scale, max_scale
            )
            frames, _ = transform.uniform_crop(frames, crop_size, spatial_idx)
        return frames

    def temporal_sampling(self, start_idx: int, end_idx: int, num_samples: int):
        """
            Given the start and end frame index, sample num_samples frames between
            the start and end with equal interval. If the number of frames is < num_samples, duplicate frames.
            Args:
                start_idx (int): the index of the start frame.
                end_idx (int): the index of the end frame.
                num_samples (int): number of frames to sample.
            Returns:
                index (???): the indexes of which frames to sample
            """
        index = np.linspace(start_idx, end_idx, num_samples)
        index = np.clip(index, 0, end_idx).astype(np.int32)
        index = index - start_idx
        return index
