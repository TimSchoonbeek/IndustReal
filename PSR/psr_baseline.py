import os
import sys
import numpy as np
import psr_utils
import datetime
from pathlib import Path
import cv2
import time


# fixed parameters for IndustReal
width = 1280
height = 720
FPS = 10
fourcc = cv2.VideoWriter_fourcc(*'mp4v')


if __name__ == '__main__':

    rec_path = Path(r"Data\IndustReal\recordings")
    recordings = psr_utils.get_recording_list(rec_path, test=True)

    create_video = False
    # only if create_video is True, set these
    video_dir = Path(r"Data\IndustReal\videos")
    vid_title = "my_video"  # no extension
    all_video_paths = list(video_dir.glob("*.mp4"))

    implementations = ["naive", "confidence", "expected"]  # these correspond to B1, B2, and B3 in the IndustReal paper
    for impl in implementations:
        print('-'*79)
        psr_config = {
            "implementation": impl,  # options: naive, confidence, expected
            "ads_dir": Path(r"Results\ASD_results_IndustRealplusSynthetic_test"),
            "proc_info": psr_utils.get_procedure_info("procedure_info.json"),
            "cum_conf_threshold": 8,  # cumulative threshold for determining an observation 'completed' in conf based
            "cum_decay": 0.75,  # multiplication factor to decay non-observations in conf based
            "conf_threshold": 0.5,  # confidence threshold for naive implementation
        }

        metrics_all = psr_utils.initiate_metrics()
        metrics_videos_no_errors = psr_utils.initiate_metrics()
        metrics_videos_errors = psr_utils.initiate_metrics()
        for i, rec in enumerate(recordings):
            print(f"Processing recording: {rec.name} \t({i / len(recordings) * 100:.2f}%)")
            result = psr_utils.perform_psr(psr_config, rec)

            gt = psr_utils.load_psr_labels(rec / "PSR_labels.csv")

            metrics = psr_utils.determine_performance(gt, result.y_hat, psr_config["proc_info"], verbose=True)
            print(result.y_hat)
            psr_utils.update_metrics(metrics_all, metrics)
            if psr_utils.video_contains_errors(gt, psr_config["proc_info"], rec):
                psr_utils.update_metrics(metrics_videos_errors, metrics)
            else:
                psr_utils.update_metrics(metrics_videos_no_errors, metrics)

            if create_video:
                vid_load_path = [path for path in all_video_paths if path.name == f"{rec.name}.mp4"][0]
                psr_utils.create_result_video(rec, psr_config, result.y_hat.copy(), vid_load_path, vid_title)

        print(f"Implementation: {impl}")
        psr_utils.print_metrics(metrics_all, title="Average metrics on all videos")
        psr_utils.print_metrics(metrics_videos_no_errors, title="Metrics for only videos without any errors")
        psr_utils.print_metrics(metrics_videos_errors, title="Metrics for only videos with at least one error")
        print('-'*69)
