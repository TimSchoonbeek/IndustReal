import numpy as np
import cv2
import os
import csv
import json
from pathlib import Path
from weighted_levenshtein import dam_lev
import matplotlib.pyplot as plt
import pprint
from bounding_box import bounding_box as bb
import time


FPS = 10
pp = pprint.PrettyPrinter(indent=4)

# fixed parameters for IndustReal
width = 1280
height = 720
fourcc = cv2.VideoWriter_fourcc(*'mp4v')

# parameters for result visualization
show_time = 50  # number of frames to show log entries for
font_scale = 2  # font size for visualizing procedure step completions
thickness = 3  # thickness for visualizing procedure step completions
black = (0, 0, 0)  # color
light_green = (25, 180, 25)  # color
font = cv2.FONT_HERSHEY_SIMPLEX


categories = ['background',
              '10000000000',  # state 1
              '10010010000',  # state 2
              '10010100000',  # state 3
              '10010110000',  # state 4
              '11100000000',  # state 5
              '11110010000',  # state 6
              '11110100000',  # state 7
              '11110110000',  # state 8
              '11110111100',  # state 9
              '11110111110',  # state 10
              '11110110001',  # state 11
              '11110111101',  # state 12
              '11110111111',  # state 13
              '11110101111',  # state 14
              '11110011111',  # state 15
              '11110011110',  # state 16
              '11110101110',  # state 17
              '11100001110',  # state 18
              '11101101110',  # state 19
              '11101011110',  # state 20
              '11101111110',  # state 21
              '11101111111',  # state 22
              'error_state']


class NaivePSR:
    """
    Naive implementation of a procedure step recognition system. The system takes each state detection, if this
    differs from the last observed state, it takes the difference and considers those steps completed.
    Initialization of states is only done based on 1st detection, not based on the procedure.
    If a frame contains multiple predictions, only highest confidence is chosen.
    """
    def __init__(self, config: dict):
        self.procedure_info = config["proc_info"]
        self.thresh = config["conf_threshold"]
        self.current_state = None
        self.current_state_str = None
        self.y_hat = []

    def update(self, pred: list, frame_n:int):
        if len(pred) == 0:
            return
        pred_class, conf = get_highest_conf_prediction(pred)
        pred_state_str = categories[int(pred_class)]

        # initialize first state
        if self.current_state is None:
            self.current_state = state_string_to_list(pred_state_str)
            self.current_state_str = pred_state_str
            return

        # no legitimate new states observed
        if self.current_state_str == pred_state_str or pred_state_str == 'error_state':
            return

        # confidence below threshold
        if conf <= self.thresh:
            return

        pred_state = state_string_to_list(pred_state_str)
        # system does not assume based on procedure, but estimates it all based on detection. So use conf=1
        actions, _ = convert_states_to_steps(self.current_state, pred_state, frame_n, self.procedure_info, conf=1)

        self.update_y_hat(actions)
        self.current_state_str = pred_state_str
        self.current_state = state_string_to_list(pred_state_str)

    def update_y_hat(self, actions: list):
        for action in actions:
            self.y_hat.append(action)


class AccumulatedConfidencePSR:
    """
    Implementation of a procedure step recognition system based on accumulated confidences per step. If the ASD system
    detects a state, we determine the steps required to reach that state, and add the confidence to each state. Thus,
    when there are two simultaneous detections for different classes, and some step completions correspond, both
    confidences are accumulated. If a prediction from a previous frame is not seen again, it is decayed by 0.75.
    If procedure is None, all actions are expected. If procedure is 'assy' or 'main', only the actions expected in
    these procedures are considered.
    """
    def __init__(self, config: dict, procedure=None):
        self.procedure_info = config["proc_info"]
        self.procedure = procedure
        self.cum_conf_threshold = config["cum_conf_threshold"]
        self.decay = config["cum_decay"]
        self.cum_confs = np.zeros(len(self.procedure_info))  # accumulation of confidences over time
        self.y_hat = []
        self.frame_n = -1
        self.updated_conf_idxes = []
        self.action_idxes = [i for i in range(len(self.procedure_info))]
        self.completed_action_ids = []
        if self.procedure is None:
            self.expected_actions = self.action_idxes.copy()
            self.current_state = None
            self.current_state_str = None
        else:
            self.procedure_actions = [action["id"] for action in self.procedure_info if
                                      action[f"expected_in_{self.procedure}"]]
            self.expected_actions = self.procedure_actions.copy()
            if self.procedure == 'assy':
                self.current_state_str = '10000000000'
            elif self.procedure == 'main':
                self.current_state_str = '11110111111'
            self.current_state = state_string_to_list(self.current_state_str)

    def update(self, pred: list, frame_n: int):
        self.frame_n = frame_n

        if len(pred) != 0:
            pred_class, pred_conf = pred[0][0], pred[0][1]

            # determine which step changes are predicted to have occurred
            pred_state_str = categories[int(pred_class)]
            pred_state = state_string_to_list(pred_state_str)

            # initialize
            if self.current_state is None:
                self.current_state = pred_state
                self.current_state_str = pred_state_str
                return

            suggested_actions, _ = convert_states_to_steps(self.current_state, pred_state, frame_n, self.procedure_info,
                                                           conf=1)

            if len(suggested_actions) != 0:
                self.update_cum_confs(suggested_actions, pred_conf)
                self.check_for_completed_actions()

        self.tick()

    def tick(self):
        """ all not updated IDXes, multiply by decay factor """
        for idx in self.action_idxes:
            if idx not in self.updated_conf_idxes:
                self.cum_confs[idx] *= self.decay
        self.updated_conf_idxes = []

    def update_cum_confs(self, actions: list, confidence: float):
        for action in actions:
            self.cum_confs[action["id"]] += confidence
            self.updated_conf_idxes.append(action["id"])

    def check_for_completed_actions(self):
        idxes_completed_steps = list(np.nonzero(self.cum_confs > self.cum_conf_threshold)[0])

        for idx in idxes_completed_steps:
            if idx in self.expected_actions:
                self.process_action(idx)
            else:
                self.cum_confs[idx] = 0  # reset confidences, in case action gets un-done later
                # todo: deal with unexpected actions! provide warning or smth

    def process_action(self, idx):
        self.cum_confs[idx] = 0  # reset confidences, in case action gets un-done later
        state_idx = self.procedure_info[idx]["state_idx"]
        install = self.procedure_info[idx]["install"]
        if install:
            self.current_state[state_idx] = 1
        else:
            self.current_state[state_idx] = 0
        self.y_hat.append(make_entry(self.frame_n, idx, self.procedure_info))
        self.completed_action_ids.append(idx)


def perform_psr(config: dict, rec_dir: Path):
    implemented = ["naive", "confidence", "expected"]
    assert config["implementation"] in implemented, f"Only implementations able to test: {implemented} but you're " \
                                                    f"trying: {config['implementation']}"
    name = rec_dir.name
    if 'assy' in name:
        procedure = 'assy'
    elif 'main' in name:
        procedure = 'main'
    else:
        procedure = None

    frames = list((rec_dir / 'RGB').glob("*.jpg"))
    frames.sort()
    n_frames = len(frames)

    # load ASD predictions
    asd_predictions = load_asd_predictions(config["ads_dir"], rec_dir, n_frames)

    # initialize PSR system
    if config["implementation"] == implemented[0]:
        PSR = NaivePSR(config)
    elif config["implementation"] == implemented[1]:
        PSR = AccumulatedConfidencePSR(config)
    elif config["implementation"] == implemented[2]:
        PSR = AccumulatedConfidencePSR(config, procedure)
    else:
        raise ValueError(f"Can't load PSR implementation for {config['implementation']}")

    times = []
    for frame_n in range(n_frames):
        # print(f"{name}: \t{frame_n}/{n_frames} ({frame_n/n_frames*100:.2f}%)")
        t1 = time.time()
        asd_preds = asd_predictions[frame_n][1]
        PSR.update(asd_preds, frame_n)
        times.append(time.time() - t1)
    print(f"Mean computation time per frame: {mean_list(times):.5f}ms")
    return PSR


def make_entry(frame: int, action_id: int, proc_info: list, conf=1) -> dict:
    """ Conf of 1 indicates observed, conf of 0 indicates implied action step. """
    return {"frame": frame, "id": action_id, "description": proc_info[action_id]["description"], "conf": conf}


def initiate_metrics():
    m = {
        'pos': [],
        'f1': [],
        'avg_delay': [],
        'system_TPs': 0,
        'system_FPs': 0,
        'system_FNs': 0,
    }
    return m


def update_metrics(avg, new):
    for key in avg:
        if type(avg[key]) == list:
            avg[key].append(new[key])
        else:
            avg[key] += new[key]


def print_metrics(m: dict, title: str):
    print('-' * 69)
    print(title)
    print(f"POS score: \t{mean_list(m['pos']):.3f}\n"
          f"F-1 score: \t{mean_list(m['f1']):.3f}\t TP: {m['system_TPs']}\tFP: {m['system_FPs']}\tFN: "
          f"{m['system_FNs']}\n"
          f"Average delay: \t{mean_list(m['avg_delay']):.0f} ({mean_list(m['avg_delay'])/10:.1f} seconds)\n")
    print('-' * 69)


def create_result_video(rec_dir: Path, config: dict, pred: list, vid_load_path: Path, title="result"):
    print("-"*69)
    print(f"Creating video for {rec_dir.name}")
    print("-" * 69)
    name = rec_dir.name
    frames = list((rec_dir / 'RGB').glob("*.jpg"))
    frames.sort()
    n_frames = len(frames)

    # load ASD predictions
    asd_predictions = load_asd_predictions(config["ads_dir"], rec_dir, n_frames)

    res_path = Path(r"Results\videos")
    vid_path = res_path / f"{config['implementation']}" / f"{name}_{title}.mp4"
    vid_path.parent.mkdir(parents=True, exist_ok=True)
    save_video = cv2.VideoWriter(str(vid_path), fourcc, FPS, (width, height))
    load_video = cv2.VideoCapture(str(vid_load_path))

    for frame_n in range(n_frames):
        if frame_n % 50 == 0:
            print(f"{name}: \t{frame_n}/{n_frames} ({frame_n/n_frames*100:.2f}%)")
        asd_preds = asd_predictions[frame_n][1]
        ret, img = load_video.read()
        if not ret:
            continue
        img = plot_bboxes(img, asd_preds)
        img = plot_steps(img, pred, frame_n, real_time=False)
        save_video.write(img)
    save_video.release()
    print("-" * 69)
    print(f"Video saved to {vid_path}")
    print("-" * 69)


def plot_bboxes(img: np.array, preds: list) -> np.array:
    for pred in preds:
        class_id, conf = pred[0], pred[1]
        x, y, w, h = pred[2]
        cat_name = categories[class_id]
        left = int(x - w/2)
        right = int(x + w/2)
        top = int(y - h/2)
        bottom = int(y + h/2)
        bb.add(img, left, top, right, bottom, label=cat_name)
    return img


def plot_steps(img: np.array, y_hat: list, frame_n: int, real_time=False) -> np.array:
    """ If real-time, it assumes the latest log entry is always at current frame number. If not real time, deletes all
    the completion steps predicted after current frame_n
    """
    if len(y_hat) == 0:
        return img

    if not real_time:
        # delete all entries made after frame_n
        del_idxes = [i for i, entry in enumerate(y_hat) if entry["frame"] > frame_n]
        y_hat = [item for index, item in enumerate(y_hat) if index not in del_idxes]

    img_original = img.copy()
    y = 60
    x_pos = 15
    c = 0
    for entry in reversed(y_hat):
        c += 1
        f_diff = frame_n - entry["frame"]
        if f_diff <= show_time:  # still display the image
            msg = entry["description"]
            y_pos = int(y * c)
            text_size, _ = cv2.getTextSize(msg, font, font_scale, thickness)
            cv2.rectangle(img, (x_pos, y_pos + 15), (x_pos + text_size[0], y_pos - text_size[1] - 10), light_green,
                          cv2.FILLED)
            cv2.putText(img, msg, (x_pos, y_pos), font, font_scale, black, thickness)
        else:
            break  # because we reversed y_hat, if any message we encounter is beyond show_time, we can break out
    img = cv2.addWeighted(img, 0.7, img_original, 0.3, 1.0)
    return img


def print_y(y, title="Y"):
    n_chars = len(title)
    print("-" * 29 + f" {title}: " + "-" * 29)
    pp.pprint(y)
    print("-" * int(29*2 + n_chars + 2) + "\n")


def state_string_to_list(state_string: str) -> list:
    state_list = []
    idx = 0
    while idx < len(state_string):
        s = state_string[idx]
        if s == '1':
            state_list.append(1)
        elif s == '0':
            state_list.append(0)
        idx += 1
    return state_list


def get_highest_conf_prediction(predictions: list) -> list:
    if len(predictions) > 1:
        highest_pred = predictions[0]  # initialize first as highest conf
        for pred, conf, _ in predictions:
            if conf > highest_pred[1]:
                highest_pred = [pred, conf]
        return highest_pred[0], highest_pred[1]
    else:
        return predictions[0][0], predictions[0][1]


def convert_ints_to_chars(ints):
    """
    The weighted_levenshtein implementation of the DamLev requires unique characters as input to the function. Therefore
    we take the integers and convert them into unique characters. Note that only real characters can be used for this,
    which means the first 33 ASCII indexes can't be used. So for this implementation, the number of unique procedure
    steps is limited to 128 - 33 = 95 characters.

    Args:
        ints: list of unique integers containing a sequence order

    Returns:
        a string of characters
    """
    result = ''
    for i in ints:
        if i > 128 - 33 or i < 0:
            print(f"Must provide unique ints between 0 and 128, but provided {i}")
            break
        result += chr(i + 33)
    return result


def procedure_order_similarity(gt, pred):
    """
    Calculates the POS measure as proposed in Procedure Step Recognition and Tracking: A  Framework towards
    Understanding Procedural Actions PSRT in section 3.2.1

    Args:
        gt: list of integers describing the IDs of the ground truth action order
        pred: list of integers describing the IDs of the predicted action order

    Returns:
        The procedure order similarity [0, 1], where 1 is a perfect score. Note that 0 is not necessarily the worst
        score, it is just a bad score. This is outlined in the paper.
    """
    delete_costs = np.ones(128, dtype=np.float64) * 1
    insert_costs = np.ones(128, dtype=np.float64) * 1
    substitute_costs = np.ones((128, 128), dtype=np.float64) * 2
    transpose_costs = np.ones((128, 128), dtype=np.float64) * 1

    gt_string = convert_ints_to_chars(gt)
    pred_string = convert_ints_to_chars(pred)
    distance = dam_lev(gt_string, pred_string, insert_costs=insert_costs, delete_costs=delete_costs,
                       substitute_costs=substitute_costs, transpose_costs=transpose_costs)
    score = 1 - min((distance/len(gt)), 1)
    return score, distance


def get_f1_score(FN, FP, TP):
    # https://en.wikipedia.org/wiki/F-score
    P = TP + FN
    PP = TP + FP
    if PP != 0:
        precision = TP / PP  # Positive predictive value
    else:
        precision = 1e-6
    if P != 0:
        recall = TP / P  # True positive rate, sensitivity
    else:
        recall = 1e-6
    f1 = 2 * (precision * recall) / (precision + recall + 1e-6)
    return f1


def get_FN_FP_single_entry(gt_frame_n, pred_frame_n, conf_pred):
    sys_FP, per_FN, per_FP = False, False, False  # assume all false until found otherwise
    delay = None
    if conf_pred == 0:
        per_FN = True  # perception model did not see state, but system implied it

    delta_frames = pred_frame_n - gt_frame_n
    # determine system false positives
    if delta_frames < 0:
        if conf_pred == 0:
            per_FP = True  # perception model saw an action that did not actually happen
        elif conf_pred == 1:
            sys_FP = True  # system implied an action that was not performed

    if delta_frames >= 0:  # if True, prediction is not false positive
        delay = pred_frame_n - gt_frame_n
    return sys_FP, per_FN, per_FP, delay


def match_indices(idxes_a, all_times_a, idxes_b, all_times_b):
    """
    Matches each index in b with the closest match in time of a. Returns indexes of a that match to the indexes of b.
    """
    assert len(idxes_a) >= len(idxes_b), "This function requires first input to have more or equal indexes than second"
    # set times of a to high if not matching our index
    times_a = np.ones(len(all_times_a))*1e9
    for idx in idxes_a:
        times_a[idx] = all_times_a[idx]
    times_b = np.array([all_times_b[i] for i in idxes_b])
    matching_idxes = []
    for time_b in times_b:
        t_diff = times_a - time_b
        t_diff_pen = np.where(t_diff > 0, t_diff, np.inf)
        min_idx = np.argmin(t_diff_pen)
        matching_idxes.append(min_idx)
        times_a[min_idx] = 1e9  # to ensure one match per index
    return matching_idxes


def determine_performance(gt, pred, proc_info, verbose=False):
    """
    function determines false positives on perceptual level (only "observed" confidence) and system level (implied too).

    Args:
        gt: list of dicts containing ground truth frame and id and description of the action
        pred: list of dicts containing predicted frame, id and description of the action, and confidence. Confidence
        value of 0 indicates the system did NOT observe the step, but assumed it based on sytem information. Confidence
        value of 1 indicates system observed the step.
        proc_info: the procedural info used by the PSR system, a list of dicts with ["id"] and ["description"] at least
        verbose: bool, indicating how verbose you want the output

    Returns:
        dict containing relevant metrics:
            perception_FPs: FPs for perceptual model,
            perception_FNs: FNs for perceptual model,
            system_FPs: FPs for system level,
            system_FNs: FNs for system level,
            system_TPs: FPs for system level,
            f1: f1-score (system level) for observed actions as described in Section 3.2.2,
            POS: Procedure Order Similarity measure, as described in Section 3.2.1,
            avg_delay: average delay between GT and pred, as described in Section 3.2.3,
    """
    gt_obs_times = np.zeros(len(gt), dtype=int)
    gt_order = np.zeros(len(gt), dtype=int)
    for i, entry in enumerate(gt):
        gt_obs_times[i] = entry["frame"]
        gt_order[i] = int(entry["id"])

    pred_obs_times = np.zeros(len(pred), dtype=int)
    pred_order = np.zeros(len(pred), dtype=int)
    pred_confs = np.zeros(len(pred))
    for i, entry in enumerate(pred):
        pred_obs_times[i] = entry["frame"]
        pred_order[i] = int(entry["id"])
        pred_confs[i] = int(entry["conf"])

    sys_FNs, sys_FPs, per_FNs, per_FPs = 0, 0, 0, 0
    delays = np.empty(len(gt_obs_times))
    delays[:] = np.nan
    for i, step_info in enumerate(proc_info):
        # find indexes of step id in gt and predictions
        idxes_gt = list(np.where(np.array(gt_order) == step_info["id"])[0])
        idxes_pred = list(np.where(np.array(pred_order) == step_info["id"])[0])
        calculate_FNs_FPs = True
        if len(idxes_gt) == len(idxes_pred) and len(idxes_pred) > 1: # same # GT and predictions and requires matching
            idxes_pred = match_indices(idxes_pred, pred_obs_times, idxes_gt, gt_obs_times)
        elif len(idxes_gt) == 0 and len(idxes_pred) > 0:  # not observed in GT but predicted at least once (FPs)
            sys_FPs += len(idxes_pred)
            per_FPs += len(idxes_pred)
            calculate_FNs_FPs = False
        elif len(idxes_gt) > 0 and len(idxes_pred) == 0:  # not predicted but observed at least once in GT (FNs)
            sys_FNs += len(idxes_gt)
            per_FNs += len(idxes_gt)
            calculate_FNs_FPs = False
        else:
            if len(idxes_gt) > len(idxes_pred):  # more GTs than preds, so # unmatched GTs become FNs
                sys_FNs += len(idxes_gt) - len(idxes_pred)
                per_FNs += len(idxes_gt) - len(idxes_pred)
                idxes_gt = match_indices(idxes_gt, gt_obs_times, idxes_pred, pred_obs_times)
            else:  # more preds than GTs, so # unmatched preds become FPs
                sys_FPs += len(idxes_pred) - len(idxes_gt)
                per_FPs += len(idxes_pred) - len(idxes_gt)
                idxes_pred = match_indices(idxes_pred, pred_obs_times, idxes_gt, gt_obs_times)

        if not calculate_FNs_FPs:
            continue
        for idx_gt, idx_pred in zip(idxes_gt, idxes_pred):
            gt_frame_n = gt_obs_times[idx_gt]
            pred_frame_n = pred_obs_times[idx_pred]
            conf_pred = pred_confs[idx_pred]
            sys_FP, per_FN, per_FP, delay = get_FN_FP_single_entry(gt_frame_n, pred_frame_n, conf_pred)
            if sys_FP:
                sys_FPs += 1
            if per_FN:
                per_FNs += 1
            if per_FP:
                per_FPs += 1

            if delay is not None:
                delays[idx_gt] = delay

    pos, _ = procedure_order_similarity(gt_order, pred_order)

    sys_TPs = len(pred_order) - sys_FPs
    f1 = get_f1_score(FN=sys_FNs, FP=sys_FPs, TP=sys_TPs)

    avg_delay = np.nanmean(delays)
    if np.isnan(avg_delay):
        avg_delay = 100
    metrics = {
        "perception_FPs": per_FPs,
        "perception_FNs": per_FNs,
        "system_FNs": sys_FNs,
        "system_FPs": sys_FPs,
        "system_TPs": sys_TPs,
        "f1": f1,
        "pos": pos,
        "avg_delay": avg_delay,
    }

    if verbose:
        print('-'*29)
        print(f"GT order\t{gt_order}\nPred order\t{pred_order}")
        print(f"GT times\t{gt_obs_times}\nPred times\t{pred_obs_times}")
        print(f"Perception: \tFNs = {metrics['perception_FNs']} \t FPs = {metrics['perception_FPs']}")
        print(f"System: \t\tFNs = {metrics['system_FNs']} \t FPs = {metrics['system_FPs']} \tF1-score = "
              f"{metrics['f1']:.3f}")
        print(f"pos: \t{metrics['pos']}")
        print(f"Average delay: \t{avg_delay} [frames]\t({avg_delay / FPS:.1f} s)")
        print('-' * 29)
    return metrics


def mean_list(l):
    return sum(l)/len(l)


def make_deltat_plot(gt, pred, fs_range=1000, fs_steps=1, threshold=0):
    """
    Plots the % of actions seen against the delay. Excludes FNs and FPs. Can be used to determine e.g., whether xx.x% of
    actions are observed within y seconds.
    """
    delta_fs = [x for x in range(0, fs_range, fs_steps)]
    gt_obs_times = np.zeros(len(gt), dtype=int)
    gt_order = np.zeros(len(gt), dtype=int)
    for i, entry in enumerate(gt):
        gt_obs_times[i] = entry["frame"]
        gt_order[i] = int(entry["id"])

    pred_obs_times = np.zeros(len(pred), dtype=int)
    pred_order = np.zeros(len(pred), dtype=int)
    for i, entry in enumerate(pred):
        pred_obs_times[i] = entry["frame"]
        pred_order[i] = int(entry["id"])

    timely_preds = np.zeros_like(delta_fs)
    for i, delta_f in enumerate(delta_fs):
        n_timely = 0
        delays = np.empty(len(gt_obs_times))
        delays[:] = np.nan
        for idx_gt, id in enumerate(gt_order):
            idx_pred = np.where(np.array(pred_order) == id)[0]
            if len(idx_pred) == 1:
                idx_pred = int(idx_pred[0])
                gt_frame_n = gt_obs_times[idx_gt]
                pred_frame_n = pred_obs_times[idx_pred]
                delta_frames = pred_frame_n - gt_frame_n
                # determine average delay and timely predictions
                if delta_frames >= -threshold:  # if True, prediction is not false positive
                    delays[idx_gt] = abs(pred_frame_n - gt_frame_n)
                    if delta_frames <= delta_f:
                        n_timely += 1
        timely_preds[i] = n_timely  # /

    total_actions = np.count_nonzero(~np.isnan(delays)) * 100  # counts timely predictions excluding nans

    plt.figure()
    plt.plot(np.array(delta_fs) / FPS, timely_preds / total_actions * 100, 'b', lw=3)
    plt.grid()
    # plt.legend()
    plt.xlabel("Delta t [s]")
    plt.ylabel("Actions observed [%]")
    plt.title("Note: FNs and FPs not included")
    plt.show()

    return timely_preds, total_actions


def flatten_list(l):
    return [item for sublist in l for item in sublist]


def load_raw_psr_csv(file: Path) -> list:
    with open(file) as fp:
        reader = csv.reader(fp, delimiter=",", quotechar='"')
        # next(reader, None)  # skip the headers
        data_read = []
        for i, row in enumerate(reader):
            frame_name = row[0]
            state_str = row[1:]
            state = [int(k) for k in state_str]
            data_read.append([frame_name, state])
    return data_read


def convert_states_to_steps(prev: list, curr: list, frame: int, proc_info: list, conf=None) -> list:
    actions = []
    n_error_steps = 0
    for k, (prev_state, curr_state) in enumerate(zip(prev, curr)):
        if prev_state == curr_state:
            continue
        elif prev_state == -1 and curr_state == 0:  # ignore: undoing something wrong is not completing a step
            continue
        elif prev_state == -1 and curr_state == 1:  # correctly assembled from error state
            action_id = k * 3 + 0
        elif prev_state == 0 and curr_state == -1:  # incorrectly assembling something
            action_id = k * 3 + 1
            n_error_steps += 1
        elif prev_state == 0 and curr_state == 1:  # correctly assembling something from normal state
            action_id = k * 3 + 0
        elif prev_state == 1 and curr_state == -1:  # incorrectly assembly/removing from correct state
            print(f"Warning: did not expect someone going from 1 to -1!!")
            n_error_steps += 1
            action_id = k * 3 + 1
        elif prev_state == 1 and curr_state == 0:  # correctly removing something
            action_id = k * 3 + 2
        entry = make_entry(frame, action_id, proc_info, conf)
        actions.append(entry)
    return actions, n_error_steps


def only_positive_states(states):
    return [0 if num == -1 else num for num in states]


def convert_all_states_to_steps(observed, proc_info, include_errors=False):
    n_errors = 0
    actions = []
    for i in range(1, len(observed)):
        frame = observed[i][0]
        prev_states = observed[i-1][1]
        curr_states = observed[i][1]
        if not include_errors:
            prev_states = only_positive_states(prev_states)
            curr_states = only_positive_states(curr_states)
        entries, n = convert_states_to_steps(prev_states, curr_states, frame, proc_info)
        n_errors += n
        for entry in entries:
            actions.append(entry)
    return actions, n_errors


def load_psr_labels(file_path: Path) -> list:
    with open(file_path) as fp:
        reader = csv.reader(fp, delimiter=",", quotechar='"')
        data_read = []
        for i, row in enumerate(reader):
            frame = int(row[0][:-4])
            action_id = int(row[1])
            description = row[2]
            entry = {
                "frame": frame,
                "id": action_id,
                "description": description,
            }
            data_read.append(entry)
    return data_read


def video_contains_errors(gt: list, proc_info: list, rec: Path) -> bool:
    """ if all expected actions are correctly observed AND we do not observe any incorrect step completion,
    we assume no errors occured in the video """
    if 'assy' in rec.name:
        expected_actions = [action["id"] for action in proc_info if action["expected_in_assy"]]
    elif 'main' in rec.name:
        expected_actions = [action["id"] for action in proc_info if action["expected_in_main"]]
    else:
        raise ValueError(f"Unable to determine whether assembly or maintenance procedure: {rec.name}")

    observed_actions = [action["id"] for action in gt]
    for expected_id in expected_actions:
        if expected_id not in observed_actions:  # if step completion not observed, it is not completed (so wrong)
            return True

    # if reached here, all expected actions were observed. Now, time to check if there were no incorrect completions
    # since we do not actively check for the prediction of incorrect completions, we need to load our raw state labels.
    raw_labels_path = rec / "PSR_labels_raw.csv"
    psr_labels_raw = load_raw_psr_csv(raw_labels_path)
    for _, (_, state) in enumerate(psr_labels_raw):
        if state.count(-1) > 0:
            return True
    # if reached here, we don't have any incorrectly completed or missing procedure steps
    return False


def save_psr_labels(labels, file_path):
    file = open(str(file_path), 'w')
    for entry in labels:
        line = f"{entry['frame']},{entry['id']},{entry['description']}\n"
        file.write(line)
    file.close()
    print(f"Successfully wrote the PSR labels to {file_path}")


def get_recording_list(folder: Path, train=False, val=False, test=False) -> list:
    assert [train, val, test].count(True) < 2, f"You can currently only retrieve one set or all sets, not two. For " \
                                               f"all sets, simply do not specify a set."
    if train:
        sets = ['train']
    elif val:
        sets = ['val']
    elif test:
        sets = ['test']
    else:
        sets = ['train', 'val', 'test']
    recordings = []
    for set in sets:
        recordings.append([Path(f.path) for f in os.scandir(folder / set) if f.is_dir()])
    return flatten_list(recordings)


def get_procedure_info(file) -> list:
    with open(str(file), "r") as read_file:
        procedure_info = json.load(read_file)
    return procedure_info


def process_asd_predictions(file: Path, n_frames: int) -> list:
    data_read = [[i, []] for i in range(n_frames)]
    with open(file) as fp:
        reader = csv.reader(fp, delimiter=",", quotechar='"')
        next(reader, None)  # skip the headers

        for i, row in enumerate(reader):
            frame_n = int(row[1])
            pred_class = int(row[2])
            # if pred_class == 0:
            #     conf = 0.0
            #     x_min, y_min, w, h = 0.0, 0.0, 0.0, 0.0
            # else:
            conf = round(float(row[3]), 3)
            x_min, y_min, w, h = float(row[4]), float(row[5]), float(row[6]), float(row[7])
            frame_preds = [pred_class, conf, [x_min, y_min, w, h]]
            data_read[frame_n][1].append(frame_preds)
    return data_read


def load_asd_predictions(asd_directory: Path, recording: Path, n_frames: int) -> list:
    asd_sub_directories = list(asd_directory.glob("*"))
    rec_set = recording.parent.name
    asd_sub = None
    for sub in asd_sub_directories:
        if rec_set in sub.name:
            asd_sub = sub
    assert asd_sub is not None, f"None of the asd directories contained {rec_set} set: {asd_sub_directories}"
    asd_path = asd_directory / asd_sub / (recording.name + "_results_pred.csv")
    preds = process_asd_predictions(asd_path, n_frames)
    return preds



