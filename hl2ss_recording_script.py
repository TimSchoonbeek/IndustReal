#------------------------------------------------------------------------------
# This script receives encoded video from the HoloLens cameras and plays it.
# Press esc to stop.
#------------------------------------------------------------------------------
import os
import shutil
import matplotlib.pyplot as plt
import matplotlib as mpl
import threading
import queue

from pynput import keyboard

import numpy as np
import multiprocessing as mp
import cv2
import hl2ss
import hl2ss_mp
import time
from datetime import datetime
from pathlib import Path
import hl2ss_3dcv
import hl2ss_utilities
import itertools

# Settings --------------------------------------------------------------------
host = "put_host_IP_here"  # e.g. 123.456.789.012

save_folder = Path("C:/temp")  # the directory where you want to save the recorded data
VISUALIZE = True
FAST = True
SAVING = True


# Ports
ports = [
    hl2ss.StreamPort.PERSONAL_VIDEO,
    hl2ss.StreamPort.RM_VLC_LEFTFRONT,
    hl2ss.StreamPort.RM_VLC_RIGHTFRONT,
    hl2ss.StreamPort.RM_DEPTH_LONGTHROW,
    hl2ss.StreamPort.SPATIAL_INPUT
    ]

# RM VLC parameters
vlc_mode    = hl2ss.StreamMode.MODE_1
vlc_profile = hl2ss.VideoProfile.H264_BASE
vlc_bitrate = 1*1024*1024

# RM Depth AHAT parameters
ahat_mode = hl2ss.StreamMode.MODE_1
ahat_profile = hl2ss.VideoProfile.H264_BASE
ahat_bitrate = 8*1024*1024

# RM Depth Long Throw parameters
lt_mode = hl2ss.StreamMode.MODE_1
lt_filter = hl2ss.PngFilterMode.Paeth
clip_depth = 750  # distance at which to clip all depth values in mm

# PV parameters
pv_mode = hl2ss.StreamMode.MODE_1
pv_width = 1280
pv_height = 720
pv_framerate = 30
pv_profile = hl2ss.VideoProfile.H265_MAIN
pv_bitrate = 5*1024*1024
pv_format = 'bgr24'

# Maximum number of frames in buffer
buffer_elements = 120

# buffer sizes in seconds. Note that PV sometimes is +- 5 seconds lagging, so allow for room in the other buffers.
buffer_length_pv = 5
buffer_length_lt = 8
buffer_length_lf = 8
buffer_length_lr = 8
buffer_length_si = 8

# timestamp reading has some offset. e.g. 5 FPS = 0.200 s per frame, but recorded as 0.199. We use margin to compensate
desired_FPS = 10

# we don't want to do computationally heavy spatial mapping in real-time, so set gaze distance at a constant value.
gaze_distance = 0.6  # meters

# visualization parameters
gaze_radius = 5
gaze_color = (50, 200, 50)
gaze_thickness = 3
# Marker properties
hand_radius = 3
left_hand_color = (255, 255, 0)
right_hand_color = (0, 255, 255)
hand_thickness = 2

# multi threading
num_threads = 2

#------------------------------------------------------------------------------


def normalize_depth(payload):
    normalized_depth = payload.depth.copy() / 7500 * 255
    return normalized_depth.astype(np.uint8)


def normalize_ab(payload):
    normalized_ab = payload.ab.copy() / np.max(payload.ab) * 255
    return normalized_ab.astype(np.uint8)


def apply_colormap(payload):
    # uses improved Turbo colormap from Google
    depth = payload.depth.copy()
    depth[depth > clip_depth] = clip_depth
    depth = depth / clip_depth
    depth = plt.cm.turbo(depth) * 255
    return depth.astype(np.uint8)


def contains_missing_data(all_data: list):
    for data in all_data:
        if data is None:
            print(f"Found missing data! Data is None.")
            return True
        elif data.payload is None:
            print(f"Found missing data! Payload is None.")
            return True
    return False


def rotate_images(payload, sensor: str):
    if sensor == 'lf':
        payload = cv2.rotate(payload, cv2.ROTATE_90_CLOCKWISE)
    elif sensor == 'rf':
        payload = cv2.rotate(payload, cv2.ROTATE_90_COUNTERCLOCKWISE)
    else:
        print(f"Warning: trying to rotate a sensor that is not programmed to rotate!")
    return payload


def project_points(P, points):
    if len(points.shape) == 1:
        x, y = hl2ss_3dcv.project_to_image(hl2ss_3dcv.to_homogeneous(point), projection)[0]
        return int(x), int(y)
    else:
        image_points = []
        for x, y in hl2ss_3dcv.project_to_image(hl2ss_3dcv.to_homogeneous(points), P)[0]:
            image_points.append((int(x), int(y)))
        return image_points


def display_basic(name: str, payload: np.ndarray, gaze: tuple = (0, 0), hands: list = [(0, 0)] * 52):
    if name == 'PV':
        # we only copy if we will manipulate image later. to save comp time
        image = payload.copy()
        left_hand = hands[0]
        right_hand = hands[1]
        if gaze is not None and gaze[0] != 0:  # gaze is only zero if gaze unavailable
            cv2.circle(image, (gaze[0], gaze[1]), gaze_radius, gaze_color, gaze_thickness)
        if left_hand is not None and left_hand[0] != 0:  # hand coordinate is only zero if hand data unavailable
            for (x, y) in left_hand:
                cv2.circle(image, (x, y), hand_radius, left_hand_color, hand_thickness)
        if right_hand is not None and right_hand[0] != 0:  # hand coordinate is only zero if hand data unavailable
            for (x, y) in right_hand:
                cv2.circle(image, (x, y), hand_radius, right_hand_color, hand_thickness)
        cv2.imshow(name, image)
    else:
        cv2.imshow(name, payload)


def save_image():
    while True:
        image, image_path = image_queue.get()
        cv2.imwrite(str(image_path), image)
        image_queue.task_done()


def unpack_list_of_tuples(lst):
    return list(itertools.chain.from_iterable(lst))


def convert_hand_detection_to_csv(img_name: str, left_hand_points: tuple, right_hand_points: tuple):
    left_hand_str = ','.join(map(str, unpack_list_of_tuples(left_hand_points)))
    right_hand_str = ','.join(map(str, unpack_list_of_tuples(right_hand_points)))
    return img_name + ',' + left_hand_str + ',' + right_hand_str + '\n'


def convert_gaze_to_csv(img_name: str, gaze: tuple):
    gaze_str = str(gaze[0]) + ',' + str(gaze[1])
    return img_name + ',' + gaze_str + '\n'


def convert_pose_to_cvs(img_name: str, pose):
    if head_pose is None:
        return img_name + ',0,0,0,0,0,0,0,0,0\n'
    else:
        list_pose_np = [pose.forward, pose.position, pose.up]
        list_pose = np.concatenate(list_pose_np, axis=0).tolist()
        list_pose_rounded = ['%.3f' % elem for elem in list_pose]
        pose_str = ','.join(map(str, list_pose_rounded))
        return img_name + ',' + pose_str + '\n'


if __name__ == '__main__':
    if ((hl2ss.StreamPort.PERSONAL_VIDEO in ports) and (hl2ss.StreamPort.RM_DEPTH_AHAT in ports)):
        print('Error: Simultaneous PV and RM Depth AHAT streaming is not supported. See known issues at https://github.com/jdibenes/hl2ss.')
        quit()

    if ((hl2ss.StreamPort.RM_DEPTH_LONGTHROW in ports) and (hl2ss.StreamPort.RM_DEPTH_AHAT in ports)):
        print('Error: Simultaneous RM Depth Long Throw and RM Depth AHAT streaming is not supported. See known issues at https://github.com/jdibenes/hl2ss.')
        quit()

    client_rc = hl2ss.tx_rc(host, hl2ss.IPCPort.REMOTE_CONFIGURATION)


    # timestamp in HL2 is in NTFS format but unix makes more sense in python.
    UTC_offset_NTFS = client_rc.get_utc_offset(32)
    unix_timestamp_in_NTFS = 11644473600
    UTC_offset = UTC_offset_NTFS // 1e7 - unix_timestamp_in_NTFS

    if (hl2ss.StreamPort.PERSONAL_VIDEO in ports):
        hl2ss.start_subsystem_pv(host, hl2ss.StreamPort.PERSONAL_VIDEO)
        client_rc.wait_for_pv_subsystem(True)

    calibration = hl2ss.download_calibration_pv(host, hl2ss.StreamPort.PERSONAL_VIDEO, pv_width, pv_height,
                                                pv_framerate)

    producer = hl2ss_mp.producer()
    producer.configure_rm_vlc(True, host, hl2ss.StreamPort.RM_VLC_LEFTFRONT, hl2ss.ChunkSize.RM_VLC, vlc_mode, vlc_profile, vlc_bitrate)
    producer.configure_rm_vlc(True, host, hl2ss.StreamPort.RM_VLC_RIGHTFRONT, hl2ss.ChunkSize.RM_VLC, vlc_mode, vlc_profile, vlc_bitrate)
    producer.configure_rm_depth_longthrow(True, host, hl2ss.StreamPort.RM_DEPTH_LONGTHROW, hl2ss.ChunkSize.RM_DEPTH_LONGTHROW, lt_mode, lt_filter)
    producer.configure_pv(True, host, hl2ss.StreamPort.PERSONAL_VIDEO, hl2ss.ChunkSize.PERSONAL_VIDEO, pv_mode, pv_width, pv_height, pv_framerate, pv_profile, pv_bitrate, pv_format)
    producer.configure_si(host, hl2ss.StreamPort.SPATIAL_INPUT, hl2ss.ChunkSize.SPATIAL_INPUT)

    producer.initialize(hl2ss.StreamPort.PERSONAL_VIDEO, buffer_length_pv * pv_framerate)
    producer.initialize(hl2ss.StreamPort.RM_DEPTH_LONGTHROW, buffer_length_lt * hl2ss.Parameters_RM_DEPTH_LONGTHROW.FPS)
    producer.initialize(hl2ss.StreamPort.RM_VLC_LEFTFRONT, buffer_length_lf * hl2ss.Parameters_RM_VLC.FPS)
    producer.initialize(hl2ss.StreamPort.RM_VLC_RIGHTFRONT, buffer_length_lr * hl2ss.Parameters_RM_VLC.FPS)
    producer.initialize(hl2ss.StreamPort.SPATIAL_INPUT, buffer_length_si * hl2ss.Parameters_SI.SAMPLE_RATE)

    for port in ports:
        producer.start(port)

    manager = mp.Manager()
    consumer = hl2ss_mp.consumer()
    sinks = {}

    for port in ports:
        sinks[port] = consumer.create_sink(producer, port, manager, None)
        sinks[port].get_attach_response()

    enable = True

    def on_press(key):
        global enable
        enable = key != keyboard.Key.esc
        return enable

    listener = keyboard.Listener(on_press=on_press)
    listener.start()

    # create the proper folders for saving
    FOLDERS = {
        'pv' : save_folder / "rgb",
        'lt' : save_folder / "depth",
        'ab' : save_folder / "ambient_light",
        'lf' : save_folder / "stereo_left",
        'rf' : save_folder / "stereo_right"
    }

    # ans = input(f"Did you name this recording correctly? Are you sure it will not overwrite anything?")

    if save_folder.exists() and SAVING:
        if len(os.listdir(save_folder)) > 0:
            ans = input(f"Output folder {str(save_folder)} is not empty. Do you want to delete current content? (y/n)")
            if ans == 'y':
                shutil.rmtree(save_folder)  # deletes the whole folder + content
                save_folder.mkdir(parents=True, exist_ok=True)  # creates empty folder
                print(f"Deleted content.")
            else:
                print(f"Not deleting content. Be aware: this might cause issues!")
    else:
        save_folder.mkdir(parents=True, exist_ok=True)

    if SAVING:
        # make the folder directories
        for folder in FOLDERS:
            FOLDERS[folder].mkdir(parents=True, exist_ok=True)
        # load the .csv files for hand and eye tracking
        f_hands = open(str(save_folder / "hands.csv"), 'w')
        f_gaze = open(str(save_folder / "gaze.csv"), 'w')
        f_pose = open(str(save_folder / "pose.csv"), 'w')

    # multi threading parameters
    image_queue = queue.Queue()
    threads = list()
    for i in range(num_threads):
        thread = threading.Thread(target=save_image)
        thread.daemon = True
        threads.append(thread)
        thread.start()

    total_c_time = []
    old_timestamp = 0
    save_counter = 0
    t_start = time.time()
    fps_collector = []
    t_depth_colmap = []
    t_rotate_VLC = []
    t_project = []
    t_save = []

    next_timestamp = 0
    warming_up = True
    while warming_up:
        data_pv = sinks[hl2ss.StreamPort.PERSONAL_VIDEO].get_most_recent_frame()
        if data_pv is not None:
            next_timestamp = data_pv.timestamp + 0.1e7
            warming_up = False

    while (enable):
        # most_recent_frame = sinks[hl2ss.StreamPort.PERSONAL_VIDEO].get_frame_stamp()
        most_recent_timestamp = sinks[hl2ss.StreamPort.PERSONAL_VIDEO].get_most_recent_frame().timestamp
        if most_recent_timestamp < next_timestamp:  # meaning: if we're trying to fetch a frame in future
            continue
        frame_stamp, data_pv = sinks[hl2ss.StreamPort.PERSONAL_VIDEO].get_nearest(next_timestamp)
        if data_pv is None:
            continue

        timestamp_unix = datetime.fromtimestamp(UTC_offset + ((data_pv.timestamp) / 1e7))
        current_time_unix = datetime.now()
        t_diff_stamps = (data_pv.timestamp - old_timestamp) / 1e7  # difference in seconds
        old_timestamp = data_pv.timestamp
        next_timestamp = data_pv.timestamp + 0.1e7
        new_buffered_frames = sinks[hl2ss.StreamPort.PERSONAL_VIDEO].get_frame_stamp() - frame_stamp
        # print(f"Loading frame n: {frame_n}. Most recent frame: {most_recent_frame}. frame_stamp: {frame_stamp}. Timestamp: {data_pv.timestamp}")
        f_stamp = sinks[hl2ss.StreamPort.PERSONAL_VIDEO].get_frame_stamp()
        if (data_pv is not None):
            t1 = time.time()
            if save_counter % 2 == 0:
                # depth sensor runs at max 5FPS, so only update at half PV rate.
                data_lt = sinks[hl2ss.StreamPort.RM_DEPTH_LONGTHROW].get_nearest(data_pv.timestamp)[1]
            data_lf = sinks[hl2ss.StreamPort.RM_VLC_LEFTFRONT].get_nearest(data_pv.timestamp)[1]
            data_rf = sinks[hl2ss.StreamPort.RM_VLC_RIGHTFRONT].get_nearest(data_pv.timestamp)[1]
            data_si = sinks[hl2ss.StreamPort.SPATIAL_INPUT].get_nearest(data_pv.timestamp)[1]

            # perform check on data: make sure data and payloads are not none.
            if contains_missing_data([data_pv, data_lt, data_lf, data_rf, data_si]):
                continue

            # process LT sensor data
            # depth_normalized = normalize_depth(data_lt.payload)
            if save_counter % 2 == 0:
                # depth sensor runs at max 5FPS, so only update at half PV rate.
                t_s1 = time.time()
                depth_colmap = apply_colormap(data_lt.payload)
                t_depth_colmap.append(time.time() - t_s1)
                ab_normalized = normalize_ab(data_lt.payload)

            # process VLC sensor data
            t_s2 = time.time()
            data_lf.payload = rotate_images(data_lf.payload, 'lf')
            data_rf.payload = rotate_images(data_rf.payload, 'rf')
            t_rotate_VLC.append(time.time() - t_s2)

            # project hand and gaze detection on PV sensor
            t_s3 = time.time()
            if hl2ss.is_valid_pose(data_pv.pose) and (data_si is not None):
                projection = hl2ss_3dcv.projection(calibration.intrinsics, hl2ss_3dcv.world_to_reference(data_pv.pose))
                si = hl2ss.unpack_si(data_si.payload)
                # Left hand
                if si.is_valid_hand_left():
                    left_hand_coordinates = project_points(projection, hl2ss_utilities.si_unpack_hand(si.get_hand_left()).positions)
                else:
                    # if coordinates are not available, return all zeros
                    left_hand_coordinates = [(0, 0)] * 26
                # right hand
                if si.is_valid_hand_right():
                    right_hand_coordinates = project_points(projection, hl2ss_utilities.si_unpack_hand(si.get_hand_right()).positions)
                else:
                    # if coordinates are not available, return all zeros
                    right_hand_coordinates = [(0, 0)] * 26
                # gaze
                if si.is_valid_eye_ray():
                    eye_ray = si.get_eye_ray()
                    point = eye_ray.origin + eye_ray.direction * gaze_distance
                    gaze = project_points(projection, point)
                else:
                    gaze = (0, 0)
                # head pose
                if si.is_valid_head_pose():
                    head_pose = si.get_head_pose()
                else:
                    head_pose = None
            t_project.append(time.time() - t_s3)
            if VISUALIZE:
                if FAST:
                    display_basic('PV', data_pv.payload)
                else:
                    display_basic('PV', data_pv.payload, gaze=gaze, hands=(left_hand_coordinates, right_hand_coordinates))
                    # display_basic('Depth', depth_normalized)
                    display_basic('Depth', depth_colmap)
                    display_basic('Ambient light', ab_normalized)
                    display_basic('Left front', data_lf.payload)
                    display_basic('Right front', data_rf.payload)

            if SAVING:
                t_s4 = time.time()
                save_name = str(save_counter).zfill(6) + '.jpg'
                image_queue.put((data_pv.payload, FOLDERS['pv'] / save_name))
                image_queue.put((depth_colmap, FOLDERS['lt'] / save_name))
                image_queue.put((ab_normalized, FOLDERS['ab'] / save_name))
                image_queue.put((data_lf.payload, FOLDERS['lf'] / save_name))
                image_queue.put((data_rf.payload, FOLDERS['rf'] / save_name))

                f_hands.write(convert_hand_detection_to_csv(save_name, left_hand_coordinates, right_hand_coordinates))
                f_gaze.write(convert_gaze_to_csv(save_name, gaze))
                f_pose.write(convert_pose_to_cvs(save_name, head_pose))
                t_save.append(time.time() - t_s4)

            if save_counter != 0:
                total_c_time.append(time.time() - t1)
                fps_collector.append(1 / (t_diff_stamps+1e-5))

                print(f"Difference in timestamps: {t_diff_stamps:.3f} seconds --> {1 / (t_diff_stamps + 1e-5):.1f} FPS. "
                      f"Computation time: {total_c_time[-1]:.4f}s\tQueue size: {image_queue.qsize()}. HL2SS buffered "
                      f"frames: {new_buffered_frames}")
            save_counter += 1
            cv2.waitKey(1)

    def mean(lst):
        return sum(lst) / len (lst)

    print(f"Mean computation time: {mean(total_c_time):.3f}s per frame ({1/mean(total_c_time):.1f} FPS)")
    print(f"Mean actual FPS: {mean(fps_collector):.2f}")

    for port in ports:
        sinks[port].detach()

    for port in ports:
        producer.stop(port)

    if (hl2ss.StreamPort.PERSONAL_VIDEO in ports):
        hl2ss.stop_subsystem_pv(host, hl2ss.StreamPort.PERSONAL_VIDEO)
        client_rc.wait_for_pv_subsystem(False)

    if SAVING:
        f_hands.close()
        f_gaze.close()
        f_pose.close()

    listener.join()
    print(f"Properly closed all connection and open files.")
