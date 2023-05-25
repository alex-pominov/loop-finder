#! /usr/bin/env python

import cv2
import time
import numpy as np
from datetime import timedelta
# from skimage.metrics import structural_similarity as compare_ssim


class LogColors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


def compute_pixel_difference(frame_a, frame_b):
    """Compare two gray scale frames"""

    # The score represents the structural similarity index between the two input images.
    # This value can fall into the range [-1, 1] with a value of one being a “perfect match”.
    # (score, diff) = compare_ssim(frame_a, frame_b, full=True)
    # diff = (diff * 255).astype("uint8")
    # return -score + 1

    return np.absolute(frame_a - frame_b).mean()
    # return cv2.absdiff(frame_a, frame_b).mean()


def parse_frames(video, resolution):
    """Parse video frames and apply filters on it"""
    frames = list()
    start_time = time.monotonic()

    video_duration = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    temp_pos = video.get(cv2.CAP_PROP_POS_FRAMES)
    video.set(cv2.CAP_PROP_POS_FRAMES, 0)

    # to handle empty/bugged frames
    prev_frame = None

    for frame_idx in range(video_duration - 1):
        frame = _parse_next_frame(video, resolution, frame_idx)
        if frame is not None:
            prev_frame = frame
            frames.append(frame)

            print(
                "{}[{}] Parse frames progress: {}%"
                .format(LogColors.OKCYAN, (timedelta(seconds=round(time.monotonic() - start_time))),
                        round(frame_idx / video_duration * 100, 1)),
                end="\r",
                flush=True
            )
        else:
            frames.append(prev_frame)
    print(LogColors.ENDC)  # reset colored printing

    video.set(cv2.CAP_PROP_POS_FRAMES, temp_pos)
    return frames


def frame_at_idx(video, resolution, frame_idx, downsample=True, grayscale=True, normalise=True):
    video.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    success, frame = video.read()
    if not success:
        print("Cannot read frame: {}".format(frame_idx))
        return

    if grayscale:
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    if downsample:
        frame = cv2.resize(frame, resolution, interpolation=cv2.INTER_AREA)
    if normalise:
        frame = cv2.normalize(frame.astype(float), None)

    return frame


def _parse_next_frame(video, resolution, frame_idx, downsample=True, grayscale=True, normalise=True):
    success, frame = video.read()
    if not success:
        print("[parse_next_frame] Cannot read frame: {}".format(frame_idx))
        return

    if grayscale:
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    if downsample:
        frame = cv2.resize(frame, resolution, interpolation=cv2.INTER_AREA)
    if normalise:
        frame = cv2.normalize(frame.astype(float), None)
    return frame
