"""
The `zbar` shared library is needed for the script to run.  It
is included with the Windows Python wheels.  For other operating
systems, extra configuration is needed.

Installation commands are listed below.

Mac OS X:

```
brew install zbar
```

Linux:

```
sudo apt-get install libzbar0
```
"""

import os
import argparse
import filetype
import cv2
import sys
from typing import Tuple, Optional
from PIL import Image

# Pyzbar fails to find libzbar on MacOS
import ctypes.util
_original_find_library = ctypes.util.find_library
def patched_find_library(name):
    if name == "zbar":
        mac_silicon_path = '/opt/homebrew/lib/libzbar.dylib'
        if os.path.exists(mac_silicon_path):
            return mac_silicon_path
    return _original_find_library(name)
ctypes.util.find_library = patched_find_library

def build_parser(argv):
    parser = argparse.ArgumentParser(prog="Video Synchronizer")
    parser.add_argument("folder1", help="Path to the first image folder")
    parser.add_argument("folder2", help="Path to the second image folder")
    parser.add_argument("-s", "--skip-frames", type=int, default=1, help="Process every Nth frame")
    parser.add_argument("-f", "--frame-rate", type=int, default=30, help="Frames per second of both videos")
    return parser

def is_image(path, file):
    kind = filetype.guess(os.path.join(path, file))
    if kind is None:
        return False
    return kind.mime.startswith("image/")

def get_images(folder):
    files = os.listdir(folder)
    files = [file for file in files if os.path.isfile(os.path.join(folder, file))]
    files.sort()
    images = []
    for file in files:
        if is_image(folder, file):
            path = os.path.join(folder, file)
            images.append(path)
    return images

def is_timestamp(s):
    from re import match
    
    pattern = r"^(?:[01]\d|2[0-3]):[0-5]\d:[0-5]\d\.(\d+)$"
    return bool(match(pattern, str(s)))

from qreader import QReader
# qreader = cv2.wechat_qrcode.WeChatQRCode("detect.prototxt", "detect.caffemodel", "sr.prototxt", "sr.caffemodel")
qreader = QReader()
def read_qrcode(filepath):
    from cv2 import imread, cvtColor

    img = imread(filepath)
    if img is None:
        print("read_qrcode(): Cannot read file {}".format(filepath))
        return ''

    def find_ts(decoded):
        for obj in decoded:
            if is_timestamp(obj):
                # print("read_qrcode(): QR Code found successfully. Content '{}'".format(obj))
                return obj
        return ''
    
    decoded_text = qreader.detect_and_decode(img)
    if result := find_ts(decoded_text):
        return result
    
    # print("read_qrcode(): RGB failed.  Trying Grayscale")
    img = cvtColor(img, cv2.COLOR_BGR2GRAY)
    decoded_text = qreader.detect_and_decode(img)
    if result := find_ts(decoded_text):
        return result
    
    # print("read_qrcode(): Failed to read QR Code. Trying cropping...")
    height, width = img.shape
    window_size = int(max(height, width) * .25)
    overlap = int(window_size * 0.4)
    step = window_size - overlap
    
    for y in range(0, height, step):
        for x in range(0, width, step):
            y_end = min(y + window_size, height)
            x_end = min(x + window_size, width)
            
            chunk = img[y:y_end, x:x_end]
            
            chunk_decoded = qreader.detect_and_decode(chunk)

            if result := find_ts(chunk_decoded):
                return result

    # print("read_qrcode(): Failed to read QR Code.")
    return ''

def read_timestamp(images):
    from tqdm import tqdm
    texts = []
    for image in tqdm(images):
        texts.append(read_qrcode(image))

    print("read_timestamp(): Read {}".format(texts))
    return texts

def get_shared_timestamps(images1, images2):
    mp = {}

    ts1 = read_timestamp(images1)
    ts2 = read_timestamp(images2)
    
    for i, ts in enumerate(ts1):
        if ts and ts not in mp: # Only add to dict if a timestamp was actually found
            mp[ts] = [i, None]
            
    for i, ts in enumerate(ts2):
        if ts and ts in mp:
            mp[ts][1] = i
            
    shared_ts = [value for value in mp.values() if value[1] is not None]
    return shared_ts

def get_time_by_index(index: int, skip_frames: int, frame_rate: int):
    index *= skip_frames
    return index / frame_rate

def time_offset(folder1, folder2, skip_frames, frame_rate) -> Optional[float]:
    """
    Returns:
        The time video 2 is ahead of video 1, in seconds.
        -ive means behind.
    """
    images1 = get_images(folder1)[::skip_frames]
    images2 = get_images(folder2)[::skip_frames]
    
    shared_ts = get_shared_timestamps(images1, images2)
    if shared_ts:
        offsets = []
        for t1, t2 in shared_ts:
            t1 = get_time_by_index(t1, skip_frames, frame_rate)
            t2 = get_time_by_index(t2, skip_frames, frame_rate)
            offsets.append(t2 - t1)
        if len(offsets) > 10:
            offsets = sorted(offsets)[1:-1]
            return sum(offsets) / len(offsets)
        return sorted(offsets)[len(offsets)//2]
    return None

def run_test(test_path="./tests"):
    files = os.listdir(test_path)
    for file in files:
        fp = os.path.join(test_path, file)
        print("TEST: QR Reader resulted in '{}', in file {}".format(read_qrcode(fp), fp))

if __name__ == "__main__":

    run_test()

    parser = build_parser(sys.argv)
    args = parser.parse_args()

    folder1 = args.folder1
    folder2 = args.folder2
    skip_frames = args.skip_frames
    frame_rate = args.frame_rate
    
    offset = time_offset(folder1, folder2, skip_frames, frame_rate)

    if offset is None:
        print("FAIL: Could not sync {} and {}".format(folder1, folder2))
    else:
        print("SUCCESS: <{}> is {} seconds behind <{}>".format(folder1, offset, folder2))
