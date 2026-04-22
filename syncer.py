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
from collections import defaultdict

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

from pyzbar.pyzbar import decode

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
    files.sort()
    images = []
    for file in files:
        if is_image(folder, file):
            path = os.path.join(folder, file)
            images.append(path)
    return images

def is_timestamp(s):
    from re import match
    
    pattern = r"^(?:[01]\d|2[0-3]):[0-5]\d:[0-5]\d.(\d+)$"
    return bool(match(pattern, str(s)))

def read_qrcode(filepath):
    from qreader import QReader
    from cv2 import imread, cvtColor

    qreader = QReader()
    img = imread(filepath)
    if img is None:
        print("read_qrcode(): Cannot read file {}".format(filepath))
        return ''

    def find_ts(decoded):
        for obj in decoded:
            if is_timestamp(obj):
                print("read_qrcode(): QR Code found successfully. Content '{}'".format(obj))
                return obj
        return ''
    
    decoded_text = qreader.detect_and_decode(image=img)
    result = find_ts(decoded_text)
    if result:
        return result
    
    print("read_qrcode(): RGB failed.  Trying Grayscale")
    img = cvtColor(img, cv2.COLOR_BGR2GRAY)
    decoded_text = qreader.detect_and_decode(image=img)
    result = find_ts(decoded_text)
    if result:
        return result
    
    print("read_qrcode(): Failed to read QR Code. Trying cropping...")
    height, width = img.shape
    window_size = int(max(height, width) * .25)
    overlap = int(window_size * 0.4)
    step = window_size - overlap
    
    for y in range(0, height, step):
        for x in range(0, width, step):
            # Calculate the boundaries for the current chunk
            y_end = min(y + window_size, height)
            x_end = min(x + window_size, width)
            
            chunk = img[y:y_end, x:x_end]
            
            # Scan the chunk
            chunk_decoded = qreader.detect_and_decode(image=chunk)

            result = find_ts(chunk_decoded)
            if result:
                return result
                    
    return ''

def read_timestamp(images):
    from tqdm import tqdm
    texts = []
    for image in tqdm(images):
        texts.append(read_qrcode(image))

    print(texts)
    return texts

def get_shared_timestamps(imgs1, imgs2):
    mp = defaultdict(list)
    
    for i, ts in enumerate(read_timestamp(imgs1)):
        if ts: # Only add to dict if a timestamp was actually found
            mp[ts].append((i, 1))
            
    for i, ts in enumerate(read_timestamp(imgs2)):
        if ts:
            mp[ts].append((i, 2))
            
    shared_ts = [(value[0][0], value[1][0]) for value in mp.values() if len(value) == 2]
    return shared_ts

def get_time_by_index(index: int, skip_frames: int, frame_rate: int):
    index *= skip_frames
    return index / frame_rate
    
def pivot_time(folder1, folder2, skip_frames, frame_rate) -> Tuple[float, float]:
    imgs1 = get_images(folder1)[::skip_frames]
    imgs2 = get_images(folder2)[::skip_frames]
    
    shared_ts = get_shared_timestamps(imgs1, imgs2)
    if shared_ts:
        t1 = get_time_by_index(shared_ts[0][0], skip_frames, frame_rate)
        t2 = get_time_by_index(shared_ts[0][1], skip_frames, frame_rate)
        return (t1, t2)
    return (-1, -1)

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
    
    t1, t2 = pivot_time(folder1, folder2, skip_frames, frame_rate)

    print("SUCCESS: {} {}".format(t1, t2))
