from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2


def extract_frames(video_path: str | Path, output_dir: str | Path, frame_skip: int = 1) -> int:
    """
    Extract frames from a video and save them as JPG files.

    Args:
        video_path: Path to the input video file.
        output_dir: Directory where extracted frames will be written.
        frame_skip: Save every Nth frame. `1` saves every frame.

    Returns:
        The number of frames written to disk.
    """
    if frame_skip < 1:
        raise ValueError("frame_skip must be at least 1")

    input_path = Path(video_path)
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        raise RuntimeError(f"could not open video file at {input_path}")

    frame_count = 0
    saved_count = 0

    print("Extracting frames, please wait...")

    try:
        while True:
            success, frame = cap.read()
            if not success:
                break

            if frame_count % frame_skip == 0:
                filename = destination / f"frame_{saved_count:04d}.jpg"
                if not cv2.imwrite(str(filename), frame):
                    raise RuntimeError(f"failed to write frame to {filename}")
                saved_count += 1

            frame_count += 1
    finally:
        cap.release()

    print(f"Success! Extracted {saved_count} frames to '{destination}'.")
    return saved_count


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Extract JPG frames from a video file.",
    )
    parser.add_argument(
        "input_video",
        help="Path to the input video file.",
    )
    parser.add_argument(
        "output_folder",
        help="Directory where extracted JPG frames will be written.",
    )
    parser.add_argument(
        "--skip",
        type=int,
        default=1,
        metavar="N",
        help="Save every Nth frame instead of every frame.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    input_path = Path(args.input_video)
    if not input_path.exists():
        parser.error(f"input video does not exist: {input_path}")
    if not input_path.is_file():
        parser.error(f"input video is not a file: {input_path}")
    if args.skip < 1:
        parser.error("--skip must be a positive integer")

    try:
        extract_frames(input_path, args.output_folder, frame_skip=args.skip)
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
