"""
images_to_video.py – Render a sequence of images to a video file.

Reads all images matching a glob pattern from an input directory, sorts them
by filename (natural order), and writes them as an MP4 video.

Usage:
    python images_to_video.py -i /path/to/frames -o output.mp4
    python images_to_video.py -i /path/to/frames -o output.mp4 --fps 30
    python images_to_video.py -i /path/to/frames -o output.mp4 --pattern "frame_*.png"
    python images_to_video.py -i /path/to/frames -o output.mp4 --fps 30 --resize 1280 720
"""

import argparse
import glob
import os
import re
import sys

import cv2


# ── Natural sort key ──────────────────────────────────────────────────────────
def natural_sort_key(path: str):
    """Sort filenames with embedded numbers in human order (e.g. frame_9 < frame_10)."""
    parts = re.split(r"(\d+)", os.path.basename(path))
    return [int(p) if p.isdigit() else p.lower() for p in parts]


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Render image sequence to video")
    parser.add_argument("-i", "--input",  required=True, metavar="DIR",
                        help="Directory containing the image frames")
    parser.add_argument("-o", "--output", required=True, metavar="FILE",
                        help="Output video file path (e.g. output.mp4)")
    parser.add_argument("--fps",     type=float, default=30.0,
                        help="Frames per second (default: 30)")
    parser.add_argument("--pattern", default="*",
                        help="Glob pattern to filter images (default: '*')")
    parser.add_argument("--resize",  type=int, nargs=2, metavar=("W", "H"),
                        help="Resize frames to W H before writing")
    args = parser.parse_args()

    if not os.path.isdir(args.input):
        print(f"ERROR: '{args.input}' is not a directory.", file=sys.stderr)
        sys.exit(1)

    # ── Collect image paths ───────────────────────────────────────────────────
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}
    candidates = sorted(
        glob.glob(os.path.join(args.input, args.pattern)),
        key=natural_sort_key,
    )
    image_paths = [p for p in candidates if os.path.splitext(p)[1].lower() in exts]

    if not image_paths:
        print(f"ERROR: No images found in '{args.input}' matching pattern '{args.pattern}'.",
              file=sys.stderr)
        sys.exit(1)

    print(f"Found {len(image_paths)} images  |  fps={args.fps}")

    # ── Determine frame size from first image ─────────────────────────────────
    first = cv2.imread(image_paths[0])
    if first is None:
        print(f"ERROR: Could not read '{image_paths[0]}'.", file=sys.stderr)
        sys.exit(1)

    if args.resize:
        w, h = args.resize
    else:
        h, w = first.shape[:2]

    print(f"Frame size: {w}×{h}")

    # ── Open VideoWriter ──────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(args.output, fourcc, args.fps, (w, h))

    if not writer.isOpened():
        print(f"ERROR: Could not open VideoWriter for '{args.output}'.", file=sys.stderr)
        sys.exit(1)

    # ── Write frames ──────────────────────────────────────────────────────────
    for idx, path in enumerate(image_paths):
        frame = cv2.imread(path)
        if frame is None:
            print(f"WARNING: skipping unreadable frame: {path}", file=sys.stderr)
            continue

        if args.resize:
            frame = cv2.resize(frame, (w, h), interpolation=cv2.INTER_AREA)

        writer.write(frame)

        # Progress every 5% or every 100 frames
        step = max(1, len(image_paths) // 20)
        if (idx + 1) % step == 0 or (idx + 1) == len(image_paths):
            print(f"  {idx+1}/{len(image_paths)}  ({(idx+1)/len(image_paths)*100:.0f}%)",
                  end="\r")

    writer.release()
    print(f"\nDone → {os.path.abspath(args.output)}")


if __name__ == "__main__":
    main()
