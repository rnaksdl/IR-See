'''
for 0_reg/4digit
python 200_batch_adjust.py ../0_reg/4digit/input ../0_reg/4digit/input_e --brightness -0.8 --saturation 3.0 --blackpoint 0.50 --temp-k 18000 --probe -v
'''

#!/usr/bin/env python3
import argparse
import subprocess
import sys
from pathlib import Path
import shlex

def run(cmd, verbose=False, check=False):
    if verbose:
        print("CMD:", " ".join(shlex.quote(x) for x in cmd))
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if verbose and proc.stdout:
        print(proc.stdout.decode(errors="replace"))
    if proc.returncode != 0 or verbose:
        # ffmpeg prints most info to stderr
        if proc.stderr:
            print(proc.stderr.decode(errors="replace"), file=sys.stderr)
    if check and proc.returncode != 0:
        raise RuntimeError(f"Command failed with code {proc.returncode}")
    return proc

def detect_filters(ffmpeg="ffmpeg"):
    out = run([ffmpeg, "-hide_banner", "-filters"], verbose=False)
    text = (out.stdout + out.stderr).decode(errors="replace")
    have = set()
    for line in text.splitlines():
        parts = line.strip().split()
        if len(parts) >= 2:
            # Format like: " TS colortemperature  V->V  Adjust color temperature..."
            name = parts[1]
            have.add(name)
    return have

def build_vf(brightness, saturation, blackpoint, temp_k, filters_available):
    # Always use eq and colorlevels (named options) to avoid the old "levels=" syntax
    chain = [
        f"eq=brightness={brightness}:saturation={saturation}",
        f"colorlevels=rimin={blackpoint}:gimin={blackpoint}:bimin={blackpoint}",
    ]
    if "colortemperature" in filters_available:
        chain.append(f"colortemperature={int(temp_k)}")
    else:
        # Cooler look fallback via colorbalance (reduce red, boost blue)
        chain.append("colorbalance=rs=-0.5:rm=-0.5:rh=-0.5:bs=0.5:bm=0.5:bh=0.5")
    return ",".join(chain)

def main():
    ap = argparse.ArgumentParser(description="Batch adjust MP4s using ffmpeg (brightness, black point, saturation, warmth)")
    ap.add_argument("input_dir", type=Path)
    ap.add_argument("output_dir", type=Path)
    ap.add_argument("--ffmpeg", default="ffmpeg", help="ffmpeg binary to use")
    # Defaults mirror your original (very strong) request; consider dialing down.
    ap.add_argument("--brightness", type=float, default=-1.0, help="eq brightness (-1..1). -1 is nearly black.")
    ap.add_argument("--saturation", type=float, default=3.0, help="eq saturation (1.0 = neutral)")
    ap.add_argument("--blackpoint", type=float, default=0.40, help="colorlevels rimin/gimin/bimin (0..1), raises input black point")
    ap.add_argument("--temp-k", type=int, default=11000, help="colortemperature in Kelvin (higher = cooler). Fallback uses colorbalance if unavailable.")
    ap.add_argument("--crf", type=int, default=18, help="x264 CRF")
    ap.add_argument("--preset", default="medium", help="x264 preset")
    ap.add_argument("--pix-fmt", default="yuv420p", help="pixel format")
    ap.add_argument("--dry-run", action="store_true", help="print commands but do not run ffmpeg")
    ap.add_argument("--probe", action="store_true", help="validate filter chain by running -f null - (no files written)")
    ap.add_argument("--verbose", "-v", action="store_true", help="print commands and ffmpeg output")
    args = ap.parse_args()

    if not args.input_dir.is_dir():
        print(f"Input dir not found: {args.input_dir}", file=sys.stderr)
        sys.exit(2)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    filters_available = detect_filters(args.ffmpeg)
    vf = build_vf(args.brightness, args.saturation, args.blackpoint, args.temp_k, filters_available)

    print("Detected filters contain:")
    for name in ("colorlevels", "colortemperature", "colorbalance", "eq"):
        print(f"  - {name}: {'YES' if name in filters_available else 'no'}")
    print("Using -vf:")
    print(vf)

    # Optionally probe the filter chain on a single file before batch run
    files = sorted(args.input_dir.glob("*.mp4"))
    if not files:
        print(f"No .mp4 files found in {args.input_dir}", file=sys.stderr)
        sys.exit(1)

    if args.probe:
        test_src = str(files[0])
        cmd = [
            args.ffmpeg, "-hide_banner", "-y",
            "-i", test_src,
            "-vf", vf,
            "-frames:v", "1",
            "-f", "null", "-"
        ]
        if args.dry_run or args.verbose:
            print("Probe command (first file):")
        if args.dry_run:
            print(" ".join(shlex.quote(x) for x in cmd))
        else:
            run(cmd, verbose=True, check=True)
        print("Probe successful.")
        if args.dry_run:
            return

    # Process all files
    for src in files:
        dst = args.output_dir / src.name
        print(f"Processing: {src.name}")
        cmd = [
            args.ffmpeg, "-hide_banner", "-loglevel", "error", "-y",
            "-i", str(src),
            "-vf", vf,
            "-c:v", "libx264", "-crf", str(args.crf), "-preset", args.preset,
            "-pix_fmt", args.pix_fmt,
            "-c:a", "copy",
            "-movflags", "+faststart",
            str(dst),
        ]
        if args.dry_run:
            print("DRY RUN:", " ".join(shlex.quote(x) for x in cmd))
            continue
        try:
            run(cmd, verbose=args.verbose, check=True)
            print(f"Done: {dst}")
        except Exception as e:
            print(f"ERROR on {src.name}: {e}", file=sys.stderr)
            # Print a ready-to-copy command to reproduce:
            print("Reproduce with:", " ".join(shlex.quote(x) for x in cmd))
            # Stop on first error so you can inspect
            sys.exit(1)

    print(f"All files written to: {args.output_dir}")

if __name__ == "__main__":
    main()
