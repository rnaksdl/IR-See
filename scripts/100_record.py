#!/usr/bin/env python3

from picamera2 import Picamera2
from picamera2.encoders import H264Encoder
from picamera2.outputs import FileOutput
from libcamera import Transform
import time
import os
import cv2
import numpy as np
import threading
import sys
import subprocess
from datetime import datetime
import shutil
import gc

# Create output folder
output_folder = "../input"
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

picam2 = Picamera2()

# Camera configuration with reduced brightness and increased contrast
video_config = picam2.create_video_configuration(
    main={"size": (4056, 3040)},  # Lower resolution for testing
    controls={
        "Brightness": -1.0,  # -1 to 1 (0 default)
        "AeEnable": False,   # Auto Exposure off
        "AnalogueGain": 1.0 # 1 to 16
        # "Contrast": 1.0,     # 0 to 32 (1 default)
        # "Saturation": 1.0,   # 0 to 32 (1 default)
        # "Sharpness": 16.0    # 0 to 16 (1 default)
    },
    transform=Transform(hflip=1)
)
picam2.configure(video_config)

encoder = H264Encoder(bitrate=10000000)  # High bitrate for quality recording

picam2.start_preview(True)
picam2.start()

recording = False
camera_started = True  # Track if camera is started
temp_filename = ""
start_time = 0
stop_thread = threading.Event()
duration_thread = None

# Gamma correction function
def apply_gamma_correction(frame, gamma=0.5):
    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(frame, table)

# Mask non-light areas
def mask_non_light_areas(frame, threshold=200):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
    return cv2.bitwise_and(frame, frame, mask=mask)

# Process each frame to emphasize light sources
def process_frame(frame):
    # Apply gamma correction
    frame = apply_gamma_correction(frame, gamma=0.5)
    # Mask non-light areas
    frame = mask_non_light_areas(frame, threshold=200)
    return frame

def display_duration():
    while recording and not stop_thread.is_set():
        elapsed = time.time() - start_time
        sys.stdout.write(f"\rRecording duration: {elapsed:.1f}s")
        sys.stdout.flush()
        time.sleep(0.1)

def convert_to_mp4(h264_path, mp4_path, fps=30):
    print(f"Converting {h264_path} to {mp4_path} using ffmpeg...")
    cmd = [
        "ffmpeg", "-y", "-framerate", str(fps),
        "-i", h264_path,
        "-c", "copy",
        mp4_path
    ]
    try:
        subprocess.run(cmd, check=True)
        print(f"Conversion complete: {mp4_path}")
    except subprocess.CalledProcessError as e:
        print(f"ffmpeg conversion failed: {e}")

print("Recording System (720p@30fps, Preview ON)")
print("Commands:")
print("  1 - Start recording")
print("  2 - Stop recording")
print("  3 - Quit")

try:
    while True:
        command = input("> ")
        
        if command == "1" and not recording:
            temp_filename = f"{output_folder}/temp_recording.h264"
            picam2.start_recording(encoder, FileOutput(temp_filename))
            recording = True
            stop_thread.clear()
            start_time = time.time()
            print("Recording started...")

            duration_thread = threading.Thread(target=display_duration, daemon=True)
            duration_thread.start()
            
        elif command == "2" and recording:
            sys.stdout.write("\n")
            try:
                picam2.stop_recording()
            except Exception:
                pass
            recording = False
            stop_thread.set()
            if duration_thread is not None:
                duration_thread.join(timeout=1)
            
            actual_duration = time.time() - start_time
            seconds = int(actual_duration)
            tenths = int((actual_duration - seconds) * 10)
            duration_str = f"{seconds}_{tenths}s"
            
            timestamp = datetime.now().strftime("%y%m%d_%H%M%S")
            final_filename = f"{output_folder}/{timestamp}_{duration_str}.h264"
            final_mp4 = f"{output_folder}/{timestamp}_{duration_str}.mp4"
            
            shutil.move(temp_filename, final_filename)
            # print(f"Saved as {final_filename}")  # <-- Commented out, don't print .h264 filename

            # Convert to mp4
            convert_to_mp4(final_filename, final_mp4, fps=30)

            # Remove the .h264 file after conversion
            if os.path.exists(final_filename):
                os.remove(final_filename)
            print(f"Saved as {final_mp4}")

            
        elif command == "3":
            if recording:
                sys.stdout.write("\n")
                try:
                    picam2.stop_recording()
                except Exception:
                    pass
                recording = False
                stop_thread.set()
                if duration_thread is not None:
                    duration_thread.join(timeout=1)
            print("Exiting...")
            break
            
        else:
            print("Unknown command")
                
except KeyboardInterrupt:
    print("\nProgram interrupted")
finally:
    # Only stop recording if still recording
    if recording:
        try:
            picam2.stop_recording()
        except Exception:
            pass
        recording = False
        stop_thread.set()
        if duration_thread is not None:
            duration_thread.join(timeout=1)

    # Stop preview before stopping camera
    try:
        picam2.stop_preview()
    except Exception:
        pass

    # Now stop the camera
    try:
        picam2.stop()
    except Exception:
        pass

    # Force garbage collection
    gc.collect()

    print("Camera resources released")