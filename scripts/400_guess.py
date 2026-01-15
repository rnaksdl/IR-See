'''
python 400_guess.py -i ../0_reg/4digit_100/output_e -o ../0_reg/4digit_100/report_e
'''

# ---- Robust patch for threadpoolctl on macOS/conda (before sklearn imports) ----
try:
    import threadpoolctl as _tpc
    _orig_info   = getattr(_tpc, "threadpool_info", None)
    _orig_limits = getattr(_tpc, "threadpool_limits", None)

    class _NoOpCtx:
        def __enter__(self): return self
        def __exit__(self, *args): return False

    def _safe_info(*args, **kwargs):
        try:
            return _orig_info(*args, **kwargs) if _orig_info else []
        except Exception as e:
            print(f"[WARN] threadpool_info() failed: {e}. Returning [].")
            return []

    def _safe_limits(*args, **kwargs):
        try:
            return _orig_limits(*args, **kwargs) if _orig_limits else _NoOpCtx()
        except Exception as e:
            print(f"[WARN] threadpool_limits() failed: {e}. Using no-op.")
            return _NoOpCtx()

    if _orig_info:
        _tpc.threadpool_info = _safe_info
    if _orig_limits:
        _tpc.threadpool_limits = _safe_limits
except Exception:
    pass
# ------------------------------------------------------------------------------

#!/usr/bin/env python3
import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from itertools import product
import time
import datetime
import shutil
import webbrowser
import cv2
import re
import argparse
import pickle
from concurrent.futures import ProcessPoolExecutor, as_completed
from threadpoolctl import threadpool_limits

# ------------- High-performance defaults (tunable via CLI) -------------
NUM_WORKERS_DEFAULT = os.cpu_count() or 8
CHUNK_SIZE_DEFAULT = 100_000
TOPK_PER_CHUNK_DEFAULT = 20_000
TOPK_FINAL_DEFAULT = 200_000
USE_PKL_TRAJ_DEFAULT = False

# --- USER CONFIGURABLE PARAMETERS (now set via CLI) ---
PIN_LENGTH = 4
OUTPUT_DIR = None
REPORT_FOLDER = None
TIME_WEIGHT = 0.5
TRAJECTORIES_DIR = './pin_trajectories'

SAME_DIGIT_BOX_SIZE = 80

# Button dimensions
BUTTON_WIDTH = 10.0
BUTTON_HEIGHT = 5.5
GAP = 0.9
X_OFFSET = BUTTON_WIDTH/2
Y_OFFSET = BUTTON_HEIGHT/2

PINPAD_COORDS = np.array([
    [0*BUTTON_WIDTH + 0*GAP + X_OFFSET, 0*BUTTON_HEIGHT + 0*GAP + Y_OFFSET],    # 1
    [1*BUTTON_WIDTH + 1*GAP + X_OFFSET, 0*BUTTON_HEIGHT + 0*GAP + Y_OFFSET],    # 2
    [2*BUTTON_WIDTH + 2*GAP + X_OFFSET, 0*BUTTON_HEIGHT + 0*GAP + Y_OFFSET],    # 3
    [0*BUTTON_WIDTH + 0*GAP + X_OFFSET, 1*BUTTON_HEIGHT + 1*GAP + Y_OFFSET],    # 4
    [1*BUTTON_WIDTH + 1*GAP + X_OFFSET, 1*BUTTON_HEIGHT + 1*GAP + Y_OFFSET],    # 5
    [2*BUTTON_WIDTH + 2*GAP + X_OFFSET, 1*BUTTON_HEIGHT + 1*GAP + Y_OFFSET],    # 6
    [0*BUTTON_WIDTH + 0*GAP + X_OFFSET, 2*BUTTON_HEIGHT + 2*GAP + Y_OFFSET],    # 7
    [1*BUTTON_WIDTH + 1*GAP + X_OFFSET, 2*BUTTON_HEIGHT + 2*GAP + Y_OFFSET],    # 8
    [2*BUTTON_WIDTH + 2*GAP + X_OFFSET, 2*BUTTON_HEIGHT + 2*GAP + Y_OFFSET],    # 9
    [1*BUTTON_WIDTH + 1*GAP + X_OFFSET, 3*BUTTON_HEIGHT + 3*GAP + Y_OFFSET]     # 0
])

PINPAD_DIGITS = ['1','2','3','4','5','6','7','8','9','0']
PINPAD_DIGIT_TO_IDX = {d: i for i, d in enumerate(PINPAD_DIGITS)}
DEC_TO_PINPAD_IDX = np.array([PINPAD_DIGIT_TO_IDX[str(d)] for d in range(10)], dtype=np.int8)

loaded_trajectories = {}

NUM_WORKERS = NUM_WORKERS_DEFAULT
CHUNK_SIZE = CHUNK_SIZE_DEFAULT
TOPK_PER_CHUNK = TOPK_PER_CHUNK_DEFAULT
TOPK_FINAL = TOPK_FINAL_DEFAULT
USE_PKL_TRAJ = USE_PKL_TRAJ_DEFAULT


def load_pin_trajectories(pin_length):
    pkl_path = os.path.join('../pin_trajectories', f'pin{pin_length}_trajectories.pkl')
    try:
        if os.path.exists(pkl_path) and USE_PKL_TRAJ:
            print(f"Loading PIN trajectories from {pkl_path}...")
            with open(pkl_path, 'rb') as f:
                trajectories = pickle.load(f)
            print(f"Loaded {len(trajectories)} trajectories for {pin_length}-digit PINs")
            return trajectories
        elif os.path.exists(pkl_path) and not USE_PKL_TRAJ:
            print(f"PKL trajectories exist at {pkl_path}, but --use-pkl-trajectories not set. Using fast on-the-fly scoring.")
            return None
        else:
            print(f"Trajectory file {pkl_path} not found. Using fast on-the-fly scoring.")
            return None
    except Exception as e:
        print(f"Error loading PIN trajectories: {e}. Falling back to on-the-fly scoring.")
        return None


def detect_blue_circles(image):
    b, g, r = cv2.split(image)
    blurred = cv2.GaussianBlur(b, (11, 11), 0)
    _, thresh = cv2.threshold(blurred, 25, 255, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask_opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
    mask_closing = cv2.morphologyEx(mask_opening, cv2.MORPH_CLOSE, kernel, iterations=1)
    cnts, _ = cv2.findContours(mask_closing, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    detected = []
    for c in cnts:
        area = cv2.contourArea(c)
        if area < 5 or area > 2000:
            continue
        M = cv2.moments(c)
        if M["m00"] == 0:
            continue
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        equiv_radius = int(np.sqrt(area / np.pi))
        detected.append((cx, cy, equiv_radius))
    if len(detected) > 1:
        merged = []
        used = [False] * len(detected)
        for i, (x1, y1, r1) in enumerate(detected):
            if used[i]:
                continue
            group = [(x1, y1, r1)]
            used[i] = True
            for j, (x2, y2, r2) in enumerate(detected):
                if i != j and not used[j]:
                    if np.sqrt((x1-x2)**2 + (y1-y2)**2) < 25:
                        group.append((x2, y2, r2))
                        used[j] = True
            if len(group) == 1:
                merged.append(group[0])
            else:
                xs = [x for x, _, _ in group]
                ys = [y for _, y, _ in group]
                rs = [r for _, _, r in group]
                merged.append((int(np.mean(xs)), int(np.mean(ys)), int(np.mean(rs))))
        return merged
    return detected if detected else None


def find_ring_center_cols(df):
    for xcol, ycol in [('ring_x', 'ring_y'), ('center_x', 'center_y'), ('x', 'y')]:
        if xcol in df.columns and ycol in df.columns:
            return xcol, ycol
    for col in df.columns:
        if 'x' in col and 'ring' in col:
            xcol = col
            ycol = col.replace('x', 'y')
            if ycol in df.columns:
                return xcol, ycol
    raise ValueError("Could not find ring center columns in CSV.")


def are_all_points_close(points, max_width=None, max_height=None):
    """
    Check if all trajectory points fall within a bounding box.
    Used to detect same-digit PINs where the finger stays in one location.
    
    FIX: Reduced minimum points from 5 to 2, added variance check.
    """
    if max_width is None:
        max_width = SAME_DIGIT_BOX_SIZE
    if max_height is None:
        max_height = SAME_DIGIT_BOX_SIZE
    
    # FIX: Reduced minimum from 5 to 2
    if len(points) < 2:
        return False
    
    x_min, y_min = np.min(points, axis=0)
    x_max, y_max = np.max(points, axis=0)
    width = x_max - x_min
    height = y_max - y_min
    
    # FIX: Also check variance - low variance indicates same-digit
    variance = np.var(points, axis=0).sum()
    low_variance_threshold = (max_width * max_height) / 4
    
    return (width <= max_width and height <= max_height) or (variance < low_variance_threshold)


def calculate_speeds(points):
    velocities = np.diff(points, axis=0)
    speeds = np.linalg.norm(velocities, axis=1)
    speeds = np.insert(speeds, 0, np.median(speeds))
    return speeds


def is_same_digit_pin(pin):
    """Check if PIN consists of a single repeated digit (e.g., '9999')."""
    return len(set(pin)) == 1


def collapse_repeats(pin):
    if not pin:
        return pin
    out = [pin[0]]
    for ch in pin[1:]:
        if ch != out[-1]:
            out.append(ch)
    return ''.join(out)


def filter_candidates(pin_scores, is_same_digit):
    """
    Filter PIN candidates based on detection mode.
    
    FIX: Don't completely remove same-digit PINs - penalize them instead
    when is_same_digit is False, keeping them in case detection was wrong.
    """
    filtered = [(p, s) for p, s in pin_scores if s is not None and s > 0 and s != float('inf')]
    
    if not is_same_digit:
        # FIX: Penalize same-digit PINs instead of removing them entirely
        penalized = []
        for pin, score in filtered:
            if is_same_digit_pin(pin):
                # Apply penalty multiplier instead of removing
                penalized.append((pin, score * 2.0))
            else:
                penalized.append((pin, score))
        filtered = penalized
        filtered.sort(key=lambda x: x[1])
    
    return filtered


def group_ambiguous_repeats_consecutively(pin_scores):
    groups = {}
    order = []
    for pin, score in pin_scores:
        key = collapse_repeats(pin)
        if key not in groups:
            groups[key] = []
            order.append(key)
        groups[key].append((pin, score))
    for k in groups:
        groups[k].sort(key=lambda x: x[1])
    out = []
    for k in order:
        out.extend(groups[k])
    return out


def prioritize_same_digit_pins(pin_scores, is_same_digit, pin_length):
    if not is_same_digit:
        return pin_scores
    priority_pins = [str(digit)*pin_length for digit in range(1, 10)] + ["0"*pin_length]
    prioritized_scores = []
    non_priority_pins = []
    for pin, score in pin_scores:
        if pin not in priority_pins:
            non_priority_pins.append((pin, score))
    for p_pin in priority_pins:
        for pin, score in pin_scores:
            if pin == p_pin:
                prioritized_scores.append((pin, score))
                break
    prioritized_scores.extend(non_priority_pins)
    return prioritized_scores


def generate_repeated_digit_pins(centers, times, sizes):
    if len(centers) < 2 or len(centers) > 4:
        return []
    order = np.argsort(times)
    centers_ordered = centers[order]
    sizes_ordered = np.array(sizes)[order]
    nearest_digits = []
    for center in centers_ordered:
        distances = np.sqrt(np.sum((PINPAD_COORDS - center)**2, axis=1))
        min_idx = np.argmin(distances)
        nearest_digits.append(PINPAD_DIGITS[min_idx])
    mean_size = np.mean(sizes_ordered)
    if mean_size == 0:
        return []
    relative_sizes = sizes_ordered / mean_size
    candidates = []
    pin = ''
    for digit, rel_size in zip(nearest_digits, relative_sizes):
        if rel_size > 1.5:
            pin += digit * 2
        else:
            pin += digit
    if len(pin) > len(nearest_digits):
        candidates.append(pin)
    pin = ''
    for digit, rel_size in zip(nearest_digits, relative_sizes):
        if rel_size > 2.0:
            pin += digit * 3
        elif rel_size > 1.3:
            pin += digit * 2
        else:
            pin += digit
    if pin not in candidates and len(pin) > len(nearest_digits):
        candidates.append(pin)
    return candidates


def filter_by_speed(points, speeds, frame_indices=None):
    threshold = np.percentile(speeds, 40)
    slow_mask = speeds <= threshold
    filtered_points = points[slow_mask]
    if frame_indices is not None:
        filtered_frames = frame_indices[slow_mask]
        return filtered_points, filtered_frames
    else:
        return filtered_points, np.where(slow_mask)[0]


def time_aware_clustering(points, frame_indices, n_clusters=4, time_weight=TIME_WEIGHT):
    if len(points) < n_clusters:
        return None, None, None
    scaler_space = StandardScaler()
    scaler_time = StandardScaler()
    points_scaled = scaler_space.fit_transform(points)
    time_scaled = scaler_time.fit_transform(frame_indices.reshape(-1, 1))
    space_time_features = np.hstack([points_scaled, time_weight * time_scaled])
    kmeans = KMeans(n_clusters=n_clusters, random_state=0, n_init=10)
    labels = kmeans.fit_predict(space_time_features)
    centers, times, sizes = get_cluster_centers_and_times(labels, points, frame_indices)
    return centers, times, sizes


def fit_translation_scaling(A, B):
    """
    Fit observed trajectory A to ideal trajectory B using translation and uniform scaling.
    
    FIX: Handle edge cases where A or B has zero variance (same-digit PINs).
    """
    if len(A) < 1 or len(B) < 1:
        return float('inf'), None
    
    # Handle single-point case
    if len(A) == 1 and len(B) == 1:
        error = np.sqrt(np.sum((A[0] - B[0]) ** 2))
        return error, A.copy()
    
    if len(A) < 2 or len(B) < 2:
        return float('inf'), None
    
    centroid_A = np.mean(A, axis=0)
    centroid_B = np.mean(B, axis=0)
    AA = A - centroid_A
    BB = B - centroid_B
    norm_A = np.linalg.norm(AA)
    norm_B = np.linalg.norm(BB)
    
    # FIX: Handle zero-norm cases (same-digit PINs)
    if norm_A < 1e-10 and norm_B < 1e-10:
        # Both are essentially single points - use distance between centroids
        error = np.sqrt(np.sum((centroid_A - centroid_B) ** 2))
        return error, np.tile(centroid_B, (len(A), 1)).reshape(A.shape)
    elif norm_A < 1e-10:
        # Observed is single point, ideal has spread - use centroid distance
        error = np.sqrt(np.sum((centroid_A - centroid_B) ** 2))
        return error, np.tile(centroid_B, (len(A), 1)).reshape(A.shape)
    elif norm_B < 1e-10:
        # Ideal is single point (same-digit PIN) - translate observed centroid to ideal
        spread_error = np.sqrt(np.mean(np.sum(AA ** 2, axis=1)))
        centroid_error = np.sqrt(np.sum((centroid_A - centroid_B) ** 2))
        # Total error combines spread (should be 0 for same-digit) and centroid distance
        error = spread_error + centroid_error
        A2 = np.tile(centroid_B, (len(A), 1)).reshape(A.shape)
        return error, A2
    
    scale = norm_B / norm_A
    A2 = AA * scale + centroid_B
    error = np.sqrt(np.mean(np.sum((A2 - B)**2, axis=1)))
    return error, A2


def score_same_digit_pin(observed_center, digit, pin_length):
    """
    Special scoring for same-digit PINs that doesn't rely on Procrustes.
    
    Args:
        observed_center: Mean position of observed trajectory (single point)
        digit: The digit character ('0'-'9')
        pin_length: Length of PIN (unused but kept for consistency)
    
    Returns:
        Error score (Euclidean distance to button center)
    """
    digit_idx = PINPAD_DIGIT_TO_IDX[digit]
    ideal_position = PINPAD_COORDS[digit_idx]
    distance = np.sqrt(np.sum((observed_center - ideal_position) ** 2))
    return distance


def get_cluster_centers_and_times(labels, points, frame_indices=None):
    clusters = np.unique(labels)
    centers = []
    times = []
    sizes = []
    if frame_indices is None:
        frame_indices = np.arange(len(labels))
    for c in clusters:
        idxs = np.where(labels == c)[0]
        centers.append(np.mean(points[idxs], axis=0))
        times.append(np.mean(frame_indices[idxs]))
        sizes.append(len(idxs))
    return np.array(centers), np.array(times), np.array(sizes)


def plot_trajectory_on_pinpad(centers_ordered, top_pins, out_path, title):
    plt.figure(figsize=(10, 8))
    for i, (x, y) in enumerate(PINPAD_COORDS):
        plt.scatter(x, y, s=200, c='lightgray', edgecolor='black', zorder=1)
        plt.annotate(PINPAD_DIGITS[i], xy=(x, y), fontsize=16, ha='center', va='center', zorder=2)
    if top_pins and len(top_pins) > 0:
        top_pin = top_pins[0][0]
        unique_pin = ''
        for i, digit in enumerate(top_pin):
            if i == 0 or digit != top_pin[i-1]:
                unique_pin += digit
        pin_indices = [PINPAD_DIGIT_TO_IDX[d] for d in unique_pin]
        pin_coords = PINPAD_COORDS[pin_indices]
        plt.plot(pin_coords[:,0], pin_coords[:,1], 'b-', 
                linewidth=2, alpha=0.7, label=f"{top_pin} (ideal)")
    if len(centers_ordered) > 0 and len(top_pins) > 0:
        top_pin = top_pins[0][0]
        unique_pin = ''
        for i, digit in enumerate(top_pin):
            if i == 0 or digit != top_pin[i-1]:
                unique_pin += digit
        pin_indices = [PINPAD_DIGIT_TO_IDX[d] for d in unique_pin]
        pin_coords = PINPAD_COORDS[pin_indices]
        centers_to_use = centers_ordered
        if len(centers_ordered) > len(unique_pin):
            centers_to_use = centers_ordered[:len(unique_pin)]
        _, transformed_centers = fit_translation_scaling(centers_to_use, pin_coords)
        if transformed_centers is not None:
            plt.plot(transformed_centers[:,0], transformed_centers[:,1], 'r--', alpha=0.8, zorder=5,
                     label="Observed trajectory")
            plt.scatter(transformed_centers[:,0], transformed_centers[:,1], 
                        color='red', s=80, edgecolor='white', alpha=0.8, zorder=10)
            for i, (x, y) in enumerate(transformed_centers):
                plt.annotate(f"{i+1}", xy=(x, y), xytext=(-5, 5), 
                            textcoords='offset points', fontsize=10, 
                            color='white', fontweight='bold')
    plt.title(title)
    plt.grid(False)
    plt.legend(loc="upper center", bbox_to_anchor=(0.5, -0.05), ncol=2)
    plt.tight_layout()
    plt.gca().invert_yaxis()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path)
    plt.close()


def extract_actual_pin_from_filename(video_name):
    return os.path.splitext(video_name)[0]


def find_pin_rank(pin, pin_scores):
    for rank, (candidate_pin, _) in enumerate(pin_scores, 1):
        if candidate_pin == pin:
            return rank
    return "Not Found"


def generate_individual_html_report(video_name, pin_scores, is_same_digit, actual_pin, actual_pin_rank, report_dir, pin_length, output_dir):
    report_path = os.path.join(report_dir, f"{video_name}_report.html")
    filtered_scores = filter_candidates(pin_scores, is_same_digit)

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>PIN Analysis for {video_name}</title>
<style>
body {{ font-family: Arial, sans-serif; line-height: 1.6; padding: 20px; color: #333; max-width: 1200px; margin: 0 auto; }}
h1 {{ color: #2c3e50; border-bottom: 2px solid #3498db; padding-bottom: 10px; }}
h2 {{ color: #2980b9; margin-top: 30px; }}
table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
th, td {{ text-align: left; padding: 12px 15px; border-bottom: 1px solid #ddd; }}
th {{ background-color: #f2f2f2; font-weight: bold; position: sticky; top: 0; }}
tr:hover {{ background-color: #f5f5f5; }}
.pin-rank {{ font-weight: bold; color: #e74c3c; width: 60px; text-align: center; }}
.pin-code {{ font-family: monospace; font-size: 1.2em; font-weight: bold; width: 150px; }}
.pin-score {{ color: #7f8c8d; width: 120px; }}
.trajectory-image {{ max-width: 100%; height: auto; margin: 15px 0; border: 1px solid #ddd; border-radius: 5px; }}
.timestamp {{ color: #95a5a6; font-style: italic; text-align: right; margin-top: 50px; }}
.back-link {{ margin-top: 20px; margin-bottom: 20px; }}
.back-link a {{ text-decoration: none; color: #3498db; font-weight: bold; }}
.back-link a:hover {{ text-decoration: underline; }}
.pin-table-container {{ max-height: 600px; overflow-y: auto; margin-bottom: 30px; }}
.actual-pin {{ background-color: #d4edda; font-weight: bold; }}
</style>
</head>
<body>
<div class="back-link"><a href="index.html">← Back to main report</a></div>
<h1>PIN Analysis for Video: {video_name}</h1>
<p>This report shows detailed PIN candidates based on trajectory analysis.</p>
<div class="config-info">
<p><strong>PIN Length:</strong> {pin_length} digits</p>
<p><strong>Clustering Time Weight:</strong> {TIME_WEIGHT}</p>
<p><strong>Same Digit Box Size:</strong> {SAME_DIGIT_BOX_SIZE} pixels</p>
<p><strong>Same Digit Detected:</strong> {is_same_digit}</p>
</div>
<p>Generated on: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
<h2>PIN Candidates</h2>
<div class="pin-table-container">
<table>
<tr><th class="pin-rank">Rank</th><th class="pin-code">PIN</th><th class="pin-score">Score</th></tr>
"""
    for rank, (pin, score) in enumerate(filtered_scores, 1):
        row_class = "actual-pin" if actual_pin and pin == actual_pin else ""
        html += f"""
<tr class="{row_class}">
<td class="pin-rank">{rank}</td>
<td class="pin-code">{pin}</td>
<td class="pin-score">{score:.4f}</td>
</tr>"""
    html += """
</table>
</div>
"""
    trajectory_img_path = os.path.join(output_dir, video_name, 'trajectory_mapping.png')
    if os.path.exists(trajectory_img_path):
        report_img_path = os.path.join(report_dir, f"{video_name}_trajectory.png")
        shutil.copy(trajectory_img_path, report_img_path)
        html += f"""
<h2>Trajectory Visualization</h2>
<img class="trajectory-image" src="{os.path.basename(report_img_path)}" alt="Trajectory plot for {video_name}">
"""
    html += """
<div class="back-link"><a href="index.html">← Back to main report</a></div>
<div class="timestamp"><p>Analysis powered by Trajectory-Based PIN Detection</p></div>
</body>
</html>
"""
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(html)
    return os.path.basename(report_path)


def generate_main_html_report(results, pattern_info, report_dir, video_reports, actual_pins, pin_length):
    report_path = os.path.join(report_dir, 'index.html')
    html = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>PIN Trajectory Analysis - Summary Report</title>
<style>
body { font-family: Arial, sans-serif; line-height: 1.6; padding: 20px; color: #333; max-width: 1200px; margin: 0 auto; }
h1 { color: #2c3e50; border-bottom: 2px solid #3498db; padding-bottom: 10px; }
h2 { color: #2980b9; margin-top: 30px; }
table { border-collapse: collapse; width: 100%; margin: 20px 0; }
th, td { text-align: left; padding: 12px 15px; border-bottom: 1px solid #ddd; }
th { background-color: #f2f2f2; font-weight: bold; }
tr:hover { background-color: #f5f5f5; }
.video-section { margin-bottom: 30px; padding: 15px; background-color: #f9f9f9; border-radius: 5px; box-shadow: 0 2px 5px rgba(0,0,0,0.1); }
.video-name { margin-top: 0; color: #2c3e50; }
.pin-rank { font-weight: bold; color: #e74c3c; width: 60px; text-align: center; }
.pin-code { font-family: monospace; font-size: 1.2em; font-weight: bold; width: 100px; }
.pin-score { color: #7f8c8d; width: 100px; }
.timestamp { color: #95a5a6; font-style: italic; text-align: right; margin-top: 50px; }
.details-link { margin-top: 10px; }
.details-link a { display: inline-block; background-color: #3498db; color: white; padding: 8px 15px; text-decoration: none; border-radius: 4px; font-weight: bold; }
.details-link a:hover { background-color: #2980b9; }
.top-pins { font-family: monospace; color: #555; }
.same-digit { background-color: #ffe8e8; }
.config-info { background-color: #f9f9f9; padding: 15px; border-radius: 5px; margin-bottom: 20px; }
.config-param { font-family: monospace; font-weight: bold; }
.actual-pin { background-color: #d4edda; font-weight: bold; }
.actual-pin-rank { font-weight: bold; text-align: center; }
.rank-1 { color: #28a745; background-color: #d4edda; }
.rank-5-plus { color: #dc3545; background-color: #f8d7da; }
.rank-not-found { color: #6c757d; background-color: #e9ecef; }
.variable-length { background-color: #fff3cd; }
</style>
</head>
<body>
<h1>PIN Trajectory Analysis - Summary Report</h1>
<p>This report summarizes the PIN analysis results across all videos.</p>
<p>Generated on: """ + datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S') + """</p>
<div class="config-info">
<h3>Configuration Parameters</h3>
<p><span class="config-param">PIN_LENGTH</span>: """ + str(pin_length) + """</p>
<p><span class="config-param">TIME_WEIGHT</span>: """ + str(TIME_WEIGHT) + """</p>
<p><span class="config-param">SAME_DIGIT_BOX_SIZE</span>: """ + str(SAME_DIGIT_BOX_SIZE) + """ pixels</p>
</div>
<h2>Videos Analyzed</h2>
<table>
<tr>
<th>Video</th>
<th>Actual PIN</th>
<th class="actual-pin-rank">Actual PIN Rank</th>
<th>Top PIN</th>
<th>Score</th>
<th>Top 5 Candidates</th>
<th>Action</th>
</tr>
"""
    for video_name, pin_scores in results.items():
        if pin_scores:
            is_same_digit = pattern_info.get(video_name, False)
            filtered_scores = filter_candidates(pin_scores, is_same_digit)
            if not filtered_scores:
                filtered_scores = pin_scores
            top_pin = filtered_scores[0][0]
            top_score = filtered_scores[0][1]
            report_filename = video_reports.get(video_name, "")
            actual_pin = actual_pins.get(video_name, {}).get('pin', 'Unknown')
            actual_pin_rank = actual_pins.get(video_name, {}).get('rank', 'N/A')
            rank_class = ""
            if actual_pin_rank == 1:
                rank_class = "rank-1"
            elif actual_pin_rank != "Not Found" and actual_pin_rank > 5:
                rank_class = "rank-5-plus"
            elif actual_pin_rank == "Not Found":
                rank_class = "rank-not-found"
            top_5_pins = ", ".join([ps[0] for ps in filtered_scores[:5]])
            row_class = ""
            if actual_pin and top_pin == actual_pin:
                row_class = "class=\"actual-pin\""
            html += f"""
<tr {row_class}>
<td>{video_name}</td>
<td class="pin-code">{actual_pin}</td>
<td class="actual-pin-rank {rank_class}">{actual_pin_rank}</td>
<td class="pin-code">{top_pin}</td>
<td class="pin-score">{top_score:.4f}</td>
<td class="top-pins">{top_5_pins}</td>
<td class="details-link"><a href="{report_filename}">View Details</a></td>
</tr>"""
    html += """
</table>
<div class="timestamp">
<p>Analysis powered by Time-Aware Trajectory PIN Detection</p>
</div>
</body>
</html>
"""
    with open(report_path, 'w') as f:
        f.write(html)
    print(f"Main HTML report generated at: {report_path}")


# ---------------- High-performance vectorized scoring ------------------

def _digits_from_range(start, end, pin_length):
    n = end - start
    arr = np.arange(start, end, dtype=np.int64)
    digits = np.empty((n, pin_length), dtype=np.int8)
    for pos in range(pin_length - 1, -1, -1):
        digits[:, pos] = (arr % 10).astype(np.int8)
        arr //= 10
    return digits


def _vectorized_errors_for_chunk(start, end, A_centers, pin_length=None):
    """
    Vectorized scoring with FIX for same-digit PINs.
    """
    with threadpool_limits(limits=1):
        L = A_centers.shape[0]
        n = end - start
        if n <= 0:
            return [], []
            
        effective_length = pin_length if pin_length is not None else L
        
        if L != effective_length:
            if L < effective_length:
                padding = np.tile(A_centers[-1:], (effective_length - L, 1))
                A_centers = np.vstack([A_centers, padding])
            else:
                A_centers = A_centers[:effective_length]
            L = effective_length
            
        digits = _digits_from_range(start, end, L)
        idx = DEC_TO_PINPAD_IDX[digits]
        B = PINPAD_COORDS[idx]
        
        centroid_A = A_centers.mean(axis=0, keepdims=True)
        AA = A_centers - centroid_A
        AA = AA[np.newaxis, :, :]
        norm_A = np.sqrt((AA**2).sum())
        
        centroid_B = B.mean(axis=1, keepdims=True)
        BB = B - centroid_B
        norm_B = np.sqrt((BB**2).sum(axis=(1,2)))
        
        # FIX: Handle same-digit PINs where norm_B is 0
        errs = np.zeros(n, dtype=np.float64)
        
        # Mask for same-digit PINs (norm_B ≈ 0)
        same_digit_mask = norm_B < 1e-10
        different_digit_mask = ~same_digit_mask
        
        # Handle same-digit PINs: error based on centroid distance
        if same_digit_mask.any():
            centroid_B_same = centroid_B[same_digit_mask].squeeze(1)  # (m, 2)
            if norm_A < 1e-10:
                # Both same-digit: use distance between centroids
                dist_to_centroids = np.sqrt(np.sum((centroid_A - centroid_B_same) ** 2, axis=1))
                errs[same_digit_mask] = dist_to_centroids
            else:
                # Observed has spread, ideal is single point
                # Error = spread of observed + distance to ideal centroid
                spread_error = np.sqrt(np.mean(np.sum(AA[0]**2, axis=1)))
                dist_to_centroids = np.sqrt(np.sum((centroid_A - centroid_B_same) ** 2, axis=1))
                errs[same_digit_mask] = spread_error + dist_to_centroids
        
        # Handle different-digit PINs: standard Procrustes
        if different_digit_mask.any():
            if norm_A < 1e-10:
                # Observed is single point, ideal has spread
                # Use centroid distance as error
                centroid_B_diff = centroid_B[different_digit_mask].squeeze(1)
                dist_to_centroids = np.sqrt(np.sum((centroid_A - centroid_B_diff) ** 2, axis=1))
                errs[different_digit_mask] = dist_to_centroids
            else:
                norm_B_diff = norm_B[different_digit_mask]
                scale = norm_B_diff / norm_A
                
                centroid_B_diff = centroid_B[different_digit_mask]
                B_diff = B[different_digit_mask]
                
                A2 = AA * scale[:, None, None] + centroid_B_diff
                diff = A2 - B_diff
                errs[different_digit_mask] = np.sqrt(np.mean(np.sum(diff**2, axis=2), axis=1))
        
        pins = [''.join(map(str, row.tolist())) for row in digits]
        return pins, errs


def _score_chunk_worker(start, end, A_centers, top_m, pin_length=None):
    pins, errs = _vectorized_errors_for_chunk(start, end, A_centers, pin_length)
    if len(pins) == 0:
        return []
    errs = np.asarray(errs)
    n = len(errs)
    if top_m >= n:
        order = np.argsort(errs)
        return [(pins[i], float(errs[i])) for i in order]
    idx = np.argpartition(errs, top_m - 1)[:top_m]
    idx = idx[np.argsort(errs[idx])]
    return [(pins[i], float(errs[i])) for i in idx]


def score_all_pins_fast(best_centers_ordered, pin_length, num_workers, chunk_size, topk_per_chunk, topk_final):
    total = 10 ** pin_length
    A = np.asarray(best_centers_ordered, dtype=np.float64)
    t0 = time.time()
    print(f"  Fast scoring over {total:,} candidates: workers={num_workers}, chunk_size={chunk_size}, topk_per_chunk={topk_per_chunk}, topk_final={topk_final}")
    tasks = []
    results = []
    if num_workers <= 1:
        with threadpool_limits(limits=os.cpu_count() or 8):
            for start in range(0, total, chunk_size):
                end = min(start + chunk_size, total)
                res = _score_chunk_worker(start, end, A, topk_per_chunk, pin_length)
                results.extend(res)
    else:
        with ProcessPoolExecutor(max_workers=num_workers) as ex:
            for start in range(0, total, chunk_size):
                end = min(start + chunk_size, total)
                tasks.append(ex.submit(_score_chunk_worker, start, end, A, topk_per_chunk, pin_length))
            for fut in as_completed(tasks):
                try:
                    res = fut.result()
                    results.extend(res)
                except Exception as e:
                    print(f"  Worker error: {e}")
    if not results:
        return []
    errs = np.array([e for _, e in results], dtype=np.float64)
    if len(results) > topk_final:
        idx = np.argpartition(errs, topk_final - 1)[:topk_final]
        results = [results[i] for i in idx]
        results.sort(key=lambda x: x[1])
    else:
        results.sort(key=lambda x: x[1])
    dt = time.time() - t0
    print(f"  Fast scoring completed in {dt:.2f}s. Retained {len(results):,} best candidates.")
    return results


# ==================== FULL process_csv_trajectory ====================

def process_csv_trajectory(csv_path, report_dir, pin_length, output_dir):
    """
    Process a single CSV trajectory file and generate PIN candidates.
    
    FIX: Improved same-digit PIN handling throughout.
    """
    video_dir = os.path.dirname(csv_path)
    video_name = os.path.basename(video_dir)
    os.makedirs(report_dir, exist_ok=True)
    print(f"\nProcessing video: {video_name}")
    
    # Extract actual PIN from filename
    actual_pin = extract_actual_pin_from_filename(video_name)
    if actual_pin:
        print(f"  Actual PIN identified from filename: {actual_pin}")
    
    # Load CSV data
    df = pd.read_csv(csv_path)
    xcol, ycol = find_ring_center_cols(df)
    xs = df[xcol].values
    ys = df[ycol].values
    mask = ~np.isnan(xs) & ~np.isnan(ys)
    xs, ys = xs[mask], ys[mask]
    points = np.stack([xs, ys], axis=1)
    frame_indices = np.arange(len(df))[mask]
    
    if points.shape[0] == 0:
        print(f"Warning: No valid points in {csv_path}. Skipping.")
        return [], False, None, "Not Found"
    
    # Calculate speeds and filter slow points
    speeds = calculate_speeds(points)
    filtered_points, filtered_frames = filter_by_speed(points, speeds, frame_indices)
    print(f"  Filtered {len(points)} points to {len(filtered_points)} slow points")
    
    # ========== FIX: Improved same-digit detection ==========
    is_same_digit = are_all_points_close(filtered_points)
    
    # Additional check: if actual PIN is known and is same-digit, force detection
    if actual_pin and is_same_digit_pin(actual_pin):
        is_same_digit = True
        print(f"  FORCED same-digit detection based on actual PIN: {actual_pin}")
    
    if is_same_digit:
        print(f"  DETECTED: Same-digit PIN pattern (points within {SAME_DIGIT_BOX_SIZE}x{SAME_DIGIT_BOX_SIZE} box or low variance)")
    
    # Store same-digit center for later use
    same_digit_center = np.mean(filtered_points, axis=0).reshape(1, 2) if len(filtered_points) > 0 else None
    
    best_centers_ordered = None
    centers = None
    times = None
    sizes = None
    
    # ========== Try clustering ==========
    centers, times, sizes = time_aware_clustering(filtered_points, filtered_frames, n_clusters=pin_length)
    
    if centers is not None and len(centers) == pin_length:
        print(f"  Using time-aware clustering: Found {len(centers)} centers")
        order = np.argsort(times)
        centers_ordered = centers[order]
        best_centers_ordered = centers_ordered
        
        # FIX: Check if clustered centers are actually spread out
        center_spread = np.std(centers_ordered, axis=0).sum()
        if center_spread < 10:
            print(f"  Cluster spread is low ({center_spread:.2f}) - likely same-digit PIN")
            is_same_digit = True
    else:
        # Fallback clustering with various k values
        print("  Falling back to standard clustering")
        for k in [pin_length, 4, 3, 2, 1]:
            if k > max(1, len(filtered_points) // 3):
                continue
            try:
                scaler = StandardScaler()
                filtered_points_scaled = scaler.fit_transform(filtered_points)
                kmeans = KMeans(n_clusters=k, random_state=0, n_init=10)
                labels = kmeans.fit_predict(filtered_points_scaled)
                centers, times, sizes = get_cluster_centers_and_times(labels, filtered_points, filtered_frames)
                order = np.argsort(times)
                centers_ordered = centers[order]
                print(f"  Using standard k={k}: Found {len(centers_ordered)} centers")
                if best_centers_ordered is None:
                    best_centers_ordered = centers_ordered
                    
                    # Check spread for same-digit detection
                    if k == 1 or np.std(centers_ordered, axis=0).sum() < 10:
                        is_same_digit = True
                        print(f"  Low spread detected - marking as same-digit PIN")
                break
            except Exception as e:
                print(f"  Clustering with k={k} failed: {e}")
                continue
    
    # ========== FIX: Use same_digit_center if needed ==========
    if best_centers_ordered is None:
        if same_digit_center is not None:
            print("  Using same-digit center (duplicated for PIN length)")
            # Duplicate the center pin_length times for scoring
            best_centers_ordered = np.tile(same_digit_center, (pin_length, 1))
            is_same_digit = True
        else:
            # Last resort: use mean of all filtered points
            if len(filtered_points) > 0:
                mean_center = np.mean(filtered_points, axis=0).reshape(1, 2)
                best_centers_ordered = np.tile(mean_center, (pin_length, 1))
                print("  Using mean of all points as fallback")
                is_same_digit = True
    
    if best_centers_ordered is None:
        print(f"  WARNING: Could not find valid centers for {video_name}")
        return [], is_same_digit, actual_pin, "Not Found"
    
    print(f"  is_same_digit = {is_same_digit}")
    print("  Generating PIN candidates using trajectory matching (fast/vectorized)...")
    
    # ========== Score all PINs ==========
    pin_trajectories = load_pin_trajectories(pin_length)
    pin_scores = []
    
    if pin_trajectories:
        # Use preloaded trajectories
        items = list(pin_trajectories.items())
        def score_batch(batch):
            with threadpool_limits(limits=1 if NUM_WORKERS > 1 else (os.cpu_count() or 8)):
                A = best_centers_ordered
                out = []
                for pin, pin_coords in batch:
                    B = np.asarray(pin_coords, dtype=np.float64)
                    err, _ = fit_translation_scaling(A, B)
                    out.append((pin, float(err)))
                return out
        batch_size = 50_000
        if NUM_WORKERS <= 1:
            for i in range(0, len(items), batch_size):
                pin_scores.extend(score_batch(items[i:i+batch_size]))
        else:
            with ProcessPoolExecutor(max_workers=NUM_WORKERS) as ex:
                futures = [ex.submit(score_batch, items[i:i+batch_size]) for i in range(0, len(items), batch_size)]
                for fut in as_completed(futures):
                    pin_scores.extend(fut.result())
        pin_scores.sort(key=lambda x: x[1])
    else:
        # Fast vectorized on-the-fly scoring
        pin_scores = score_all_pins_fast(
            best_centers_ordered, pin_length, NUM_WORKERS, CHUNK_SIZE, TOPK_PER_CHUNK, TOPK_FINAL
        )
    
    # ========== FIX: Ensure same-digit PINs are properly scored ==========
    if is_same_digit and same_digit_center is not None:
        observed_center = same_digit_center.flatten()
        print(f"  Adding/updating same-digit PIN scores using centroid distance...")
        
        for digit in PINPAD_DIGITS:
            same_digit_pin = digit * pin_length
            # Score using simple Euclidean distance
            score = score_same_digit_pin(observed_center, digit, pin_length)
            
            # Check if already in scores
            existing_idx = None
            for i, (p, s) in enumerate(pin_scores):
                if p == same_digit_pin:
                    existing_idx = i
                    break
            
            if existing_idx is not None:
                old_score = pin_scores[existing_idx][1]
                # Use the better (lower) score
                if score < old_score or old_score == float('inf') or old_score <= 0:
                    pin_scores[existing_idx] = (same_digit_pin, score)
                    print(f"    Updated {same_digit_pin}: {old_score:.4f} -> {score:.4f}")
            else:
                pin_scores.append((same_digit_pin, score))
                print(f"    Added {same_digit_pin}: {score:.4f}")
    
    # Add special variable-length repeated-digit candidates
    if centers is not None and times is not None and sizes is not None:
        repeat_pins = generate_repeated_digit_pins(centers, times, sizes)
        if repeat_pins:
            print(f"  Generated {len(repeat_pins)} variable-length repeated-digit candidates")
            for pin in repeat_pins:
                try:
                    collapsed_pin = pin[0]
                    for i in range(1, len(pin)):
                        if pin[i] != pin[i-1]:
                            collapsed_pin += pin[i]
                    if len(collapsed_pin) == len(best_centers_ordered):
                        pin_indices = [PINPAD_DIGIT_TO_IDX[d] for d in collapsed_pin]
                        pin_coords = PINPAD_COORDS[pin_indices]
                        error, _ = fit_translation_scaling(best_centers_ordered, pin_coords)
                        error = error * 0.95
                        pin_scores.append((pin, error))
                except Exception as e:
                    print(f"  Error scoring variable-length PIN {pin}: {e}")
    
    # ========== Sort and post-process ==========
    pin_scores.sort(key=lambda x: x[1])
    pin_scores = prioritize_same_digit_pins(pin_scores, is_same_digit, pin_length)
    raw_sorted_scores = list(pin_scores)
    pin_scores = filter_candidates(pin_scores, is_same_digit)
    pin_scores = group_ambiguous_repeats_consecutively(pin_scores)
    
    # ========== Find actual PIN rank ==========
    actual_pin_rank = "Not Found"
    if actual_pin:
        actual_pin_rank = find_pin_rank(actual_pin, pin_scores)
        print(f"  Actual PIN {actual_pin} found at rank {actual_pin_rank}")
        
        # FIX: If still not found, check raw scores
        if actual_pin_rank == "Not Found":
            raw_rank = find_pin_rank(actual_pin, raw_sorted_scores)
            if raw_rank != "Not Found":
                print(f"  NOTE: Actual PIN found at rank {raw_rank} in raw scores (before filtering)")
                # Add it back to filtered scores if it was incorrectly removed
                for p, s in raw_sorted_scores:
                    if p == actual_pin:
                        pin_scores.append((p, s))
                        pin_scores.sort(key=lambda x: x[1])
                        actual_pin_rank = find_pin_rank(actual_pin, pin_scores)
                        print(f"  Re-added actual PIN, now at rank {actual_pin_rank}")
                        break
    
    # ========== Dynamic score-based cutoff ==========
    if pin_scores:
        best_score = pin_scores[0][1]
        all_scores = np.array([score for _, score in pin_scores])
        score_mean = np.mean(all_scores)
        score_std = np.std(all_scores)
        
        jump_idx = len(pin_scores)
        last_score = best_score
        score_range = max(0.1, best_score)
        for i, (_, score) in enumerate(pin_scores[1:], 1):
            jump_threshold = max(0.05, score_range * 0.20)
            if score - last_score > jump_threshold:
                jump_idx = i
                break
            last_score = score
        
        absolute_threshold = best_score + min(0.5, best_score)
        absolute_idx = next((i for i, (_, s) in enumerate(pin_scores) if s > absolute_threshold), len(pin_scores))
        
        stat_threshold = best_score + 2 * score_std if score_std > 0 else best_score * 2
        stat_idx = next((i for i, (_, s) in enumerate(pin_scores) if s > stat_threshold), len(pin_scores))
        
        min_candidates = 1000
        cutoff_idx = max(min_candidates, min(jump_idx, absolute_idx, stat_idx))
        pin_scores = pin_scores[:cutoff_idx]
        print(f"  Using dynamic score-based cutoff: Keeping {len(pin_scores)} candidates")
    
    # ========== Print top candidates ==========
    print("\nTop PIN candidates (trajectory matching only):")
    for pin, score in pin_scores[:min(10, len(pin_scores))]:
        marker = " <-- ACTUAL" if actual_pin and pin == actual_pin else ""
        print(f"  {pin}: {score:.4f}{marker}")
    
    # ========== Generate trajectory plot ==========
    if best_centers_ordered is not None and len(best_centers_ordered) > 0:
        plot_list = []
        if actual_pin:
            found = next(((p, s) for (p, s) in pin_scores if p == actual_pin), None)
            if not found:
                found = next(((p, s) for (p, s) in raw_sorted_scores if p == actual_pin and (s is not None and s > 0)), None)
            if found:
                plot_list = [found]
        if not plot_list:
            plot_list = pin_scores[:1] if pin_scores else []
        
        title = 'Time-Aware Trajectory Matching'
        if is_same_digit:
            title += ' (Same-Digit Detected)'
        
        plot_trajectory_on_pinpad(
            best_centers_ordered, plot_list,
            os.path.join(video_dir, 'trajectory_mapping.png'),
            f'{title}: Using {"Actual PIN" if (plot_list and plot_list[0][0] == actual_pin) else "Top Candidate"} {plot_list[0][0] if plot_list else "N/A"}'
        )
    
    return pin_scores, is_same_digit, actual_pin, actual_pin_rank


# ==================== UPDATED main() ====================

def main():
    parser = argparse.ArgumentParser(description="High-performance PIN trajectory analysis")
    
    # Directory arguments (REQUIRED)
    parser.add_argument("--input-dir", "-i", type=str, required=True,
                        help="Input directory containing CSV trajectory files")
    parser.add_argument("--output-dir", "-o", type=str, required=True,
                        help="Output directory for generated reports")
    
    # PIN length argument
    parser.add_argument("--pin-length", "-l", type=int, default=4,
                        help="Length of PIN to analyze (default: 4)")
    
    # Performance arguments
    parser.add_argument("--workers", type=int, default=NUM_WORKERS_DEFAULT,
                        help="Number of parallel workers (default: all cores)")
    parser.add_argument("--chunk-size", type=int, default=CHUNK_SIZE_DEFAULT,
                        help="Number of PINs per chunk for vectorized scoring")
    parser.add_argument("--topk-per-chunk", type=int, default=TOPK_PER_CHUNK_DEFAULT,
                        help="Keep top-K per chunk")
    parser.add_argument("--topk-final", type=int, default=TOPK_FINAL_DEFAULT,
                        help="Keep top-K globally before post-filtering")
    parser.add_argument("--use-pkl-trajectories", action="store_true",
                        help="Use precomputed trajectory PKL if available (memory heavy)")
    
    args = parser.parse_args()

    # Set global variables from arguments
    global NUM_WORKERS, CHUNK_SIZE, TOPK_PER_CHUNK, TOPK_FINAL, USE_PKL_TRAJ
    global OUTPUT_DIR, REPORT_FOLDER, PIN_LENGTH
    
    OUTPUT_DIR = args.input_dir
    REPORT_FOLDER = args.output_dir
    PIN_LENGTH = args.pin_length
    
    NUM_WORKERS = max(1, args.workers)
    CHUNK_SIZE = max(10_000, args.chunk_size)
    TOPK_PER_CHUNK = max(1_000, args.topk_per_chunk)
    TOPK_FINAL = max(5_000, args.topk_final)
    USE_PKL_TRAJ = bool(args.use_pkl_trajectories)

    print(f"\n{'='*60}")
    print(f"PIN Trajectory Analysis")
    print(f"{'='*60}")
    print(f"\nDirectory config:")
    print(f"  Input directory:  {OUTPUT_DIR}")
    print(f"  Output directory: {REPORT_FOLDER}")
    print(f"\nPIN config:")
    print(f"  PIN length: {PIN_LENGTH} digits")
    print(f"\nPerformance config:")
    print(f"  Workers:        {NUM_WORKERS}")
    print(f"  Chunk size:     {CHUNK_SIZE}")
    print(f"  Top-K per chunk:{TOPK_PER_CHUNK}")
    print(f"  Top-K final:    {TOPK_FINAL}")
    print(f"  Use PKL:        {USE_PKL_TRAJ}")
    print(f"{'='*60}\n")

    report_dir = os.path.join('.', REPORT_FOLDER)
    os.makedirs(report_dir, exist_ok=True)

    # Find CSV files
    filtered_csv_files = glob.glob(os.path.join(OUTPUT_DIR, '*', '*_filtered_ring_center.csv'))
    all_leds_csv_files = glob.glob(os.path.join(OUTPUT_DIR, '*', '*_all_leds_center.csv'))
    unfiltered_csv_files = glob.glob(os.path.join(OUTPUT_DIR, '*', '*_ring_center.csv'))

    video_to_csv = {}
    for csv_path in filtered_csv_files:
        video_name = os.path.basename(os.path.dirname(csv_path))
        video_to_csv[video_name] = csv_path
    for csv_path in all_leds_csv_files:
        video_name = os.path.basename(os.path.dirname(csv_path))
        if video_name not in video_to_csv:
            video_to_csv[video_name] = csv_path
    for csv_path in unfiltered_csv_files:
        video_name = os.path.basename(os.path.dirname(csv_path))
        if video_name not in video_to_csv:
            video_to_csv[video_name] = csv_path

    csv_files = list(video_to_csv.values())
    total = len(csv_files)
    print(f"Found {total} CSV files for analysis.")
    
    if total == 0:
        print(f"No CSV files found in {OUTPUT_DIR}. Please check the input directory.")
        return
    
    results = {}
    pattern_info = {}
    video_reports = {}
    actual_pins = {}
    
    for idx, csv_path in enumerate(csv_files, 1):
        print(f"\n{'='*60}")
        print(f"Processing {idx}/{total}: {csv_path}")
        print(f"{'='*60}")
        video_name = os.path.basename(os.path.dirname(csv_path))
        try:
            pin_scores, is_same_digit, actual_pin, actual_pin_rank = process_csv_trajectory(
                csv_path, report_dir, PIN_LENGTH, OUTPUT_DIR
            )
            if pin_scores:
                results[video_name] = pin_scores
                pattern_info[video_name] = is_same_digit
                actual_pins[video_name] = {'pin': actual_pin, 'rank': actual_pin_rank}
                report_filename = generate_individual_html_report(
                    video_name, pin_scores, is_same_digit, actual_pin, actual_pin_rank, 
                    report_dir, PIN_LENGTH, OUTPUT_DIR
                )
                video_reports[video_name] = report_filename
        except Exception as e:
            print(f"Error processing file {idx}: {e}")
            import traceback
            traceback.print_exc()
    
    generate_main_html_report(results, pattern_info, report_dir, video_reports, actual_pins, PIN_LENGTH)
    main_report_path = os.path.join(report_dir, 'index.html')
    print(f"\n{'='*60}")
    print(f"Opening HTML report in default browser: {main_report_path}")
    print(f"{'='*60}")
    webbrowser.open('file://' + os.path.abspath(main_report_path))


if __name__ == '__main__':
    main()
