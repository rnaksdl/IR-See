'''
python 401_guess_angle.py -i ../1_angle_pitch_p60/output_e -o ../1_angle_/pitch_p60/report_e --yaw 0 0 --pitch 30 90
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

import math
import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.preprocessing import StandardScaler
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

# --- USER CONFIGURABLE PARAMETERS ---
PIN_LENGTH = 4
OUTPUT_DIR = '../output_lux10_show'
REPORT_FOLDER = './report_best_per_pin_lux10_show'
TIME_WEIGHT = 0.5
TRAJECTORIES_DIR = './pin_trajectories'


YAW_RANGE   = (0.0, 0.0)
PITCH_RANGE = (5.0, 10.0)
STEP_DEG = 5.0


RANK_LT_THRESHOLD = 5

YAW_DEG   = float(YAW_RANGE[0])
PITCH_DEG = float(PITCH_RANGE[0])

def _cos_safe(deg: float) -> float:
    v = math.cos(math.radians(deg))
    return v if abs(v) > 1e-6 else 1e-6

print(f"Using PIN length: {PIN_LENGTH} digits")

SAME_DIGIT_BOX_SIZE = 80

BUTTON_WIDTH = 10.0
BUTTON_HEIGHT = 5.5
GAP = 0.9
X_OFFSET = BUTTON_WIDTH / 2
Y_OFFSET = BUTTON_HEIGHT / 2

PINPAD_COORDS = np.array([
    [0 * BUTTON_WIDTH + 0 * GAP + X_OFFSET, 0 * BUTTON_HEIGHT + 0 * GAP + Y_OFFSET],  # 1
    [1 * BUTTON_WIDTH + 1 * GAP + X_OFFSET, 0 * BUTTON_HEIGHT + 0 * GAP + Y_OFFSET],  # 2
    [2 * BUTTON_WIDTH + 2 * GAP + X_OFFSET, 0 * BUTTON_HEIGHT + 0 * GAP + Y_OFFSET],  # 3
    [0 * BUTTON_WIDTH + 0 * GAP + X_OFFSET, 1 * BUTTON_HEIGHT + 1 * GAP + Y_OFFSET],  # 4
    [1 * BUTTON_WIDTH + 1 * GAP + X_OFFSET, 1 * BUTTON_HEIGHT + 1 * GAP + Y_OFFSET],  # 5
    [2 * BUTTON_WIDTH + 2 * GAP + X_OFFSET, 1 * BUTTON_HEIGHT + 1 * GAP + Y_OFFSET],  # 6
    [0 * BUTTON_WIDTH + 0 * GAP + X_OFFSET, 2 * BUTTON_HEIGHT + 2 * GAP + Y_OFFSET],  # 7
    [1 * BUTTON_WIDTH + 1 * GAP + X_OFFSET, 2 * BUTTON_HEIGHT + 2 * GAP + Y_OFFSET],  # 8
    [2 * BUTTON_WIDTH + 2 * GAP + X_OFFSET, 2 * BUTTON_HEIGHT + 2 * GAP + Y_OFFSET],  # 9
    [1 * BUTTON_WIDTH + 1 * GAP + X_OFFSET, 3 * BUTTON_HEIGHT + 3 * GAP + Y_OFFSET]   # 0
])

PINPAD_DIGITS = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '0']
PINPAD_DIGIT_TO_IDX = {d: i for i, d in enumerate(PINPAD_DIGITS)}
DEC_TO_PINPAD_IDX = np.array([PINPAD_DIGIT_TO_IDX[str(d)] for d in range(10)], dtype=np.int8)

loaded_trajectories = {}

NUM_WORKERS = NUM_WORKERS_DEFAULT
CHUNK_SIZE = CHUNK_SIZE_DEFAULT
TOPK_PER_CHUNK = TOPK_PER_CHUNK_DEFAULT
TOPK_FINAL = TOPK_FINAL_DEFAULT
USE_PKL_TRAJ = USE_PKL_TRAJ_DEFAULT

def count_rank1_in_index(report_dir: str) -> int:
    idx = os.path.join(report_dir, "index.html")
    if not os.path.exists(idx):
        candidates = []
        for root, _, files in os.walk(report_dir):
            for fn in files:
                if fn.lower() == "index.html":
                    p = os.path.join(root, fn)
                    candidates.append((p, os.path.getmtime(p)))
        if not candidates:
            print(f"[WARN] No index.html found under {report_dir}")
            return 0
        idx = max(candidates, key=lambda x: x[1])[0]
    try:
        with open(idx, "r", encoding="utf-8", errors="ignore") as f:
            html = f.read()
    except Exception as e:
        print(f"[WARN] Failed to read index.html: {e}")
        return 0
    try:
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(html, "html.parser")
        total = 0
        for tbl in soup.find_all("table"):
            headers = [th.get_text(strip=True).lower() for th in tbl.find_all("th")]
            if not headers: continue
            if any("rank" in h for h in headers):
                try:
                    ridx = next(i for i, h in enumerate(headers) if "rank" in h)
                except StopIteration:
                    ridx = None
                if ridx is None: continue
                for tr in tbl.find_all("tr"):
                    tds = tr.find_all("td")
                    if len(tds) > ridx:
                        val = tds[ridx].get_text(strip=True)
                        if val.isdigit() and int(val) == 1:
                            total += 1
        total += len(soup.select('[data-rank="1"], [data-rank=\'1\']'))
        text = soup.get_text(" ", strip=True)
        total += len(re.findall(r"\brank\b\s*[:=]\s*1\b", text, flags=re.I))
        return total
    except Exception:
        return len(re.findall(r"\brank\b\s*[:=]\s*1\b", html, flags=re.I))

def count_ranklt_in_index(report_dir: str, lt_threshold: int) -> int:
    idx = os.path.join(report_dir, "index.html")
    if not os.path.exists(idx):
        candidates = []
        for root, _, files in os.walk(report_dir):
            for fn in files:
                if fn.lower() == "index.html":
                    p = os.path.join(root, fn)
                    candidates.append((p, os.path.getmtime(p)))
        if not candidates:
            print(f"[WARN] No index.html found under {report_dir}")
            return 0
        idx = max(candidates, key=lambda x: x[1])[0]
    try:
        with open(idx, "r", encoding="utf-8", errors="ignore") as f:
            html = f.read()
    except Exception as e:
        print(f"[WARN] Failed to read index.html: {e}")
        return 0
    try:
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(html, "html.parser")
        total = 0
        for tbl in soup.find_all("table"):
            headers_raw = [th.get_text(strip=True) for th in tbl.find_all("th")]
            headers = [h.lower() for h in headers_raw]
            if not headers:
                continue
            try:
                ridx = next(i for i, h in enumerate(headers) if "actual pin rank" in h)
            except StopIteration:
                continue
            for tr in tbl.find_all("tr"):
                tds = tr.find_all("td")
                if len(tds) > ridx:
                    val = tds[ridx].get_text(strip=True)
                    if val.isdigit() and int(val) < lt_threshold:
                        total += 1
        return total
    except Exception:
        nums = re.findall(r'class="actual-pin-rank[^"]*">\s*(\d+)\s*<', html, flags=re.I)
        return sum(int(s) < lt_threshold for s in nums if s.isdigit())


def load_pin_trajectories(pin_length):
    pkl_path = os.path.join(TRAJECTORIES_DIR, f'pin{pin_length}_trajectories.pkl')
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

def find_ring_center_cols(df):
    for xcol, ycol in [('ring_x','ring_y'),('center_x','center_y'),('x','y')]:
        if xcol in df.columns and ycol in df.columns: return xcol, ycol
    for col in df.columns:
        if 'x' in col and 'ring' in col:
            xcol = col; ycol = col.replace('x','y')
            if ycol in df.columns: return xcol, ycol
    raise ValueError("Could not find ring center columns in CSV.")

def are_all_points_close(points, max_width=None, max_height=None):
    if max_width is None: max_width = SAME_DIGIT_BOX_SIZE
    if max_height is None: max_height = SAME_DIGIT_BOX_SIZE
    if len(points) < 5: return False
    x_min, y_min = np.min(points, axis=0); x_max, y_max = np.max(points, axis=0)
    return (x_max - x_min) <= max_width and (y_max - y_min) <= max_height

def calculate_speeds(points):
    velocities = np.diff(points, axis=0)
    speeds = np.linalg.norm(velocities, axis=1)
    speeds = np.insert(speeds, 0, np.median(speeds))
    return speeds

def is_same_digit_pin(pin): return len(set(pin)) == 1

def collapse_repeats(pin):
    if not pin: return pin
    out = [pin[0]]
    for ch in pin[1:]:
        if ch != out[-1]: out.append(ch)
    return ''.join(out)

def filter_candidates(pin_scores, is_same_digit):
    filtered = [(p, s) for p, s in pin_scores if s is not None and s > 0]
    if not is_same_digit:
        filtered = [(p, s) for p, s in filtered if not is_same_digit_pin(p)]
    return filtered

def group_ambiguous_repeats_consecutively(pin_scores):
    groups, order = {}, []
    for pin, score in pin_scores:
        key = collapse_repeats(pin)
        if key not in groups:
            groups[key] = []; order.append(key)
        groups[key].append((pin, score))
    for k in groups: groups[k].sort(key=lambda x: x[1])
    out = []
    for k in order: out.extend(groups[k])
    return out

def prioritize_same_digit_pins(pin_scores, is_same_digit):
    if not is_same_digit: return pin_scores
    priority_pins = [str(d)*PIN_LENGTH for d in range(1,10)] + ["0"*PIN_LENGTH]
    prioritized_scores, non_priority = [], []
    for pin, score in pin_scores:
        if pin not in priority_pins: non_priority.append((pin, score))
    for p_pin in priority_pins:
        for pin, score in pin_scores:
            if pin == p_pin:
                prioritized_scores.append((pin, score)); break
    prioritized_scores.extend(non_priority)
    return prioritized_scores

def generate_repeated_digit_pins(centers, times, sizes):
    if len(centers) < 2 or len(centers) > 4: return []
    order = np.argsort(times)
    centers_ordered = centers[order]
    sizes_ordered = np.array(sizes)[order]
    nearest_digits = []
    for center in centers_ordered:
        distances = np.sqrt(np.sum((PINPAD_COORDS - center)**2, axis=1))
        nearest_digits.append(PINPAD_DIGITS[int(np.argmin(distances))])
    mean_size = np.mean(sizes_ordered)
    if mean_size == 0: return []
    relative_sizes = sizes_ordered / mean_size
    candidates = []
    pin = ''
    for digit, rel_size in zip(nearest_digits, relative_sizes):
        if rel_size > 1.5: pin += digit*2
        else:              pin += digit
    if len(pin) > len(nearest_digits): candidates.append(pin)
    pin = ''
    for digit, rel_size in zip(nearest_digits, relative_sizes):
        if   rel_size > 2.0: pin += digit*3
        elif rel_size > 1.3: pin += digit*2
        else:                pin += digit
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

def _fit_predict_kmeans_safe(features, n_clusters, random_state=0):
    try:
        return KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10).fit_predict(features)
    except Exception as e:
        print(f"[WARN] KMeans failed ({e}), falling back to MiniBatchKMeans.")
        return MiniBatchKMeans(n_clusters=n_clusters, random_state=random_state, batch_size=256).fit_predict(features)

def time_aware_clustering(points, frame_indices, n_clusters=4, time_weight=TIME_WEIGHT):
    if len(points) < n_clusters:
        return None, None, None
    scaler_space = StandardScaler()
    scaler_time = StandardScaler()
    points_scaled = scaler_space.fit_transform(points)
    time_scaled = scaler_time.fit_transform(frame_indices.reshape(-1, 1))
    space_time_features = np.hstack([points_scaled, time_weight * time_scaled])
    labels = _fit_predict_kmeans_safe(space_time_features, n_clusters)
    return get_cluster_centers_and_times(labels, points, frame_indices)

# def fit_translation_scaling(A, B):
#     if len(A) < 2 or len(B) < 2: return float('inf'), None
#     centroid_A = np.mean(A, axis=0); centroid_B = np.mean(B, axis=0)
#     AA = A - centroid_A; BB = B - centroid_B
#     norm_A = np.linalg.norm(AA); norm_B = np.linalg.norm(BB)
#     scale = 1.0 if (norm_A == 0 or norm_B == 0) else norm_B / norm_A
#     A2 = AA * scale + centroid_B
#     error = np.sqrt(np.mean(np.sum((A2 - B)**2, axis=1)))
#     return error, A2


def fit_translation_scaling(A, B, debug=False, pin_label="Unknown"):
    """
    Full Procrustes Analysis: Translation + Scaling + Rotation (via SVD)
    A: Observed sequence (N x 2)
    B: Template sequence (N x 2)
    
    Returns: (error_rms, A_prime) where A_prime is the transformed observed points
    """
    if len(A) < 2 or len(B) < 2:
        return float('inf'), None

    # 1. Compute centroids
    centroid_A = np.mean(A, axis=0)
    centroid_B = np.mean(B, axis=0)

    # 2. Center both point sets
    A_centered = A - centroid_A
    B_centered = B - centroid_B

    # 3. Compute norms for scaling
    norm_A = np.linalg.norm(A_centered)
    norm_B = np.linalg.norm(B_centered)

    if norm_A < 1e-9:
        return float('inf'), None

    # 4. Compute cross-covariance matrix H = A^T @ B
    H = A_centered.T @ B_centered

    # 5. SVD to find optimal rotation
    U, S, Vt = np.linalg.svd(H)
    
    # 6. Compute optimal rotation matrix R = V @ U^T
    R = Vt.T @ U.T

    # 7. Handle reflection case (ensure det(R) = +1, not -1)
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T

    # 8. Compute scale factor
    scale_s = norm_B / norm_A

    # 9. Apply full transformation: rotate, then scale, then translate
    A_prime = (A_centered @ R) * scale_s + centroid_B

    # 10. Compute RMS error
    diff = A_prime - B
    error_rms = np.sqrt(np.mean(np.sum(diff**2, axis=1)))

    if debug:
        # Extract rotation angle from R for debugging
        rotation_angle = np.degrees(np.arctan2(R[1, 0], R[0, 0]))
        print(f"\n--- [Full Procrustes Debug: {pin_label}] ---")
        print(f"Centroid A (Observed): {centroid_A}")
        print(f"Centroid B (Template): {centroid_B}")
        print(f"Rotation Matrix R:\n{R}")
        print(f"Rotation Angle: {rotation_angle:.2f}°")
        print(f"Scale Factor (s): {scale_s:.6f}")
        print(f"Final Registered A':\n{A_prime}")
        print(f"Procrustes Distance (RMS Error): {error_rms:.4f}")
        print("-" * 50)

    return error_rms, A_prime


def get_cluster_centers_and_times(labels, points, frame_indices=None):
    clusters = np.unique(labels)
    centers, times, sizes = [], [], []
    if frame_indices is None: frame_indices = np.arange(len(labels))
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
    if top_pins:
        top_pin = top_pins[0][0]
        unique_pin = ''
        for i, digit in enumerate(top_pin):
            if i == 0 or digit != top_pin[i - 1]:
                unique_pin += digit
        pin_indices = [PINPAD_DIGIT_TO_IDX[d] for d in unique_pin]
        pin_coords = PINPAD_COORDS[pin_indices]
        plt.plot(pin_coords[:, 0], pin_coords[:, 1], 'b-', linewidth=2, alpha=0.7, label=f"{top_pin} (ideal)")
    if len(centers_ordered) > 0 and top_pins:
        top_pin = top_pins[0][0]
        unique_pin = ''
        for i, digit in enumerate(top_pin):
            if i == 0 or digit != top_pin[i - 1]:
                unique_pin += digit
        pin_indices = [PINPAD_DIGIT_TO_IDX[d] for d in unique_pin]
        pin_coords = PINPAD_COORDS[pin_indices]
        centers_to_use = centers_ordered
        if len(centers_ordered) > len(unique_pin):
            centers_to_use = centers_ordered[:len(unique_pin)]
        _, transformed_centers = fit_translation_scaling(centers_to_use, pin_coords)
        if transformed_centers is not None:
            plt.plot(transformed_centers[:, 0], transformed_centers[:, 1], 'r--', alpha=0.8, zorder=5, label="Observed trajectory")
            plt.scatter(transformed_centers[:, 0], transformed_centers[:, 1], color='red', s=80, edgecolor='white', alpha=0.8, zorder=10)
            for i, (x, y) in enumerate(transformed_centers):
                plt.annotate(f"{i + 1}", xy=(x, y), xytext=(-5, 5), textcoords='offset points', fontsize=10, color='white', fontweight='bold')
    plt.title(title)
    plt.grid(False)
    plt.legend(loc="upper center", bbox_to_anchor=(0.5, -0.05), ncol=2)
    plt.tight_layout()
    plt.gca().invert_yaxis()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path)
    plt.close()

def extract_actual_pin_from_filename(video_name): return os.path.splitext(video_name)[0]

def find_pin_rank(pin, pin_scores):
    for rank, (candidate_pin, _) in enumerate(pin_scores, 1):
        if candidate_pin == pin:
            return rank
    return "Not Found"

def generate_individual_html_report(video_name, pin_scores, is_same_digit, actual_pin, actual_pin_rank, report_dir):
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
<p><strong>PIN Length:</strong> {PIN_LENGTH} digits</p>
<p><strong>Clustering Time Weight:</strong> {TIME_WEIGHT}</p>
<p><strong>Same Digit Box Size:</strong> {SAME_DIGIT_BOX_SIZE} pixels</p>
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
    trajectory_img_path = os.path.join(OUTPUT_DIR, video_name, 'trajectory_mapping.png')
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
    with open(report_path, 'w') as f:
        f.write(html)
    return os.path.basename(report_path)


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
    Optimized Procrustes scoring with partial vectorization.
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
        B_all = PINPAD_COORDS[idx]  # (n, L, 2)

        # A statistics (computed once)
        centroid_A = A_centers.mean(axis=0)
        A_centered = A_centers - centroid_A
        norm_A = np.linalg.norm(A_centered)
        
        if norm_A < 1e-9:
            pins = [''.join(map(str, row.tolist())) for row in digits]
            return pins, [float('inf')] * n

        # B statistics (vectorized)
        centroid_B = B_all.mean(axis=1)  # (n, 2)
        B_centered = B_all - centroid_B[:, np.newaxis, :]  # (n, L, 2)
        norm_B = np.sqrt((B_centered ** 2).sum(axis=(1, 2)))  # (n,)
        
        # Scale factors (vectorized)
        scales = norm_B / norm_A  # (n,)

        # Cross-covariance matrices H[i] = A_centered.T @ B_centered[i]
        # A_centered: (L, 2), B_centered: (n, L, 2)
        # H: (n, 2, 2)
        H = np.einsum('ji,nji->nij', A_centered, B_centered)

        errs = np.empty(n, dtype=np.float64)
        
        for i in range(n):
            if norm_B[i] < 1e-9:
                errs[i] = float('inf')
                continue
                
            # SVD per matrix (cannot easily vectorize)
            U, S, Vt = np.linalg.svd(H[i])
            R = Vt.T @ U.T
            
            if np.linalg.det(R) < 0:
                Vt[1, :] *= -1
                R = Vt.T @ U.T
            
            # Transform and compute error
            A_transformed = (A_centered @ R) * scales[i] + centroid_B[i]
            diff = A_transformed - B_all[i]
            errs[i] = np.sqrt(np.mean(np.sum(diff**2, axis=1)))

        pins = [''.join(map(str, row.tolist())) for row in digits]
        return pins, errs


def _score_chunk_worker(start, end, A_centers, top_m, pin_length=None):
    pins, errs = _vectorized_errors_for_chunk(start, end, A_centers, pin_length)
    if len(pins) == 0: return []
    errs = np.asarray(errs); n = len(errs)
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
    tasks, results = [], []
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
    if not results: return []
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


def process_csv_trajectory(csv_path, yaw_deg, pitch_deg, do_plot=False):
    """返回 pin_scores, is_same_digit, actual_pin, actual_pin_rank, centers_ordered"""
    video_dir = os.path.dirname(csv_path)
    video_name = os.path.basename(video_dir)
    print(f"\nProcessing video: {video_name}")
    actual_pin = extract_actual_pin_from_filename(video_name)
    if actual_pin: print(f"  Actual PIN identified from filename: {actual_pin}")

    df = pd.read_csv(csv_path)
    xcol, ycol = find_ring_center_cols(df)
    xs = df[xcol].values; ys = df[ycol].values
    mask = ~np.isnan(xs) & ~np.isnan(ys)
    xs, ys = xs[mask], ys[mask]
    points = np.stack([xs, ys], axis=1)
    frame_indices = np.arange(len(df))[mask]
    if points.shape[0] == 0:
        print(f"Warning: No valid points in {csv_path}. Skipping.")
        return [], False, actual_pin, "Not Found", None

    speeds = calculate_speeds(points)
    filtered_points, filtered_frames = filter_by_speed(points, speeds, frame_indices)
    print(f"  Filtered {len(points)} points to {len(filtered_points)} slow points")

    best_centers_ordered = None
    same_digit_center = None
    is_same_digit = are_all_points_close(filtered_points)
    if is_same_digit:
        print(f"  DETECTED: points within {SAME_DIGIT_BOX_SIZE} box - likely same-digit PIN")
        same_digit_center = np.mean(filtered_points, axis=0).reshape(1, 2)

    centers, times, sizes = time_aware_clustering(filtered_points, filtered_frames, n_clusters=PIN_LENGTH)
    if centers is not None and len(centers) == PIN_LENGTH:
        order = np.argsort(times)
        best_centers_ordered = centers[order]
    else:
        print("  Falling back to standard clustering")
        for k in [4, 3, 2]:
            if k > max(1, len(filtered_points) // 5): continue
            scaler = StandardScaler()
            filtered_points_scaled = scaler.fit_transform(filtered_points)
            labels = _fit_predict_kmeans_safe(filtered_points_scaled, k)
            centers2, times2, _ = get_cluster_centers_and_times(labels, filtered_points, filtered_frames)
            order = np.argsort(times2)
            centers_ordered = centers2[order]
            if best_centers_ordered is None:
                best_centers_ordered = centers_ordered
        if best_centers_ordered is None and same_digit_center is not None:
            best_centers_ordered = same_digit_center

    if best_centers_ordered is None:
        print(f"  WARNING: Could not find valid clustering for {video_name}")
        return [], is_same_digit, actual_pin, "Not Found", None

    obs = np.asarray(best_centers_ordered, dtype=float)
    obs_corr = obs.copy()
    obs_corr[:, 0] = obs[:, 0] / _cos_safe(yaw_deg)
    obs_corr[:, 1] = obs[:, 1] / _cos_safe(pitch_deg)

    dist_list = [math.hypot(obs_corr[i+1,0]-obs_corr[i,0],
                            obs_corr[i+1,1]-obs_corr[i,1]) for i in range(len(obs_corr)-1)]
    angle_list = [(math.degrees(math.atan2(obs_corr[i+1,1]-obs_corr[i,1],
                                           obs_corr[i+1,0]-obs_corr[i,0])) % 360.0)
                  for i in range(len(obs_corr)-1)]
    print("  IKD:", [round(d,2) for d in dist_list], " | IKR:", [round(a,1) for a in angle_list])

    best_centers_ordered = obs_corr

    print("  Scoring candidates (fast/vectorized)...")
    pin_trajectories = load_pin_trajectories(PIN_LENGTH)
    if pin_trajectories:
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
        batch_size = 50_000; pin_scores = []
        if NUM_WORKERS <= 1:
            for i in range(0, len(items), batch_size):
                pin_scores.extend(score_batch(items[i:i+batch_size]))
        else:
            with ProcessPoolExecutor(max_workers=NUM_WORKERS) as ex:
                futures = [ex.submit(score_batch, items[i:i+batch_size]) for i in range(0, len(items), batch_size)]
                for fut in as_completed(futures): pin_scores.extend(fut.result())
        pin_scores.sort(key=lambda x: x[1])
    else:
        pin_scores = score_all_pins_fast(best_centers_ordered, PIN_LENGTH, NUM_WORKERS, CHUNK_SIZE, TOPK_PER_CHUNK, TOPK_FINAL)

    if centers is not None and times is not None and sizes is not None:
        repeat_pins = generate_repeated_digit_pins(centers, times, sizes)
        for pin in repeat_pins:
            try:
                collapsed_pin = pin[0]
                for i in range(1, len(pin)):
                    if pin[i] != pin[i - 1]:
                        collapsed_pin += pin[i]
                if len(collapsed_pin) == len(best_centers_ordered):
                    pin_indices = [PINPAD_DIGIT_TO_IDX[d] for d in collapsed_pin]
                    pin_coords = PINPAD_COORDS[pin_indices]
                    error, _ = fit_translation_scaling(best_centers_ordered, pin_coords)
                    error = error * 0.95
                    pin_scores.append((pin, error))
            except Exception as e:
                print(f"  Error scoring variable-length PIN {pin}: {e}")

    pin_scores.sort(key=lambda x: x[1])
    pin_scores = prioritize_same_digit_pins(pin_scores, is_same_digit)
    raw_sorted_scores = list(pin_scores)
    pin_scores = filter_candidates(pin_scores, is_same_digit)
    pin_scores = group_ambiguous_repeats_consecutively(pin_scores)

    actual_pin_rank = "Not Found"
    if actual_pin:
        actual_pin_rank = find_pin_rank(actual_pin, pin_scores)
        print(f"  Actual PIN {actual_pin} rank = {actual_pin_rank}")

    if actual_pin and len(actual_pin) == len(best_centers_ordered):
        try:
            actual_indices = [PINPAD_DIGIT_TO_IDX[d] for d in actual_pin]
            actual_template = PINPAD_COORDS[actual_indices]
            fit_translation_scaling(best_centers_ordered, actual_template, debug=True, pin_label=actual_pin)
        except Exception as e:
            print(f"Debug print failed: {e}")

    if do_plot and best_centers_ordered is not None and len(best_centers_ordered) > 0:
        plot_list = []
        if actual_pin:
            found = next(((p, s) for (p, s) in pin_scores if p == actual_pin), None)
            if not found:
                found = next(((p, s) for (p, s) in raw_sorted_scores if p == actual_pin and (s is not None and s > 0)), None)
            if found: plot_list = [found]
        if not plot_list: plot_list = pin_scores[:1] if pin_scores else []
        title = 'Time-Aware Trajectory Matching'
        plot_trajectory_on_pinpad(
            best_centers_ordered, plot_list,
            os.path.join(os.path.dirname(csv_path), 'trajectory_mapping.png'),
            f'{title}: Using {"Actual PIN" if (plot_list and actual_pin and plot_list[0][0]==actual_pin) else "Top Candidate"} {plot_list[0][0] if plot_list else "N/A"}'
        )
    return pin_scores, is_same_digit, actual_pin, actual_pin_rank, best_centers_ordered


def _angle_list(a, b, step=5.0):
    a, b = float(a), float(b)
    if a > b: a, b = b, a
    vals = []
    x = a
    while x <= b + 1e-9:
        vals.append(round(x, 6))
        x += step
    if abs(vals[-1] - b) > 1e-6:
        vals.append(round(b, 6))
    vals = sorted(set(vals))
    return vals

def _tag_from_angles(yaw, pitch):
    def _fmt(v):
        return str(int(round(v))) if abs(v - round(v)) < 1e-6 else f"{v:.1f}".rstrip('0').rstrip('.')
    return f"yaw{_fmt(yaw)}_pitch{_fmt(pitch)}"


def generate_final_html(best_map, report_dir, yaw_values, pitch_values, angle_stats):
    """best_map: video -> dict with keys:
        actual_pin, best_rank (int or 'Not Found'), best_yaw, best_pitch,
        pin_scores (list at best angle), is_same_digit, top_pin, top_score, details_file
    """
    path = os.path.join(report_dir, 'index.html')
    os.makedirs(report_dir, exist_ok=True)

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Per-PIN Best Angle Summary</title>
<style>
body {{ font-family: Arial, sans-serif; line-height: 1.6; padding: 20px; color: #333; max-width: 1400px; margin: 0 auto; }}
h1 {{ color: #2c3e50; border-bottom: 2px solid #3498db; padding-bottom: 10px; }}
h2 {{ color: #2980b9; margin-top: 30px; }}
table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
th, td {{ text-align: left; padding: 10px 12px; border-bottom: 1px solid #ddd; }}
th {{ background-color: #f2f2f2; font-weight: bold; }}
tr:hover {{ background-color: #f9f9f9; }}
.pin-code {{ font-family: monospace; font-weight: bold; }}
.top-pins {{ font-family: monospace; color: #555; }}
.rank-ok {{ background-color: #d4edda; font-weight: bold; }}
.rank-bad {{ background-color: #f8d7da; }}
.small {{ color: #666; font-size: 0.95em; }}
.summary-table td, .summary-table th {{ text-align:center; }}
</style>
</head>
<body>
<h1>PIN Trajectory Analysis — Per-PIN Best Angle</h1>
<p class="small">Generated on: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
<div class="config-info">
<p><strong>PIN_LENGTH</strong>: {PIN_LENGTH} &nbsp;&nbsp; <strong>TIME_WEIGHT</strong>: {TIME_WEIGHT} &nbsp;&nbsp; <strong>SAME_DIGIT_BOX_SIZE</strong>: {SAME_DIGIT_BOX_SIZE}px</p>
<p><strong>Yaw sweep</strong>: {yaw_values} &nbsp;&nbsp; <strong>Pitch sweep</strong>: {pitch_values}</p>
<p><strong>Threshold</strong>: Actual PIN Rank &lt; {RANK_LT_THRESHOLD}</p>
</div>

<h2>Angle Summary (counts of videos with Actual PIN Rank &lt; {RANK_LT_THRESHOLD})</h2>
<table class="summary-table">
<tr><th>Yaw</th><th>Pitch</th><th>Count</th><th>Rank==1</th></tr>
"""
    # angle_stats: [{'yaw':..., 'pitch':..., 'ranklt':..., 'rank1':...}, ...] sorted desc
    for rec in sorted(angle_stats, key=lambda d: (d['ranklt'], d['rank1']), reverse=True):
        html += f"<tr><td>{rec['yaw']:.1f}°</td><td>{rec['pitch']:.1f}°</td><td>{rec['ranklt']}</td><td>{rec['rank1']}</td></tr>\n"
    html += "</table>\n"

    html += """
<h2>Per-PIN Best Results</h2>
<table>
<tr>
<th>Video</th>
<th>Actual PIN</th>
<th>Best Actual Rank</th>
<th>Best Angle (yaw, pitch)</th>
<th>Top PIN @ Best</th>
<th>Top 5 @ Best</th>
<th>Details</th>
</tr>
"""
    for video, rec in sorted(best_map.items()):
        best_rank = rec['best_rank']
        cls = "rank-ok" if isinstance(best_rank, int) and best_rank < RANK_LT_THRESHOLD else ""
        top5 = ", ".join([p for p,_ in rec['pin_scores'][:5]]) if rec['pin_scores'] else ""
        details = rec.get('details_file', '')
        html += f"""<tr class="{cls}">
<td>{video}</td>
<td class="pin-code">{rec.get('actual_pin','Unknown')}</td>
<td>{best_rank}</td>
<td>{rec['best_yaw']}°, {rec['best_pitch']}°</td>
<td class="pin-code">{rec.get('top_pin','')}</td>
<td class="top-pins">{top5}</td>
<td><a href="{details}">View</a></td>
</tr>
"""
    html += """
</table>
<p class="small">Analysis powered by time-aware clustering + vectorized Procrustes scoring (scale+translation).</p>
</body>
</html>
"""
    with open(path, 'w') as f:
        f.write(html)
    print(f"Final HTML written to: {path}")


def _collect_csvs():
    filtered_csv_files   = glob.glob(os.path.join(OUTPUT_DIR, '*', '*_filtered_ring_center.csv'))
    all_leds_csv_files   = glob.glob(os.path.join(OUTPUT_DIR, '*', '*_all_leds_center.csv'))
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
    print(f"Found {len(csv_files)} CSV files.")
    return csv_files

def _rank_to_val(r):
    if isinstance(r, int): return r
    if isinstance(r, str) and r.isdigit(): return int(r)
    return 10**9  # Not Found -> huge

def main():
    parser = argparse.ArgumentParser(description="PIN trajectory analysis (per-PIN best angle).")
    parser.add_argument("-i", "--input", type=str, default='../output_lux10_show',
                        help="Input directory containing CSV files")
    parser.add_argument("-o", "--output", type=str, default='./report_best_per_pin_lux10_show',
                        help="Output directory for reports")
    parser.add_argument("--yaw", type=float, nargs=2, default=[0.0, 0.0], metavar=('MIN', 'MAX'),
                        help="Yaw range in degrees (default: 0.0 0.0)")
    parser.add_argument("--pitch", type=float, nargs=2, default=[5.0, 10.0], metavar=('MIN', 'MAX'),
                        help="Pitch range in degrees (default: 5.0 10.0)")
    parser.add_argument("--step", type=float, default=5.0,
                        help="Step size in degrees for yaw/pitch sweep (default: 5.0)")
    parser.add_argument("--workers", type=int, default=NUM_WORKERS_DEFAULT)
    parser.add_argument("--chunk-size", type=int, default=CHUNK_SIZE_DEFAULT)
    parser.add_argument("--topk-per-chunk", type=int, default=TOPK_PER_CHUNK_DEFAULT)
    parser.add_argument("--topk-final", type=int, default=TOPK_FINAL_DEFAULT)
    parser.add_argument("--use-pkl-trajectories", action="store_true")
    parser.add_argument("--open", action="store_true", help="Open final report in browser")
    args = parser.parse_args()

    # Update global variables from arguments
    global OUTPUT_DIR, REPORT_FOLDER, YAW_RANGE, PITCH_RANGE, STEP_DEG
    OUTPUT_DIR = args.input
    REPORT_FOLDER = args.output
    YAW_RANGE = tuple(args.yaw)
    PITCH_RANGE = tuple(args.pitch)
    STEP_DEG = args.step

    global NUM_WORKERS, CHUNK_SIZE, TOPK_PER_CHUNK, TOPK_FINAL, USE_PKL_TRAJ
    NUM_WORKERS   = max(1, args.workers)
    CHUNK_SIZE    = max(10_000, args.chunk_size)
    TOPK_PER_CHUNK= max(1_000, args.topk_per_chunk)
    TOPK_FINAL    = max(5_000, args.topk_final)
    USE_PKL_TRAJ  = bool(args.use_pkl_trajectories)

    print(f"\nInput directory: {OUTPUT_DIR}")
    print(f"Output directory: {REPORT_FOLDER}")
    print(f"Yaw range: {YAW_RANGE}, Pitch range: {PITCH_RANGE}, Step: {STEP_DEG}°")
    print(f"Perf: workers={NUM_WORKERS}, chunk_size={CHUNK_SIZE}, topk_per_chunk={TOPK_PER_CHUNK}, topk_final={TOPK_FINAL}, use_pkl={USE_PKL_TRAJ}")
    print(f"Threshold for summary: Actual PIN Rank < {RANK_LT_THRESHOLD}\n")

    yaw_values   = _angle_list(YAW_RANGE[0],   YAW_RANGE[1],   STEP_DEG)
    pitch_values = _angle_list(PITCH_RANGE[0], PITCH_RANGE[1], STEP_DEG)
    print(f"Yaw sweep:   {yaw_values}")
    print(f"Pitch sweep: {pitch_values}")

    csv_files = _collect_csvs()
    if not csv_files:
        print("No CSVs found. Check OUTPUT_DIR.")
        return

    angle_stats = []
    per_video_best = {os.path.basename(os.path.dirname(p)): {
        'best_rank': 10**9, 'best_yaw': None, 'best_pitch': None,
        'pin_scores': None, 'is_same_digit': False, 'actual_pin': extract_actual_pin_from_filename(os.path.basename(os.path.dirname(p)))
    } for p in csv_files}

    for yaw in yaw_values:
        for pitch in pitch_values:
            tag = _tag_from_angles(yaw, pitch)
            print(f"\n===== Running angle {tag} =====")
            rank1_cnt = 0
            ranklt_cnt = 0
            for csv_path in csv_files:
                video_name = os.path.basename(os.path.dirname(csv_path))
                pin_scores, is_same_digit, actual_pin, actual_rank, _ = process_csv_trajectory(csv_path, yaw, pitch, do_plot=False)


                if isinstance(actual_rank, int) and actual_rank == 1: rank1_cnt += 1
                if isinstance(actual_rank, int) and actual_rank < RANK_LT_THRESHOLD: ranklt_cnt += 1

                cur_best = per_video_best[video_name]
                cur_val = _rank_to_val(cur_best['best_rank'])
                new_val = _rank_to_val(actual_rank)
                if new_val < cur_val:
                    per_video_best[video_name].update({
                        'best_rank': actual_rank,
                        'best_yaw': yaw,
                        'best_pitch': pitch,
                        'pin_scores': pin_scores,
                        'is_same_digit': is_same_digit,
                        'actual_pin': actual_pin
                    })
                elif new_val == cur_val and pin_scores:
                    old_top = cur_best['pin_scores'][0][1] if cur_best['pin_scores'] else float('inf')
                    if pin_scores[0][1] < old_top:
                        per_video_best[video_name].update({
                            'best_rank': actual_rank,
                            'best_yaw': yaw,
                            'best_pitch': pitch,
                            'pin_scores': pin_scores,
                            'is_same_digit': is_same_digit,
                            'actual_pin': actual_pin
                        })

            angle_stats.append({'yaw': yaw, 'pitch': pitch, 'rank1': rank1_cnt, 'ranklt': ranklt_cnt})
            print(f"[{tag}] rank<{RANK_LT_THRESHOLD}: {ranklt_cnt}, rank==1: {rank1_cnt}")

    if os.path.exists(REPORT_FOLDER): shutil.rmtree(REPORT_FOLDER, ignore_errors=True)
    os.makedirs(REPORT_FOLDER, exist_ok=True)

    for csv_path in csv_files:
        video = os.path.basename(os.path.dirname(csv_path))
        rec = per_video_best[video]
        if rec['pin_scores'] is None:
            rec['top_pin'] = ''
            rec['top_score'] = float('inf')
            rec['details_file'] = ''
            continue

        _ = process_csv_trajectory(csv_path, rec['best_yaw'], rec['best_pitch'], do_plot=True)
        top_pin, top_score = rec['pin_scores'][0] if rec['pin_scores'] else ('', float('inf'))
        rec['top_pin'] = top_pin; rec['top_score'] = top_score
        detail = generate_individual_html_report(
            video, rec['pin_scores'], rec['is_same_digit'], rec.get('actual_pin','Unknown'), rec['best_rank'], REPORT_FOLDER
        )
        rec['details_file'] = detail

    generate_final_html(per_video_best, REPORT_FOLDER, yaw_values, pitch_values, angle_stats)

    final_index = os.path.abspath(os.path.join(REPORT_FOLDER, "index.html"))
    print(f"\nFinal best-per-PIN report: {final_index}")
    if args.open:
        webbrowser.open('file://' + final_index)

if __name__ == '__main__':
    main()