'''
python 402_guess_90.py -i ../2_90/pitch_p90/output_e -o ../2_90/pitch_p90/report_e --camera-position top


Camera positions:
  left (yaw -90): Y directly observable, X (column) from depth
  right (yaw +90): Y directly observable, X (column) from depth  
  bottom (pitch -90): X directly observable, Y (row) from depth
  top (pitch +90): X directly observable, Y (row) from depth
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
from sklearn.mixture import GaussianMixture
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
from scipy.signal import find_peaks, savgol_filter

# ------------- High-performance defaults (tunable via CLI) -------------
NUM_WORKERS_DEFAULT = os.cpu_count() or 8
CHUNK_SIZE_DEFAULT = 100_000
TOPK_PER_CHUNK_DEFAULT = 20_000
TOPK_FINAL_DEFAULT = 200_000
USE_PKL_TRAJ_DEFAULT = True

# --- USER CONFIGURABLE PARAMETERS ---
PIN_LENGTH = 4
TIME_WEIGHT = 0.5
TRAJECTORIES_DIR = './pin_trajectories'

# These will be set via CLI arguments
OUTPUT_DIR = None
REPORT_FOLDER = None
CAMERA_POSITION = 'left'  # Default camera position

print(f"Using PIN length: {PIN_LENGTH} digits")

SAME_DIGIT_BOX_SIZE = 80

# Button dimensions for PIN pad
BUTTON_WIDTH = 10.0
BUTTON_HEIGHT = 5.5
GAP = 0.9
X_OFFSET = BUTTON_WIDTH/2
Y_OFFSET = BUTTON_HEIGHT/2

# PIN pad coordinate system
# Columns: 0 (1,4,7), 1 (2,5,8), 2 (3,6,9,0)
# Rows: 0 (1,2,3), 1 (4,5,6), 2 (7,8,9), 3 (0)
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

# Digit to column mapping (0=left, 1=middle, 2=right)
DIGIT_TO_COLUMN = {
    '1': 0, '4': 0, '7': 0,
    '2': 1, '5': 1, '8': 1, '0': 1,
    '3': 2, '6': 2, '9': 2
}

# Digit to row mapping (0=top, 1, 2, 3=bottom)
DIGIT_TO_ROW = {
    '1': 0, '2': 0, '3': 0,
    '4': 1, '5': 1, '6': 1,
    '7': 2, '8': 2, '9': 2,
    '0': 3
}

# Column to digits mapping
COLUMN_TO_DIGITS = {
    0: ['1', '4', '7'],
    1: ['2', '5', '8', '0'],
    2: ['3', '6', '9']
}

# Row to digits mapping
ROW_TO_DIGITS = {
    0: ['1', '2', '3'],
    1: ['4', '5', '6'],
    2: ['7', '8', '9'],
    3: ['0']
}

DEC_TO_PINPAD_IDX = np.array([PINPAD_DIGIT_TO_IDX[str(d)] for d in range(10)], dtype=np.int8)

loaded_trajectories = {}

NUM_WORKERS = NUM_WORKERS_DEFAULT
CHUNK_SIZE = CHUNK_SIZE_DEFAULT
TOPK_PER_CHUNK = TOPK_PER_CHUNK_DEFAULT
TOPK_FINAL = TOPK_FINAL_DEFAULT
USE_PKL_TRAJ = USE_PKL_TRAJ_DEFAULT


# ============================================================================
# Camera Position Specific Functions
# ============================================================================

def get_axis_mapping(camera_position):
    """
    Get axis mapping based on camera position.
    
    Returns dict with:
        - 'direct_axis': which world axis is directly observable ('X' or 'Y')
        - 'depth_axis': which world axis is inferred from depth ('X' or 'Y')
        - 'direct_maps_to': what the direct axis maps to on PIN pad ('column' or 'row')
        - 'depth_maps_to': what the depth axis maps to on PIN pad ('column' or 'row')
        - 'depth_direction': 'larger_closer' or 'larger_farther'
    """
    if camera_position in ['left', 'right']:
        # Lateral view: Y directly observable, X from depth
        return {
            'direct_axis': 'Y',
            'depth_axis': 'X',
            'direct_maps_to': 'row',
            'depth_maps_to': 'column',
            'depth_direction': 'larger_closer' if camera_position == 'left' else 'larger_farther',
            'direct_image_coord': 'ring_y',
            'perpendicular_image_coord': 'ring_x'
        }
    else:  # top, bottom
        # Top-down view: X directly observable, Y from depth
        return {
            'direct_axis': 'X',
            'depth_axis': 'Y',
            'direct_maps_to': 'column',
            'depth_maps_to': 'row',
            'depth_direction': 'larger_closer' if camera_position == 'bottom' else 'larger_farther',
            'direct_image_coord': 'ring_x',
            'perpendicular_image_coord': 'ring_y'
        }


def cluster_depth_for_axis(depth_values, max_clusters, axis_type='column'):
    """
    Cluster depth values to determine discrete positions on PIN pad.
    
    For columns (lateral view): max 3 clusters (3 columns)
    For rows (top-down view): max 4 clusters (4 rows)
    
    Args:
        depth_values: Array of depth proxy values (larger = closer to camera)
        max_clusters: Maximum number of clusters (3 for columns, 4 for rows)
        axis_type: 'column' or 'row'
        
    Returns:
        dict with cluster info
    """
    valid_mask = ~np.isnan(depth_values)
    valid_depth = depth_values[valid_mask].reshape(-1, 1)
    
    if len(valid_depth) < 2:
        return {
            'n_clusters': 1,
            'labels': np.zeros(len(depth_values), dtype=int),
            'cluster_centers': [np.nanmean(depth_values)],
            'cluster_to_position': {0: max_clusters // 2}  # Default to middle
        }
    
    # Use BIC to select optimal number of clusters
    best_bic = np.inf
    best_n = 1
    best_model = None
    
    for n in range(1, min(max_clusters + 1, len(valid_depth))):
        try:
            gmm = GaussianMixture(n_components=n, random_state=42, n_init=3)
            gmm.fit(valid_depth)
            bic = gmm.bic(valid_depth)
            if bic < best_bic:
                best_bic = bic
                best_n = n
                best_model = gmm
        except:
            continue
    
    # Get cluster assignments
    labels = np.full(len(depth_values), -1, dtype=int)
    if best_model is not None:
        labels[valid_mask] = best_model.predict(valid_depth)
    
    # Compute cluster centers
    cluster_centers = []
    for c in range(best_n):
        cluster_mask = labels == c
        if np.any(cluster_mask):
            cluster_centers.append(np.nanmean(depth_values[cluster_mask]))
        else:
            cluster_centers.append(np.nan)
    
    # Map clusters to positions
    # Larger depth = closer = smaller position index for left camera (column 0)
    # Sort clusters by depth value (descending for left camera)
    sorted_indices = np.argsort(cluster_centers)[::-1]  # Descending (largest first = closest)
    
    cluster_to_position = {}
    if axis_type == 'column':
        # 3 possible columns: 0, 1, 2
        if best_n == 1:
            cluster_to_position[0] = None  # Ambiguous - could be any column
        elif best_n == 2:
            # Two clusters: could be (0,1), (0,2), or (1,2)
            cluster_to_position[sorted_indices[0]] = 0  # Closest = column 0
            cluster_to_position[sorted_indices[1]] = 2  # Farthest = column 2
        else:  # 3 clusters
            cluster_to_position[sorted_indices[0]] = 0
            cluster_to_position[sorted_indices[1]] = 1
            cluster_to_position[sorted_indices[2]] = 2
    else:  # row
        # 4 possible rows: 0, 1, 2, 3
        if best_n == 1:
            cluster_to_position[0] = None
        elif best_n == 2:
            cluster_to_position[sorted_indices[0]] = 0  # Closest
            cluster_to_position[sorted_indices[1]] = 3  # Farthest
        elif best_n == 3:
            cluster_to_position[sorted_indices[0]] = 0
            cluster_to_position[sorted_indices[1]] = 1
            cluster_to_position[sorted_indices[2]] = 3
        else:  # 4 clusters
            cluster_to_position[sorted_indices[0]] = 0
            cluster_to_position[sorted_indices[1]] = 1
            cluster_to_position[sorted_indices[2]] = 2
            cluster_to_position[sorted_indices[3]] = 3
    
    return {
        'n_clusters': best_n,
        'labels': labels,
        'cluster_centers': cluster_centers,
        'cluster_to_position': cluster_to_position,
        'sorted_indices': sorted_indices.tolist()
    }


def detect_keypresses(ring_center_df, depth_df=None, min_pause_frames=5, speed_percentile=30):
    """
    Detect keypress moments based on low velocity periods.
    
    Args:
        ring_center_df: DataFrame with ring_x, ring_y columns
        depth_df: DataFrame with depth_proxy_smoothed column (optional)
        min_pause_frames: Minimum frames for a pause to be considered a keypress
        speed_percentile: Percentile threshold for "slow" movement
        
    Returns:
        List of dicts with keypress info: frame, x, y, depth_proxy
    """
    if 'ring_x' in ring_center_df.columns:
        x = ring_center_df['ring_x'].values
        y = ring_center_df['ring_y'].values
    elif 'center_x' in ring_center_df.columns:
        x = ring_center_df['center_x'].values
        y = ring_center_df['center_y'].values
    else:
        raise ValueError("Could not find ring center columns")
    
    # Calculate speeds
    dx = np.diff(x)
    dy = np.diff(y)
    speeds = np.sqrt(dx**2 + dy**2)
    speeds = np.insert(speeds, 0, np.median(speeds))
    
    # Smooth speeds
    if len(speeds) > 11:
        speeds_smooth = savgol_filter(speeds, 11, 2)
    else:
        speeds_smooth = speeds
    
    # Find slow periods
    threshold = np.percentile(speeds_smooth, speed_percentile)
    slow_mask = speeds_smooth <= threshold
    
    # Find contiguous slow regions
    keypresses = []
    in_pause = False
    pause_start = 0
    
    for i in range(len(slow_mask)):
        if slow_mask[i] and not in_pause:
            in_pause = True
            pause_start = i
        elif not slow_mask[i] and in_pause:
            in_pause = False
            pause_length = i - pause_start
            if pause_length >= min_pause_frames:
                # Use middle of pause as keypress moment
                mid_frame = pause_start + pause_length // 2
                keypress_info = {
                    'frame': mid_frame,
                    'x': np.mean(x[pause_start:i]),
                    'y': np.mean(y[pause_start:i]),
                    'start_frame': pause_start,
                    'end_frame': i - 1
                }
                
                # Add depth info if available
                if depth_df is not None and 'depth_proxy_smoothed' in depth_df.columns:
                    depth_vals = depth_df['depth_proxy_smoothed'].values[pause_start:i]
                    keypress_info['depth_proxy'] = np.nanmean(depth_vals)
                    
                    if 'depth_cluster' in depth_df.columns:
                        cluster_vals = depth_df['depth_cluster'].values[pause_start:i]
                        # Use mode (most common cluster)
                        unique, counts = np.unique(cluster_vals[cluster_vals >= 0], return_counts=True)
                        if len(unique) > 0:
                            keypress_info['depth_cluster'] = unique[np.argmax(counts)]
                
                keypresses.append(keypress_info)
    
    # Handle case where video ends during a pause
    if in_pause:
        pause_length = len(slow_mask) - pause_start
        if pause_length >= min_pause_frames:
            mid_frame = pause_start + pause_length // 2
            keypress_info = {
                'frame': mid_frame,
                'x': np.mean(x[pause_start:]),
                'y': np.mean(y[pause_start:]),
                'start_frame': pause_start,
                'end_frame': len(slow_mask) - 1
            }
            if depth_df is not None and 'depth_proxy_smoothed' in depth_df.columns:
                depth_vals = depth_df['depth_proxy_smoothed'].values[pause_start:]
                keypress_info['depth_proxy'] = np.nanmean(depth_vals)
            keypresses.append(keypress_info)
    
    return keypresses


def map_keypress_to_digit_candidates(keypress, axis_mapping, depth_cluster_info, 
                                      direct_axis_bounds, camera_position):
    """
    Map a keypress to candidate digits based on camera view geometry.
    
    Args:
        keypress: Dict with x, y, depth_proxy, depth_cluster
        axis_mapping: Dict from get_axis_mapping()
        depth_cluster_info: Dict from cluster_depth_for_axis()
        direct_axis_bounds: (min, max) for the directly observable axis
        camera_position: 'left', 'right', 'bottom', 'top'
        
    Returns:
        List of candidate digits
    """
    candidates = []
    
    # Get the directly observable coordinate
    if axis_mapping['direct_maps_to'] == 'row':
        # Y is direct, maps to row
        direct_val = keypress['y']
        direct_min, direct_max = direct_axis_bounds
        
        # Normalize to 0-1 range
        if direct_max > direct_min:
            normalized = (direct_val - direct_min) / (direct_max - direct_min)
        else:
            normalized = 0.5
        
        # Map to row (0-3)
        # For left/right cameras: smaller y = top of image = row 0
        if camera_position == 'left':
            row = int(np.clip(normalized * 4, 0, 3))
        else:  # right camera - might need to invert
            row = int(np.clip(normalized * 4, 0, 3))
        
        row_digits = ROW_TO_DIGITS.get(row, ['5'])  # Default to middle
        
        # Get column from depth cluster
        if 'depth_cluster' in keypress and depth_cluster_info['n_clusters'] > 1:
            cluster = keypress['depth_cluster']
            col = depth_cluster_info['cluster_to_position'].get(cluster, None)
            
            if col is not None:
                col_digits = COLUMN_TO_DIGITS.get(col, ['1', '2', '3'])
                # Find intersection
                for d in row_digits:
                    if d in col_digits:
                        candidates.append(d)
                if not candidates:
                    candidates = row_digits  # Fall back to all row digits
            else:
                candidates = row_digits
        else:
            # Single cluster - column is ambiguous
            candidates = row_digits
            
    else:  # direct_maps_to == 'column'
        # X is direct, maps to column
        direct_val = keypress['x']
        direct_min, direct_max = direct_axis_bounds
        
        if direct_max > direct_min:
            normalized = (direct_val - direct_min) / (direct_max - direct_min)
        else:
            normalized = 0.5
        
        # Map to column (0-2)
        col = int(np.clip(normalized * 3, 0, 2))
        col_digits = COLUMN_TO_DIGITS.get(col, ['2', '5', '8', '0'])
        
        # Get row from depth cluster
        if 'depth_cluster' in keypress and depth_cluster_info['n_clusters'] > 1:
            cluster = keypress['depth_cluster']
            row = depth_cluster_info['cluster_to_position'].get(cluster, None)
            
            if row is not None:
                row_digits = ROW_TO_DIGITS.get(row, ['1', '2', '3'])
                for d in col_digits:
                    if d in row_digits:
                        candidates.append(d)
                if not candidates:
                    candidates = col_digits
            else:
                candidates = col_digits
        else:
            candidates = col_digits
    
    return candidates if candidates else ['5']  # Default to center


def generate_pin_candidates_from_keypresses(keypresses, axis_mapping, depth_cluster_info,
                                            direct_axis_bounds, camera_position, pin_length):
    """
    Generate all possible PIN candidates from detected keypresses.
    
    Args:
        keypresses: List of keypress dicts
        axis_mapping: Dict from get_axis_mapping()
        depth_cluster_info: Dict from cluster_depth_for_axis()
        direct_axis_bounds: (min, max) for directly observable axis
        camera_position: Camera position string
        pin_length: Expected PIN length
        
    Returns:
        List of (pin, score) tuples where each PIN has exactly pin_length digits
    """
    if len(keypresses) == 0:
        return []
    
    # Get digit candidates for each keypress
    all_digit_candidates = []
    for kp in keypresses:
        digits = map_keypress_to_digit_candidates(
            kp, axis_mapping, depth_cluster_info, direct_axis_bounds, camera_position
        )
        all_digit_candidates.append(digits)
    
    # Ensure we have exactly pin_length positions
    if len(all_digit_candidates) > pin_length:
        # Use time-ordered subset of keypresses
        best_candidates = all_digit_candidates[:pin_length]
    elif len(all_digit_candidates) < pin_length:
        # Pad with all digits as candidates for missing positions
        best_candidates = all_digit_candidates.copy()
        while len(best_candidates) < pin_length:
            best_candidates.append(['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'])
    else:
        best_candidates = all_digit_candidates
    
    # Generate all combinations
    all_pins = []
    for combo in product(*best_candidates):
        pin = ''.join(combo)
        # Only include PINs of exactly pin_length
        if len(pin) == pin_length:
            # Score based on how confident we are (fewer candidates per position = better)
            confidence = sum(1.0 / len(cands) for cands in best_candidates)
            score = 1.0 / (confidence + 0.1)  # Lower score = better
            all_pins.append((pin, score))
    
    # Sort by score only
    all_pins.sort(key=lambda x: x[1])
    
    return all_pins


# ============================================================================
# Original Helper Functions (kept for compatibility)
# ============================================================================

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
    if max_width is None:
        max_width = SAME_DIGIT_BOX_SIZE
    if max_height is None:
        max_height = SAME_DIGIT_BOX_SIZE
    if len(points) < 5:
        return False
    x_min, y_min = np.min(points, axis=0)
    x_max, y_max = np.max(points, axis=0)
    width = x_max - x_min
    height = y_max - y_min
    return width <= max_width and height <= max_height


def calculate_speeds(points):
    velocities = np.diff(points, axis=0)
    speeds = np.linalg.norm(velocities, axis=1)
    speeds = np.insert(speeds, 0, np.median(speeds))
    return speeds


def is_same_digit_pin(pin):
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
    filtered = [(p, s) for p, s in pin_scores if s is not None and s > 0]
    if not is_same_digit:
        filtered = [(p, s) for p, s in filtered if not is_same_digit_pin(p)]
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


def prioritize_same_digit_pins(pin_scores, is_same_digit):
    if not is_same_digit:
        return pin_scores
    priority_pins = [str(digit)*PIN_LENGTH for digit in range(1, 10)] + ["0"*PIN_LENGTH]
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


def fit_translation_scaling(A, B):
    if len(A) < 2 or len(B) < 2:
        return float('inf'), None
    centroid_A = np.mean(A, axis=0)
    centroid_B = np.mean(B, axis=0)
    AA = A - centroid_A
    BB = B - centroid_B
    norm_A = np.linalg.norm(AA)
    norm_B = np.linalg.norm(BB)
    scale = 1.0 if (norm_A == 0 or norm_B == 0) else norm_B / norm_A
    A2 = AA * scale + centroid_B
    error = np.sqrt(np.mean(np.sum((A2 - B)**2, axis=1)))
    return error, A2


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


def filter_by_speed(points, speeds, frame_indices=None):
    """
    Filter points by speed using KMeans clustering (k=2).
    Splits points into 'slow' and 'fast' clusters, keeping only slow points.
    """
    # Handle edge case: not enough points for clustering
    if len(speeds) < 2:
        if frame_indices is not None:
            return points, frame_indices
        else:
            return points, np.arange(len(points))
    
    # Reshape speeds for KMeans (needs 2D array)
    speeds_reshaped = speeds.reshape(-1, 1)
    
    # Apply KMeans with k=2 to split into slow/fast clusters
    kmeans = KMeans(n_clusters=2, random_state=0, n_init=10)
    labels = kmeans.fit_predict(speeds_reshaped)
    
    # Determine which cluster is "slow" (lower centroid value)
    cluster_centers = kmeans.cluster_centers_.flatten()
    slow_cluster = 0 if cluster_centers[0] < cluster_centers[1] else 1
    
    # Create mask for slow points
    slow_mask = labels == slow_cluster
    
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


# ============================================================================
# Plotting Functions
# ============================================================================

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


def plot_keypress_analysis(keypresses, ring_center_df, depth_df, out_path, video_name, camera_position):
    """Plot keypress detection and depth analysis."""
    fig, axes = plt.subplots(3, 1, figsize=(14, 12), sharex=True)
    
    frames = ring_center_df['frame'].values if 'frame' in ring_center_df.columns else np.arange(len(ring_center_df))
    
    # Get ring center coordinates
    if 'ring_x' in ring_center_df.columns:
        ring_x = ring_center_df['ring_x'].values
        ring_y = ring_center_df['ring_y'].values
    else:
        ring_x = ring_center_df['center_x'].values
        ring_y = ring_center_df['center_y'].values
    
    # Top: Position over time
    ax1 = axes[0]
    ax1.plot(frames, ring_x, 'b-', label='X position', alpha=0.7)
    ax1.plot(frames, ring_y, 'r-', label='Y position', alpha=0.7)
    
    # Mark keypresses
    for kp in keypresses:
        ax1.axvline(x=kp['frame'], color='green', linestyle='--', alpha=0.5)
        ax1.axvspan(kp['start_frame'], kp['end_frame'], alpha=0.2, color='green')
    
    ax1.set_ylabel('Position (pixels)')
    ax1.set_title(f'{video_name} - Keypress Detection (Camera: {camera_position})')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Middle: Speed
    ax2 = axes[1]
    dx = np.diff(ring_x)
    dy = np.diff(ring_y)
    speeds = np.sqrt(dx**2 + dy**2)
    speeds = np.insert(speeds, 0, np.median(speeds))
    
    ax2.plot(frames, speeds, 'k-', alpha=0.7, label='Speed')
    threshold = np.percentile(speeds, 30)
    ax2.axhline(y=threshold, color='orange', linestyle='--', label=f'Threshold (30th percentile)')
    
    for kp in keypresses:
        ax2.axvspan(kp['start_frame'], kp['end_frame'], alpha=0.2, color='green')
    
    ax2.set_ylabel('Speed (pixels/frame)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Bottom: Depth proxy
    ax3 = axes[2]
    if depth_df is not None and 'depth_proxy_smoothed' in depth_df.columns:
        ax3.plot(frames[:len(depth_df)], depth_df['depth_proxy_smoothed'].values, 
                 'purple', label='Depth Proxy (smoothed)')
        
        # Color by cluster if available
        if 'depth_cluster' in depth_df.columns:
            clusters = depth_df['depth_cluster'].values
            for c in np.unique(clusters):
                if c >= 0:
                    mask = clusters == c
                    ax3.scatter(frames[:len(depth_df)][mask], 
                               depth_df['depth_proxy_smoothed'].values[mask],
                               s=10, alpha=0.5, label=f'Cluster {c}')
        
        # Mark keypress depth values
        for i, kp in enumerate(keypresses):
            if 'depth_proxy' in kp:
                ax3.scatter([kp['frame']], [kp['depth_proxy']], 
                           s=100, c='red', marker='*', zorder=10)
                ax3.annotate(f'KP{i+1}', (kp['frame'], kp['depth_proxy']),
                            xytext=(5, 5), textcoords='offset points')
    
    ax3.set_xlabel('Frame')
    ax3.set_ylabel('Depth Proxy\n(Larger = Closer)')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


# ============================================================================
# Vectorized Scoring Functions
# ============================================================================

def _digits_from_range(start, end, pin_length):
    n = end - start
    arr = np.arange(start, end, dtype=np.int64)
    digits = np.empty((n, pin_length), dtype=np.int8)
    for pos in range(pin_length - 1, -1, -1):
        digits[:, pos] = (arr % 10).astype(np.int8)
        arr //= 10
    return digits


def _vectorized_errors_for_chunk(start, end, A_centers, pin_length=None):
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
        
        if norm_A == 0:
            scale = np.ones_like(norm_B)
        else:
            scale = norm_B / norm_A
            
        A2 = AA * scale[:, None, None] + centroid_B
        diff = A2 - B
        errs = np.sqrt(np.mean(np.sum(diff**2, axis=2), axis=1))
        
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
    print(f"  Fast scoring over {total:,} candidates: workers={num_workers}, chunk_size={chunk_size}")
    
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


# ============================================================================
# Report Generation Functions
# ============================================================================

def extract_actual_pin_from_filename(video_name):
    return os.path.splitext(video_name)[0]


def find_pin_rank(pin, pin_scores):
    for rank, (candidate_pin, _) in enumerate(pin_scores, 1):
        if candidate_pin == pin:
            return rank
    return "Not Found"


def generate_individual_html_report(video_name, pin_scores, is_same_digit, actual_pin, 
                                     actual_pin_rank, report_dir, camera_position):
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
.config-info {{ background-color: #e8f4f8; padding: 15px; border-radius: 5px; margin-bottom: 20px; }}
</style>
</head>
<body>
<div class="back-link"><a href="index.html">← Back to main report</a></div>
<h1>PIN Analysis for Video: {video_name}</h1>
<div class="config-info">
<p><strong>Camera Position:</strong> {camera_position}</p>
<p><strong>PIN Length:</strong> {PIN_LENGTH} digits</p>
<p><strong>Actual PIN:</strong> {actual_pin if actual_pin else 'Unknown'}</p>
<p><strong>Actual PIN Rank:</strong> {actual_pin_rank}</p>
</div>
<p>Generated on: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
<h2>PIN Candidates</h2>
<div class="pin-table-container">
<table>
<tr><th class="pin-rank">Rank</th><th class="pin-code">PIN</th><th class="pin-score">Score</th></tr>
"""
    for rank, (pin, score) in enumerate(filtered_scores[:500], 1):
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
    # Add trajectory image
    trajectory_img_path = os.path.join(OUTPUT_DIR, video_name, 'trajectory_mapping.png')
    if os.path.exists(trajectory_img_path):
        report_img_path = os.path.join(report_dir, f"{video_name}_trajectory.png")
        shutil.copy(trajectory_img_path, report_img_path)
        html += f"""
<h2>Trajectory Visualization</h2>
<img class="trajectory-image" src="{os.path.basename(report_img_path)}" alt="Trajectory plot">
"""
    
    # Add keypress analysis image
    keypress_img_path = os.path.join(OUTPUT_DIR, video_name, 'plots', f'{video_name}_keypress_analysis.png')
    if os.path.exists(keypress_img_path):
        report_kp_img_path = os.path.join(report_dir, f"{video_name}_keypress_analysis.png")
        shutil.copy(keypress_img_path, report_kp_img_path)
        html += f"""
<h2>Keypress Analysis</h2>
<img class="trajectory-image" src="{os.path.basename(report_kp_img_path)}" alt="Keypress analysis">
"""

    html += """
<div class="back-link"><a href="index.html">← Back to main report</a></div>
<div class="timestamp"><p>Analysis powered by Camera-Aware PIN Detection</p></div>
</body>
</html>
"""
    with open(report_path, 'w') as f:
        f.write(html)
    return os.path.basename(report_path)


def generate_main_html_report(results, pattern_info, report_dir, video_reports, actual_pins, camera_position):
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
.pin-rank { font-weight: bold; color: #e74c3c; width: 60px; text-align: center; }
.pin-code { font-family: monospace; font-size: 1.2em; font-weight: bold; width: 100px; }
.pin-score { color: #7f8c8d; width: 100px; }
.timestamp { color: #95a5a6; font-style: italic; text-align: right; margin-top: 50px; }
.details-link a { display: inline-block; background-color: #3498db; color: white; padding: 8px 15px; text-decoration: none; border-radius: 4px; }
.details-link a:hover { background-color: #2980b9; }
.config-info { background-color: #f9f9f9; padding: 15px; border-radius: 5px; margin-bottom: 20px; }
.config-param { font-family: monospace; font-weight: bold; }
.actual-pin { background-color: #d4edda; font-weight: bold; }
.rank-1 { color: #28a745; background-color: #d4edda; }
.rank-5-plus { color: #dc3545; background-color: #f8d7da; }
.rank-not-found { color: #6c757d; background-color: #e9ecef; }
</style>
</head>
<body>
<h1>PIN Trajectory Analysis - Summary Report</h1>
<p>Generated on: """ + datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S') + """</p>
<div class="config-info">
<h3>Configuration Parameters</h3>
<p><span class="config-param">CAMERA_POSITION</span>: """ + camera_position + """</p>
<p><span class="config-param">PIN_LENGTH</span>: """ + str(PIN_LENGTH) + """</p>
<p><span class="config-param">TIME_WEIGHT</span>: """ + str(TIME_WEIGHT) + """</p>
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
            top_pin = filtered_scores[0][0] if filtered_scores else "N/A"
            top_score = filtered_scores[0][1] if filtered_scores else 0
            report_filename = video_reports.get(video_name, "")
            actual_pin = actual_pins.get(video_name, {}).get('pin', 'Unknown')
            actual_pin_rank = actual_pins.get(video_name, {}).get('rank', 'N/A')
            
            rank_class = ""
            if actual_pin_rank == 1:
                rank_class = "rank-1"
            elif actual_pin_rank != "Not Found" and isinstance(actual_pin_rank, int) and actual_pin_rank > 5:
                rank_class = "rank-5-plus"
            elif actual_pin_rank == "Not Found":
                rank_class = "rank-not-found"
                
            top_5_pins = ", ".join([ps[0] for ps in filtered_scores[:5]])
            row_class = "actual-pin" if actual_pin and top_pin == actual_pin else ""
            
            html += f"""
<tr class="{row_class}">
<td>{video_name}</td>
<td class="pin-code">{actual_pin}</td>
<td class="actual-pin-rank {rank_class}">{actual_pin_rank}</td>
<td class="pin-code">{top_pin}</td>
<td class="pin-score">{top_score:.4f}</td>
<td>{top_5_pins}</td>
<td class="details-link"><a href="{report_filename}">View Details</a></td>
</tr>"""
    
    html += """
</table>
<div class="timestamp">
<p>Analysis powered by Camera-Aware PIN Detection</p>
</div>
</body>
</html>
"""
    with open(report_path, 'w') as f:
        f.write(html)
    print(f"Main HTML report generated at: {report_path}")


# ============================================================================
# Main Processing Function (Updated for Camera Position)
# ============================================================================

def process_csv_trajectory(csv_path, report_dir, camera_position):
    """
    Process trajectory CSV with camera-position-aware analysis.
    """
    video_dir = os.path.dirname(csv_path)
    video_name = os.path.basename(video_dir)
    os.makedirs(report_dir, exist_ok=True)
    plots_dir = os.path.join(video_dir, 'plots')
    os.makedirs(plots_dir, exist_ok=True)
    
    print(f"\nProcessing video: {video_name}")
    print(f"  Camera position: {camera_position}")
    
    actual_pin = extract_actual_pin_from_filename(video_name)
    if actual_pin:
        print(f"  Actual PIN identified from filename: {actual_pin}")
    
    # Load ring center data
    df = pd.read_csv(csv_path)
    xcol, ycol = find_ring_center_cols(df)
    
    # Add frame column if not present
    if 'frame' not in df.columns:
        df['frame'] = np.arange(len(df))
    
    xs = df[xcol].values
    ys = df[ycol].values
    mask = ~np.isnan(xs) & ~np.isnan(ys)
    
    ring_center_df = pd.DataFrame({
        'frame': df['frame'].values,
        'ring_x': xs,
        'ring_y': ys
    })
    
    # Load depth estimation data if available
    depth_csv_path = os.path.join(video_dir, f'{video_name}_depth_estimation.csv')
    depth_df = None
    if os.path.exists(depth_csv_path):
        depth_df = pd.read_csv(depth_csv_path)
        print(f"  Loaded depth estimation data")
    else:
        print(f"  Warning: Depth estimation file not found: {depth_csv_path}")
    
    # Load 3D trajectory if available
    trajectory_csv_path = os.path.join(video_dir, f'{video_name}_trajectory_3d.csv')
    trajectory_df = None
    if os.path.exists(trajectory_csv_path):
        trajectory_df = pd.read_csv(trajectory_csv_path)
        print(f"  Loaded 3D trajectory data")
    
    # Get axis mapping for this camera position
    axis_mapping = get_axis_mapping(camera_position)
    print(f"  Axis mapping: direct={axis_mapping['direct_axis']} → {axis_mapping['direct_maps_to']}, "
          f"depth={axis_mapping['depth_axis']} → {axis_mapping['depth_maps_to']}")
    
    # Detect keypresses
    print(f"  Detecting keypresses...")
    keypresses = detect_keypresses(ring_center_df, depth_df, min_pause_frames=5, speed_percentile=30)
    print(f"  Detected {len(keypresses)} potential keypresses")
    
    # Cluster depth values for the depth-inferred axis
    depth_cluster_info = {'n_clusters': 1, 'labels': np.zeros(len(df)), 'cluster_to_position': {0: 1}}
    
    if depth_df is not None and 'depth_proxy_smoothed' in depth_df.columns:
        # Determine max clusters based on what axis depth maps to
        max_clusters = 3 if axis_mapping['depth_maps_to'] == 'column' else 4
        
        # Get depth values at keypress moments
        keypress_depths = []
        for kp in keypresses:
            if 'depth_proxy' in kp:
                keypress_depths.append(kp['depth_proxy'])
        
        if len(keypress_depths) >= 2:
            depth_cluster_info = cluster_depth_for_axis(
                np.array(keypress_depths), 
                max_clusters, 
                axis_mapping['depth_maps_to']
            )
            print(f"  Clustered keypress depths into {depth_cluster_info['n_clusters']} levels")
            
            # Assign clusters back to keypresses
            if depth_cluster_info['n_clusters'] > 1:
                from sklearn.mixture import GaussianMixture
                gmm = GaussianMixture(n_components=depth_cluster_info['n_clusters'], random_state=42)
                gmm.fit(np.array(keypress_depths).reshape(-1, 1))
                for i, kp in enumerate(keypresses):
                    if 'depth_proxy' in kp:
                        kp['depth_cluster'] = gmm.predict([[kp['depth_proxy']]])[0]
    
    # Generate PIN candidates based on camera view
    pin_scores = []
    
    if len(keypresses) >= 2:
        # Calculate bounds for the directly observable axis
        if axis_mapping['direct_maps_to'] == 'row':
            direct_values = [kp['y'] for kp in keypresses]
        else:
            direct_values = [kp['x'] for kp in keypresses]
        
        direct_min = min(direct_values) - 10
        direct_max = max(direct_values) + 10
        direct_axis_bounds = (direct_min, direct_max)
        
        # Generate candidates from keypresses
        keypress_candidates = generate_pin_candidates_from_keypresses(
            keypresses, axis_mapping, depth_cluster_info,
            direct_axis_bounds, camera_position, PIN_LENGTH
        )
        print(f"  Generated {len(keypress_candidates)} candidates from keypress analysis")
        
        # Merge with trajectory-based scoring
        pin_scores.extend(keypress_candidates)
    
    # Also run traditional trajectory matching for comparison
    points = np.stack([xs[mask], ys[mask]], axis=1)
    frame_indices = np.arange(len(df))[mask]
    
    if points.shape[0] > 0:
        speeds = calculate_speeds(points)
        filtered_points, filtered_frames = filter_by_speed(points, speeds, frame_indices)
        
        is_same_digit = are_all_points_close(filtered_points)
        if is_same_digit:
            print(f"  DETECTED: All points within {SAME_DIGIT_BOX_SIZE}x{SAME_DIGIT_BOX_SIZE} box")
        
        # Time-aware clustering
        centers, times, sizes = time_aware_clustering(filtered_points, filtered_frames, n_clusters=PIN_LENGTH)
        
        if centers is not None and len(centers) == PIN_LENGTH:
            order = np.argsort(times)
            best_centers_ordered = centers[order]
            
            # Fast trajectory scoring
            traj_scores = score_all_pins_fast(
                best_centers_ordered, PIN_LENGTH, NUM_WORKERS, CHUNK_SIZE, TOPK_PER_CHUNK, TOPK_FINAL
            )
            
            # Merge scores (boost keypress-based candidates)
            score_dict = {pin: score for pin, score in traj_scores}
            for pin, kp_score in pin_scores:
                if pin in score_dict:
                    # Average the scores, giving weight to keypress analysis
                    score_dict[pin] = (score_dict[pin] + kp_score) / 2 * 0.9  # 10% boost
                else:
                    score_dict[pin] = kp_score
            
            pin_scores = [(pin, score) for pin, score in score_dict.items()]
        else:
            # Use only keypress-based candidates
            pass
    else:
        is_same_digit = False
        best_centers_ordered = None
    
    # Sort and filter
    pin_scores.sort(key=lambda x: x[1])
    pin_scores = filter_candidates(pin_scores, is_same_digit)
    
    # Find actual PIN rank
    actual_pin_rank = "Not Found"
    if actual_pin:
        actual_pin_rank = find_pin_rank(actual_pin, pin_scores)
        print(f"  Actual PIN {actual_pin} found at rank {actual_pin_rank}")
    
    # Apply dynamic cutoff
    if pin_scores:
        best_score = pin_scores[0][1]
        all_scores = np.array([score for _, score in pin_scores])
        score_std = np.std(all_scores)
        
        min_candidates = 500
        stat_threshold = best_score + 2 * score_std if score_std > 0 else best_score * 2
        stat_idx = next((i for i, (_, s) in enumerate(pin_scores) if s > stat_threshold), len(pin_scores))
        cutoff_idx = max(min_candidates, stat_idx)
        pin_scores = pin_scores[:cutoff_idx]
    
    # Print top candidates
    print("\nTop PIN candidates:")
    for pin, score in pin_scores[:min(10, len(pin_scores))]:
        marker = " <-- ACTUAL" if actual_pin and pin == actual_pin else ""
        print(f"  {pin}: {score:.4f}{marker}")
    
    # Generate plots
    if keypresses:
        keypress_plot_path = os.path.join(plots_dir, f'{video_name}_keypress_analysis.png')
        plot_keypress_analysis(keypresses, ring_center_df, depth_df, 
                               keypress_plot_path, video_name, camera_position)
    
    if best_centers_ordered is not None and len(best_centers_ordered) > 0 and pin_scores:
        plot_list = []
        if actual_pin:
            found = next(((p, s) for (p, s) in pin_scores if p == actual_pin), None)
            if found:
                plot_list = [found]
        if not plot_list:
            plot_list = pin_scores[:1]
            
        plot_trajectory_on_pinpad(
            best_centers_ordered, plot_list,
            os.path.join(video_dir, 'trajectory_mapping.png'),
            f'Camera: {camera_position} | Top: {plot_list[0][0] if plot_list else "N/A"}'
        )
    
    return pin_scores, is_same_digit, actual_pin, actual_pin_rank


# ============================================================================
# Main Entry Point
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Camera-aware PIN trajectory analysis")
    
    # Directory arguments
    parser.add_argument("--input-dir", "-i", type=str, required=True,
                        help="Input directory containing CSV trajectory files")
    parser.add_argument("--output-dir", "-o", type=str, required=True,
                        help="Output directory for generated reports")
    
    # Camera position argument
    parser.add_argument("--camera-position", "-c", type=str, 
                        choices=['left', 'right', 'bottom', 'top'],
                        default='left',
                        help="Camera position: left (yaw -90), right (yaw +90), "
                             "bottom (pitch -90), top (pitch +90)")
    
    # Performance arguments
    parser.add_argument("--workers", type=int, default=NUM_WORKERS_DEFAULT,
                        help="Number of parallel workers")
    parser.add_argument("--chunk-size", type=int, default=CHUNK_SIZE_DEFAULT,
                        help="Number of PINs per chunk")
    parser.add_argument("--topk-per-chunk", type=int, default=TOPK_PER_CHUNK_DEFAULT,
                        help="Keep top-K per chunk")
    parser.add_argument("--topk-final", type=int, default=TOPK_FINAL_DEFAULT,
                        help="Keep top-K globally")
    parser.add_argument("--use-pkl-trajectories", action="store_true",
                        help="Use precomputed trajectory PKL")
    
    args = parser.parse_args()

    global NUM_WORKERS, CHUNK_SIZE, TOPK_PER_CHUNK, TOPK_FINAL, USE_PKL_TRAJ
    global OUTPUT_DIR, REPORT_FOLDER, CAMERA_POSITION
    
    OUTPUT_DIR = args.input_dir
    REPORT_FOLDER = args.output_dir
    CAMERA_POSITION = args.camera_position
    
    NUM_WORKERS = max(1, args.workers)
    CHUNK_SIZE = max(10_000, args.chunk_size)
    TOPK_PER_CHUNK = max(1_000, args.topk_per_chunk)
    TOPK_FINAL = max(5_000, args.topk_final)
    USE_PKL_TRAJ = bool(args.use_pkl_trajectories)

    print(f"\n{'='*60}")
    print(f"PIN Trajectory Analysis")
    print(f"{'='*60}")
    print(f"Camera position: {CAMERA_POSITION}")
    print(f"Input directory: {OUTPUT_DIR}")
    print(f"Output directory: {REPORT_FOLDER}")
    print(f"Workers: {NUM_WORKERS}, Chunk size: {CHUNK_SIZE}")
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
        print(f"\n{'='*40}")
        print(f"Processing {idx}/{total}: {os.path.basename(os.path.dirname(csv_path))}")
        print(f"{'='*40}")
        
        video_name = os.path.basename(os.path.dirname(csv_path))
        try:
            pin_scores, is_same_digit, actual_pin, actual_pin_rank = process_csv_trajectory(
                csv_path, report_dir, CAMERA_POSITION
            )
            if pin_scores:
                results[video_name] = pin_scores
                pattern_info[video_name] = is_same_digit
                actual_pins[video_name] = {'pin': actual_pin, 'rank': actual_pin_rank}
                report_filename = generate_individual_html_report(
                    video_name, pin_scores, is_same_digit, actual_pin, 
                    actual_pin_rank, report_dir, CAMERA_POSITION
                )
                video_reports[video_name] = report_filename
        except Exception as e:
            print(f"Error processing {video_name}: {e}")
            import traceback
            traceback.print_exc()
    
    generate_main_html_report(results, pattern_info, report_dir, video_reports, 
                              actual_pins, CAMERA_POSITION)
    
    main_report_path = os.path.join(report_dir, 'index.html')
    print(f"\n{'='*60}")
    print(f"Analysis complete!")
    print(f"Report: {main_report_path}")
    print(f"{'='*60}")
    
    webbrowser.open('file://' + os.path.abspath(main_report_path))


if __name__ == '__main__':
    main()
