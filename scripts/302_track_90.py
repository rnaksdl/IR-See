'''
python 302_track_90.py -i ../2_90/pitch_p90/input_e -o ../2_90/pitch_p90/output_e --camera-position top

Camera positions:
  left (yaw -90): looks along +X axis, Y directly observable
  right (yaw +90): looks along -X axis, Y directly observable  
  bottom (pitch -90): looks along +Y axis, X directly observable
  top (pitch +90): looks along -Y axis, X directly observable

Depth estimation:
  - Larger LED size/spacing = closer to camera
  - Depth mapped to axis along camera's optic axis
  
Outputs:
  - LED size data and plots
  - LED pair distance data and plots
  - Trajectory on PIN pad using LED size for depth
  - Trajectory on PIN pad using LED pair distance (HMM) for depth
'''

import cv2
import numpy as np
import pandas as pd
import os
import csv
import argparse
import itertools
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from scipy.signal import savgol_filter
from hmmlearn import hmm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


# ============================================================================
# PIN Pad Configuration
# ============================================================================

BUTTON_WIDTH = 10.0
BUTTON_HEIGHT = 5.5
GAP = 0.9
X_OFFSET = BUTTON_WIDTH / 2
Y_OFFSET = BUTTON_HEIGHT / 2

# PIN pad layout:
# 1 2 3   (row 0)
# 4 5 6   (row 1)
# 7 8 9   (row 2)
#   0     (row 3)
PINPAD_COORDS = np.array([
    [0*BUTTON_WIDTH + 0*GAP + X_OFFSET, 0*BUTTON_HEIGHT + 0*GAP + Y_OFFSET],  # 1
    [1*BUTTON_WIDTH + 1*GAP + X_OFFSET, 0*BUTTON_HEIGHT + 0*GAP + Y_OFFSET],  # 2
    [2*BUTTON_WIDTH + 2*GAP + X_OFFSET, 0*BUTTON_HEIGHT + 0*GAP + Y_OFFSET],  # 3
    [0*BUTTON_WIDTH + 0*GAP + X_OFFSET, 1*BUTTON_HEIGHT + 1*GAP + Y_OFFSET],  # 4
    [1*BUTTON_WIDTH + 1*GAP + X_OFFSET, 1*BUTTON_HEIGHT + 1*GAP + Y_OFFSET],  # 5
    [2*BUTTON_WIDTH + 2*GAP + X_OFFSET, 1*BUTTON_HEIGHT + 1*GAP + Y_OFFSET],  # 6
    [0*BUTTON_WIDTH + 0*GAP + X_OFFSET, 2*BUTTON_HEIGHT + 2*GAP + Y_OFFSET],  # 7
    [1*BUTTON_WIDTH + 1*GAP + X_OFFSET, 2*BUTTON_HEIGHT + 2*GAP + Y_OFFSET],  # 8
    [2*BUTTON_WIDTH + 2*GAP + X_OFFSET, 2*BUTTON_HEIGHT + 2*GAP + Y_OFFSET],  # 9
    [1*BUTTON_WIDTH + 1*GAP + X_OFFSET, 3*BUTTON_HEIGHT + 3*GAP + Y_OFFSET]   # 0
])

PINPAD_DIGITS = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '0']
PINPAD_DIGIT_TO_IDX = {d: i for i, d in enumerate(PINPAD_DIGITS)}

# Digit to column mapping (for lateral view depth inference)
DIGIT_TO_COLUMN = {
    '1': 0, '4': 0, '7': 0,
    '2': 1, '5': 1, '8': 1, '0': 1,
    '3': 2, '6': 2, '9': 2
}

# Digit to row mapping (for top-down view depth inference)
DIGIT_TO_ROW = {
    '1': 0, '2': 0, '3': 0,
    '4': 1, '5': 1, '6': 1,
    '7': 2, '8': 2, '9': 2,
    '0': 3
}


# ============================================================================
# LED Detection Functions
# ============================================================================

def detect_blue_circles(image):
    """Detect blue/IR LEDs in image, returning (cx, cy, radius) for each."""
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
        if area > 5000:
            continue
            
        M = cv2.moments(c)
        if M["m00"] == 0:
            continue
            
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        equiv_radius = int(np.sqrt(area / np.pi))
        detected.append((cx, cy, equiv_radius))
    
    # Merge nearby detections
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
                    if np.sqrt((x1-x2)**2 + (y1-y2)**2) < 10:
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


def draw_leds(frame, centers, ring_center=None):
    """Draw detected LEDs on the frame with colored circles."""
    frame_copy = frame.copy()
    
    for (x, y, r) in centers:
        cv2.circle(frame_copy, (x, y), r, (0, 255, 0), 2)
        cv2.circle(frame_copy, (x, y), 2, (0, 0, 255), 3)
    
    if ring_center is not None:
        x, y = int(ring_center[0]), int(ring_center[1])
        cv2.circle(frame_copy, (x, y), 8, (255, 0, 0), 2)
        cv2.circle(frame_copy, (x, y), 2, (255, 0, 0), 3)
    
    led_count = len(centers)
    text = f"LEDs: {led_count}"
    
    text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
    cv2.rectangle(frame_copy, 
                 (10, frame_copy.shape[0] - 10 - text_size[1] - 10), 
                 (10 + text_size[0] + 10, frame_copy.shape[0] - 10), 
                 (0, 0, 0), -1)
    
    cv2.putText(frame_copy, text, 
               (15, frame_copy.shape[0] - 15),
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    return frame_copy


# ============================================================================
# Trajectory Processing Functions
# ============================================================================

def interpolate_centers(centers):
    """Interpolate missing center positions."""
    centers = np.array([
        [c[0], c[1]] if c is not None else [np.nan, np.nan]
        for c in centers
    ])
    for i in range(2):
        valid = ~np.isnan(centers[:, i])
        if np.sum(valid) < 2:
            continue
        centers[:, i] = np.interp(
            np.arange(len(centers)),
            np.flatnonzero(valid),
            centers[valid, i]
        )
    return centers


def smooth_centers(centers, window=9, poly=2):
    """Apply Savitzky-Golay smoothing to trajectory."""
    if len(centers) < window:
        return centers
    x = savgol_filter(centers[:, 0], window, poly)
    y = savgol_filter(centers[:, 1], window, poly)
    return np.stack([x, y], axis=1)


def smooth_signal(signal, window=11, poly=2):
    """Apply Savitzky-Golay smoothing to 1D signal."""
    if len(signal) < window:
        return signal
    return savgol_filter(signal, window, poly)


def filter_ambient_led_tracks(led_tracks, min_frames=10, min_disp=20, group_corr_thresh=0.5):
    """Remove LED tracks that are too short, don't move enough, or don't correlate with group."""
    if not led_tracks:
        return []
        
    max_len = max(len(track) for track in led_tracks)
    led_arr = np.full((len(led_tracks), max_len, 2), np.nan)
    for i, track in enumerate(led_tracks):
        for j, pt in enumerate(track):
            if pt is not None:
                led_arr[i, j] = pt

    group_mean = np.nanmean(led_arr, axis=0)

    filtered = []
    for i, track in enumerate(led_tracks):
        pts = np.array([pt for pt in track if pt is not None])
        if len(pts) < min_frames:
            continue
        
        if len(pts) >= 2:
            dists = np.linalg.norm(pts[None, :, :] - pts[:, None, :], axis=2)
            max_disp = np.max(dists)
        else:
            max_disp = 0
            
        if max_disp < min_disp:
            continue
            
        led_traj = led_arr[i]
        group_traj = group_mean
        
        valid = ~np.isnan(led_traj[:, 0])
        if np.sum(valid) < min_frames:
            continue
            
        led_xy = led_traj[valid]
        group_xy = group_traj[valid]
        
        corr_x = np.corrcoef(led_xy[:, 0], group_xy[:, 0])[0, 1] if np.std(led_xy[:, 0]) > 0 and np.std(group_xy[:, 0]) > 0 else 0
        corr_y = np.corrcoef(led_xy[:, 1], group_xy[:, 1])[0, 1] if np.std(led_xy[:, 1]) > 0 and np.std(group_xy[:, 1]) > 0 else 0
        mean_corr = np.nanmean([corr_x, corr_y])
        
        if mean_corr < group_corr_thresh:
            continue
            
        filtered.append(track)
    
    return filtered


# ============================================================================
# LED Size and Distance Data Functions
# ============================================================================

def compute_led_pair_distances_frame(centers):
    """Compute distances between all LED pairs in a single frame."""
    if centers is None or len(centers) < 2:
        return []
    
    pairs = []
    for i in range(len(centers)):
        for j in range(i + 1, len(centers)):
            x1, y1, r1 = centers[i]
            x2, y2, r2 = centers[j]
            dist = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
            pairs.append({
                'led1_idx': i,
                'led2_idx': j,
                'led1_x': x1, 'led1_y': y1,
                'led2_x': x2, 'led2_y': y2,
                'distance': dist
            })
    return pairs


def compute_led_sizes_frame(centers):
    """Extract LED sizes (radii) from detected centers for a single frame."""
    if centers is None:
        return []
    return [{'led_idx': i, 'x': x, 'y': y, 'radius': r} 
            for i, (x, y, r) in enumerate(centers)]


def aggregate_pair_distances(all_frame_pairs):
    """Aggregate LED pair distance data across all frames."""
    records = []
    for frame_idx, pairs in enumerate(all_frame_pairs):
        if not pairs:
            records.append({
                'frame': frame_idx,
                'num_pairs': 0,
                'mean_distance': np.nan,
                'median_distance': np.nan,
                'min_distance': np.nan,
                'max_distance': np.nan,
                'std_distance': np.nan,
                'all_distances': []
            })
        else:
            distances = [p['distance'] for p in pairs]
            records.append({
                'frame': frame_idx,
                'num_pairs': len(distances),
                'mean_distance': np.mean(distances),
                'median_distance': np.median(distances),
                'min_distance': np.min(distances),
                'max_distance': np.max(distances),
                'std_distance': np.std(distances),
                'all_distances': distances
            })
    return pd.DataFrame(records)


def aggregate_led_sizes(all_frame_sizes):
    """Aggregate LED size data across all frames."""
    records = []
    for frame_idx, sizes in enumerate(all_frame_sizes):
        if not sizes:
            records.append({
                'frame': frame_idx,
                'num_leds': 0,
                'mean_radius': np.nan,
                'median_radius': np.nan,
                'min_radius': np.nan,
                'max_radius': np.nan,
                'std_radius': np.nan,
                'all_radii': []
            })
        else:
            radii = [s['radius'] for s in sizes]
            records.append({
                'frame': frame_idx,
                'num_leds': len(radii),
                'mean_radius': np.mean(radii),
                'median_radius': np.median(radii),
                'min_radius': np.min(radii),
                'max_radius': np.max(radii),
                'std_radius': np.std(radii),
                'all_radii': radii
            })
    return pd.DataFrame(records)


def save_detailed_size_data(all_frame_sizes, output_path):
    """Save detailed per-LED size data to CSV."""
    records = []
    for frame_idx, sizes in enumerate(all_frame_sizes):
        for s in sizes:
            records.append({
                'frame': frame_idx,
                'led_idx': s['led_idx'],
                'x': s['x'],
                'y': s['y'],
                'radius': s['radius']
            })
    df = pd.DataFrame(records)
    df.to_csv(output_path, index=False)
    return df


def save_detailed_distance_data(all_frame_pairs, output_path):
    """Save detailed per-pair distance data to CSV."""
    records = []
    for frame_idx, pairs in enumerate(all_frame_pairs):
        for p in pairs:
            records.append({
                'frame': frame_idx,
                'led1_idx': p['led1_idx'],
                'led2_idx': p['led2_idx'],
                'led1_x': p['led1_x'],
                'led1_y': p['led1_y'],
                'led2_x': p['led2_x'],
                'led2_y': p['led2_y'],
                'distance': p['distance']
            })
    df = pd.DataFrame(records)
    df.to_csv(output_path, index=False)
    return df


# ============================================================================
# Depth Estimation Functions
# ============================================================================

def estimate_depth_from_size(size_df):
    """
    Estimate depth proxy from LED sizes only.
    Larger LED size = closer to camera = larger depth proxy value.
    """
    sizes = size_df['mean_radius'].values.copy()
    
    # Normalize to median = 1.0
    median_size = np.nanmedian(sizes)
    if median_size > 0:
        depth_proxy = sizes / median_size
    else:
        depth_proxy = sizes
    
    # Interpolate NaN values
    depth_series = pd.Series(depth_proxy)
    depth_proxy = depth_series.interpolate(method='linear').ffill().bfill().values
    
    return depth_proxy


def estimate_depth_from_distance(distance_df):
    """
    Estimate depth proxy from LED pair distances only.
    Larger distance = closer to camera = larger depth proxy value.
    """
    distances = distance_df['mean_distance'].values.copy()
    
    # Normalize to median = 1.0
    median_dist = np.nanmedian(distances)
    if median_dist > 0:
        depth_proxy = distances / median_dist
    else:
        depth_proxy = distances
    
    # Interpolate NaN values
    depth_series = pd.Series(depth_proxy)
    depth_proxy = depth_series.interpolate(method='linear').ffill().bfill().values
    
    return depth_proxy


def estimate_depth_from_distance_hmm(distance_df, n_components=3):
    """
    Estimate depth using HMM on LED pair distances.
    Uses Gaussian HMM to identify discrete depth states.
    
    Returns:
        depth_proxy: Smoothed depth values
        hmm_states: State assignments per frame
        hmm_model: Fitted HMM model
    """
    distances = distance_df['mean_distance'].values.copy()
    
    # Handle NaN values
    valid_mask = ~np.isnan(distances)
    if np.sum(valid_mask) < n_components * 2:
        # Not enough data for HMM
        return estimate_depth_from_distance(distance_df), np.zeros(len(distances)), None
    
    # Interpolate for HMM fitting
    distances_interp = pd.Series(distances).interpolate(method='linear').ffill().bfill().values
    
    # Prepare data for HMM (needs 2D array)
    X = distances_interp.reshape(-1, 1)
    
    # Fit Gaussian HMM
    best_model = None
    best_score = -np.inf
    
    for n in range(1, n_components + 1):
        try:
            model = hmm.GaussianHMM(
                n_components=n,
                covariance_type='full',
                n_iter=100,
                random_state=42
            )
            model.fit(X)
            score = model.score(X)
            
            # Use BIC-like criterion
            n_params = n * n + n * 2  # Transition matrix + means + variances
            bic = -2 * score + n_params * np.log(len(X))
            
            if best_model is None or bic < best_score:
                best_score = bic
                best_model = model
        except Exception as e:
            continue
    
    if best_model is None:
        return estimate_depth_from_distance(distance_df), np.zeros(len(distances)), None
    
    # Get state sequence
    hmm_states = best_model.predict(X)
    
    # Map states to depth values based on mean distance per state
    state_means = []
    for s in range(best_model.n_components):
        state_mask = hmm_states == s
        if np.any(state_mask):
            state_means.append((s, np.mean(distances_interp[state_mask])))
    
    # Sort states by mean distance (larger = closer)
    state_means.sort(key=lambda x: x[1])
    state_to_depth = {s: i / (len(state_means) - 1) if len(state_means) > 1 else 0.5 
                      for i, (s, _) in enumerate(state_means)}
    
    # Create depth proxy from HMM states
    depth_proxy = np.array([state_to_depth.get(s, 0.5) for s in hmm_states])
    
    # Normalize to have similar scale as size-based depth
    median_dist = np.nanmedian(distances)
    if median_dist > 0:
        depth_proxy = distances_interp / median_dist
    
    return depth_proxy, hmm_states, best_model


def cluster_depth_levels(depth_proxy, max_clusters=3):
    """
    Cluster depth values using GMM with BIC selection.
    """
    valid_mask = ~np.isnan(depth_proxy)
    valid_depth = depth_proxy[valid_mask].reshape(-1, 1)
    
    if len(valid_depth) < max_clusters:
        return {
            'n_clusters': 1,
            'labels': np.zeros(len(depth_proxy), dtype=int),
            'cluster_centers': [np.nanmean(depth_proxy)],
            'cluster_order': [0]
        }
    
    best_bic = np.inf
    best_n = 1
    best_model = None
    
    for n in range(1, max_clusters + 1):
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
    
    labels = np.full(len(depth_proxy), -1, dtype=int)
    if best_model is not None:
        labels[valid_mask] = best_model.predict(valid_depth)
    
    cluster_centers = []
    for c in range(best_n):
        cluster_mask = labels == c
        if np.any(cluster_mask):
            cluster_centers.append(np.mean(depth_proxy[cluster_mask]))
        else:
            cluster_centers.append(np.nan)
    
    cluster_order = np.argsort(cluster_centers)
    
    return {
        'n_clusters': best_n,
        'labels': labels,
        'cluster_centers': cluster_centers,
        'cluster_order': cluster_order.tolist()
    }


# ============================================================================
# 3D Trajectory Functions
# ============================================================================

def compute_3d_trajectory(ring_centers_smooth, depth_proxy_smoothed, camera_position):
    """
    Compute 3D trajectory based on camera position.
    
    Lateral View (left/right):
        - Image Y → World Y (direct)
        - Depth → World X
        - Image X → World Z
        
    Top-Down View (top/bottom):
        - Image X → World X (direct)
        - Depth → World Y
        - Image Y → World Z
    """
    num_frames = len(ring_centers_smooth)
    
    img_x = ring_centers_smooth[:, 0]
    img_y = ring_centers_smooth[:, 1]
    
    img_x_centered = img_x - np.nanmean(img_x)
    img_y_centered = img_y - np.nanmean(img_y)
    
    depth_centered = depth_proxy_smoothed - np.nanmean(depth_proxy_smoothed)
    
    # Scale depth to match image displacement magnitude
    img_scale = max(np.nanstd(img_x_centered), np.nanstd(img_y_centered), 1.0)
    depth_scale = np.nanstd(depth_centered) if np.nanstd(depth_centered) > 0 else 1.0
    depth_scaled = depth_centered * (img_scale / depth_scale)
    
    if camera_position == 'left':
        world_y = -img_y_centered
        world_z = img_x_centered
        world_x = depth_scaled
    elif camera_position == 'right':
        world_y = -img_y_centered
        world_z = -img_x_centered
        world_x = -depth_scaled
    elif camera_position == 'bottom':
        world_x = img_x_centered
        world_z = img_y_centered
        world_y = depth_scaled
    elif camera_position == 'top':
        world_x = img_x_centered
        world_z = -img_y_centered
        world_y = -depth_scaled
    else:
        raise ValueError(f"Unknown camera position: {camera_position}")
    
    return pd.DataFrame({
        'frame': range(num_frames),
        'X': world_x,
        'Y': world_y,
        'Z': world_z
    })


# ============================================================================
# PIN Pad Trajectory Functions
# ============================================================================

def extract_actual_pin_from_filename(video_name):
    """Extract PIN from video filename (assumes filename is the PIN)."""
    # Remove extension and any non-digit characters
    pin = ''.join(c for c in video_name if c.isdigit())
    return pin if len(pin) >= 4 else None


def get_ideal_pin_trajectory(pin):
    """Get ideal trajectory coordinates for a PIN on the PIN pad."""
    if not pin:
        return None
    
    coords = []
    for digit in pin:
        if digit in PINPAD_DIGIT_TO_IDX:
            idx = PINPAD_DIGIT_TO_IDX[digit]
            coords.append(PINPAD_COORDS[idx])
    
    return np.array(coords) if coords else None


def fit_translation_scaling(A, B):
    """
    Fit trajectory A to trajectory B using translation and scaling.
    Returns error and transformed A.
    """
    if len(A) < 2 or len(B) < 2:
        return float('inf'), None
    
    # Match lengths
    min_len = min(len(A), len(B))
    A = A[:min_len]
    B = B[:min_len]
    
    centroid_A = np.mean(A, axis=0)
    centroid_B = np.mean(B, axis=0)
    
    AA = A - centroid_A
    BB = B - centroid_B
    
    norm_A = np.linalg.norm(AA)
    norm_B = np.linalg.norm(BB)
    
    scale = 1.0 if (norm_A == 0 or norm_B == 0) else norm_B / norm_A
    
    A_transformed = AA * scale + centroid_B
    
    error = np.sqrt(np.mean(np.sum((A_transformed - B)**2, axis=1)))
    
    return error, A_transformed


def map_trajectory_to_pinpad(trajectory_3d_df, camera_position):
    """
    Map 3D trajectory to 2D PIN pad coordinates based on camera position.
    
    For lateral (left/right): Use X (depth) and Y (direct) → PIN pad X and Y
    For top-down (top/bottom): Use X (direct) and Y (depth) → PIN pad X and Y
    """
    if camera_position in ['left', 'right']:
        # Depth (X) maps to PIN pad X (columns), direct Y maps to PIN pad Y (rows)
        pinpad_x = trajectory_3d_df['X'].values
        pinpad_y = trajectory_3d_df['Y'].values
    else:  # top, bottom
        # Direct X maps to PIN pad X (columns), Depth (Y) maps to PIN pad Y (rows)
        pinpad_x = trajectory_3d_df['X'].values
        pinpad_y = trajectory_3d_df['Y'].values
    
    # Normalize to PIN pad coordinate range
    x_range = PINPAD_COORDS[:, 0].max() - PINPAD_COORDS[:, 0].min()
    y_range = PINPAD_COORDS[:, 1].max() - PINPAD_COORDS[:, 1].min()
    
    # Center and scale
    pinpad_x_norm = (pinpad_x - np.nanmean(pinpad_x))
    pinpad_y_norm = (pinpad_y - np.nanmean(pinpad_y))
    
    # Scale to fit PIN pad
    x_scale = x_range / (2 * np.nanstd(pinpad_x_norm)) if np.nanstd(pinpad_x_norm) > 0 else 1
    y_scale = y_range / (2 * np.nanstd(pinpad_y_norm)) if np.nanstd(pinpad_y_norm) > 0 else 1
    
    scale = min(x_scale, y_scale)
    
    pinpad_x_scaled = pinpad_x_norm * scale + np.mean(PINPAD_COORDS[:, 0])
    pinpad_y_scaled = pinpad_y_norm * scale + np.mean(PINPAD_COORDS[:, 1])
    
    return np.stack([pinpad_x_scaled, pinpad_y_scaled], axis=1)


def detect_keypress_moments(trajectory_2d, min_pause_frames=5, speed_percentile=30):
    """
    Detect keypress moments based on trajectory speed.
    Returns indices of likely keypress frames.
    """
    if len(trajectory_2d) < 3:
        return []
    
    # Calculate speed
    dx = np.diff(trajectory_2d[:, 0])
    dy = np.diff(trajectory_2d[:, 1])
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
                mid_frame = pause_start + pause_length // 2
                keypresses.append({
                    'frame': mid_frame,
                    'x': np.mean(trajectory_2d[pause_start:i, 0]),
                    'y': np.mean(trajectory_2d[pause_start:i, 1])
                })
    
    # Handle end of trajectory
    if in_pause:
        pause_length = len(slow_mask) - pause_start
        if pause_length >= min_pause_frames:
            mid_frame = pause_start + pause_length // 2
            keypresses.append({
                'frame': mid_frame,
                'x': np.mean(trajectory_2d[pause_start:, 0]),
                'y': np.mean(trajectory_2d[pause_start:, 1])
            })
    
    return keypresses


# ============================================================================
# Plotting Functions
# ============================================================================

def plot_led_sizes(size_df, output_path, video_name):
    """Plot LED size statistics over time."""
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    
    ax1 = axes[0]
    frames = size_df['frame'].values
    
    ax1.fill_between(frames, size_df['min_radius'], size_df['max_radius'],
                     alpha=0.3, color='orange', label='Min-Max Range')
    ax1.plot(frames, size_df['mean_radius'], 'orange', linewidth=2, label='Mean Radius')
    ax1.plot(frames, size_df['median_radius'], 'red', linestyle='--', linewidth=1.5, 
             label='Median Radius')
    
    ax1.set_ylabel('Radius (pixels)')
    ax1.set_title(f'{video_name} - LED Sizes Over Time\n(Larger = Closer to camera)')
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)
    
    ax2 = axes[1]
    ax2.bar(frames, size_df['num_leds'], color='purple', alpha=0.7)
    ax2.set_xlabel('Frame')
    ax2.set_ylabel('Number of LEDs')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_led_distances(distance_df, output_path, video_name):
    """Plot LED pair distance statistics over time."""
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    
    ax1 = axes[0]
    frames = distance_df['frame'].values
    
    ax1.fill_between(frames, distance_df['min_distance'], distance_df['max_distance'],
                     alpha=0.3, color='blue', label='Min-Max Range')
    ax1.plot(frames, distance_df['mean_distance'], 'b-', linewidth=2, label='Mean Distance')
    ax1.plot(frames, distance_df['median_distance'], 'r--', linewidth=1.5, label='Median Distance')
    
    ax1.set_ylabel('Distance (pixels)')
    ax1.set_title(f'{video_name} - LED Pair Distances Over Time\n(Larger = Closer to camera)')
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)
    
    ax2 = axes[1]
    ax2.bar(frames, distance_df['num_pairs'], color='green', alpha=0.7)
    ax2.set_xlabel('Frame')
    ax2.set_ylabel('Number of LED Pairs')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_depth_comparison(size_depth, distance_depth, hmm_states, output_path, video_name):
    """Plot comparison of depth estimation methods."""
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
    
    frames = np.arange(len(size_depth))
    
    # Size-based depth
    ax1 = axes[0]
    ax1.plot(frames, size_depth, 'orange', linewidth=2, label='Size-based Depth')
    ax1.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5)
    ax1.set_ylabel('Depth Proxy\n(Larger = Closer)')
    ax1.set_title(f'{video_name} - Depth from LED Size')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Distance-based depth
    ax2 = axes[1]
    ax2.plot(frames, distance_depth, 'blue', linewidth=2, label='Distance-based Depth')
    ax2.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5)
    ax2.set_ylabel('Depth Proxy\n(Larger = Closer)')
    ax2.set_title(f'{video_name} - Depth from LED Pair Distance')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # HMM states
    ax3 = axes[2]
    if hmm_states is not None:
        n_states = len(np.unique(hmm_states))
        colors = plt.cm.viridis(np.linspace(0.2, 0.8, n_states))
        for s in range(n_states):
            mask = hmm_states == s
            ax3.scatter(frames[mask], distance_depth[mask], c=[colors[s]], 
                       s=10, alpha=0.7, label=f'HMM State {s}')
    ax3.set_xlabel('Frame')
    ax3.set_ylabel('Depth with HMM States')
    ax3.set_title(f'{video_name} - HMM State Segmentation')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_trajectory_on_pinpad(observed_trajectory, actual_pin, title, output_path, 
                               keypress_points=None):
    """
    Plot trajectory on PIN pad with actual PIN for comparison.
    
    Args:
        observed_trajectory: Nx2 array of observed trajectory points
        actual_pin: String of actual PIN (e.g., "1234")
        title: Plot title
        output_path: Output file path
        keypress_points: Optional list of detected keypress locations
    """
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Draw PIN pad buttons
    for i, (x, y) in enumerate(PINPAD_COORDS):
        # Button background
        rect = plt.Rectangle((x - BUTTON_WIDTH/2.5, y - BUTTON_HEIGHT/2.5), 
                             BUTTON_WIDTH/1.25, BUTTON_HEIGHT/1.25,
                             fill=True, facecolor='lightgray', edgecolor='black', linewidth=2)
        ax.add_patch(rect)
        # Button label
        ax.text(x, y, PINPAD_DIGITS[i], fontsize=20, ha='center', va='center', 
                fontweight='bold')
    
    # Draw actual PIN trajectory (ideal)
    if actual_pin:
        ideal_coords = get_ideal_pin_trajectory(actual_pin)
        if ideal_coords is not None and len(ideal_coords) > 0:
            ax.plot(ideal_coords[:, 0], ideal_coords[:, 1], 'b-', 
                   linewidth=3, alpha=0.7, label=f'Actual PIN: {actual_pin}', zorder=5)
            ax.scatter(ideal_coords[:, 0], ideal_coords[:, 1], 
                      c='blue', s=150, edgecolor='white', linewidth=2, zorder=6)
            # Number the keypress order
            for i, (x, y) in enumerate(ideal_coords):
                ax.annotate(str(i+1), (x, y), xytext=(8, 8), textcoords='offset points',
                           fontsize=12, fontweight='bold', color='blue')
    
    # Draw observed trajectory
    if observed_trajectory is not None and len(observed_trajectory) > 0:
        # Full trajectory (faded)
        ax.plot(observed_trajectory[:, 0], observed_trajectory[:, 1], 
               'gray', linewidth=1, alpha=0.3, zorder=3)
        
        # Fit observed to ideal and plot
        if actual_pin:
            ideal_coords = get_ideal_pin_trajectory(actual_pin)
            if ideal_coords is not None and keypress_points is not None and len(keypress_points) > 0:
                # Use keypress points for comparison
                kp_coords = np.array([[kp['x'], kp['y']] for kp in keypress_points])
                if len(kp_coords) >= len(ideal_coords):
                    kp_coords = kp_coords[:len(ideal_coords)]
                error, transformed = fit_translation_scaling(kp_coords, ideal_coords)
                
                if transformed is not None:
                    ax.plot(transformed[:, 0], transformed[:, 1], 'r--', 
                           linewidth=2, alpha=0.8, label=f'Observed (fitted, error={error:.2f})', zorder=7)
                    ax.scatter(transformed[:, 0], transformed[:, 1], 
                              c='red', s=100, marker='x', linewidth=2, zorder=8)
        
        # Plot detected keypress points
        if keypress_points is not None:
            kp_coords = np.array([[kp['x'], kp['y']] for kp in keypress_points])
            ax.scatter(kp_coords[:, 0], kp_coords[:, 1], 
                      c='green', s=80, marker='o', alpha=0.7, 
                      label='Detected Keypresses', zorder=4)
    
    ax.set_xlim(-5, PINPAD_COORDS[:, 0].max() + 10)
    ax.set_ylim(-5, PINPAD_COORDS[:, 1].max() + 10)
    ax.set_aspect('equal')
    ax.invert_yaxis()  # PIN pad has row 0 at top
    ax.set_xlabel('X (PIN pad columns)')
    ax.set_ylabel('Y (PIN pad rows)')
    ax.set_title(title)
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_3d_trajectory(trajectory_df, output_path, video_name, camera_position):
    """Plot 3D trajectory with projections."""
    fig = plt.figure(figsize=(16, 12))
    
    X = trajectory_df['X'].values
    Y = trajectory_df['Y'].values
    Z = trajectory_df['Z'].values
    frames = trajectory_df['frame'].values
    
    # 3D plot
    ax1 = fig.add_subplot(2, 2, 1, projection='3d')
    ax1.scatter(X, Z, Y, c=frames, cmap='viridis', s=10, alpha=0.7)
    ax1.plot(X, Z, Y, 'gray', alpha=0.3, linewidth=0.5)
    ax1.scatter([X[0]], [Z[0]], [Y[0]], c='green', s=100, marker='o', label='Start')
    ax1.scatter([X[-1]], [Z[-1]], [Y[-1]], c='red', s=100, marker='s', label='End')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Z')
    ax1.set_zlabel('Y')
    ax1.set_title(f'3D Trajectory (Camera: {camera_position})')
    ax1.legend()
    
    # Top view (X-Z)
    ax2 = fig.add_subplot(2, 2, 2)
    ax2.scatter(X, Z, c=frames, cmap='viridis', s=10, alpha=0.7)
    ax2.plot(X, Z, 'gray', alpha=0.3)
    ax2.scatter([X[0]], [Z[0]], c='green', s=100, marker='o')
    ax2.scatter([X[-1]], [Z[-1]], c='red', s=100, marker='s')
    ax2.set_xlabel('X')
    ax2.set_ylabel('Z')
    ax2.set_title('Top View (X-Z)')
    ax2.grid(True, alpha=0.3)
    
    # Front view (X-Y)
    ax3 = fig.add_subplot(2, 2, 3)
    ax3.scatter(X, Y, c=frames, cmap='viridis', s=10, alpha=0.7)
    ax3.plot(X, Y, 'gray', alpha=0.3)
    ax3.scatter([X[0]], [Y[0]], c='green', s=100, marker='o')
    ax3.scatter([X[-1]], [Y[-1]], c='red', s=100, marker='s')
    ax3.set_xlabel('X')
    ax3.set_ylabel('Y')
    ax3.set_title('Front View (X-Y)')
    ax3.grid(True, alpha=0.3)
    
    # Side view (Z-Y)
    ax4 = fig.add_subplot(2, 2, 4)
    ax4.scatter(Z, Y, c=frames, cmap='viridis', s=10, alpha=0.7)
    ax4.plot(Z, Y, 'gray', alpha=0.3)
    ax4.scatter([Z[0]], [Y[0]], c='green', s=100, marker='o')
    ax4.scatter([Z[-1]], [Y[-1]], c='red', s=100, marker='s')
    ax4.set_xlabel('Z')
    ax4.set_ylabel('Y')
    ax4.set_title('Side View (Z-Y)')
    ax4.grid(True, alpha=0.3)
    
    plt.suptitle(f'{video_name} - 3D Trajectory', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


# ============================================================================
# Main Processing Function
# ============================================================================

def process_video(video_path, output_folder, video_idx, total_videos, camera_position='left'):
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    print(f"\n{'='*60}")
    print(f"Processing video {video_idx}/{total_videos}: {video_name}")
    print(f"Camera position: {camera_position}")
    print(f"{'='*60}")
    
    # Extract actual PIN from filename
    actual_pin = extract_actual_pin_from_filename(video_name)
    if actual_pin:
        print(f"Actual PIN from filename: {actual_pin}")
    
    # Create output directories
    video_out_folder = os.path.join(output_folder, video_name)
    frames_folder = os.path.join(video_out_folder, 'frames')
    filtered_frames_folder = os.path.join(video_out_folder, 'filtered_frames')
    plots_folder = os.path.join(video_out_folder, 'plots')
    os.makedirs(frames_folder, exist_ok=True)
    os.makedirs(filtered_frames_folder, exist_ok=True)
    os.makedirs(plots_folder, exist_ok=True)

    # Open video
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Storage
    led_tracks = []
    ring_centers = []
    original_frames = []
    detected_centers = []
    all_frame_pair_distances = []
    all_frame_led_sizes = []
    
    frame_idx = 0

    # ========================================================================
    # Stage 1: Frame-by-frame LED detection and tracking
    # ========================================================================
    print("\n[Stage 1] Processing frames and tracking LEDs...")
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        original_frames.append(frame.copy())
        centers = detect_blue_circles(frame)

        if centers is None:
            centers = []
        
        detected_centers.append(centers)
        
        # Compute LED data for this frame
        pair_distances = compute_led_pair_distances_frame(centers)
        led_sizes = compute_led_sizes_frame(centers)
        all_frame_pair_distances.append(pair_distances)
        all_frame_led_sizes.append(led_sizes)
        
        centers_xy = [(x, y) for (x, y, r) in centers]

        # Simple nearest-neighbor tracking
        if frame_idx == 0:
            for c in centers_xy:
                led_tracks.append([c])
        else:
            prev_centers = [track[-1] if len(track) > 0 else None for track in led_tracks]
            assignments = [-1] * len(centers_xy)
            used = set()
            for i, pc in enumerate(prev_centers):
                if pc is None:
                    continue
                min_dist = float('inf')
                min_j = -1
                for j, cc in enumerate(centers_xy):
                    if j in used:
                        continue
                    dist = np.linalg.norm(np.array(pc) - np.array(cc))
                    if dist < min_dist and dist < 30:
                        min_dist = dist
                        min_j = j
                if min_j != -1:
                    assignments[min_j] = i
                    used.add(min_j)
            assigned = [False] * len(centers_xy)
            for i, track in enumerate(led_tracks):
                found = False
                for j, assign in enumerate(assignments):
                    if assign == i:
                        track.append(centers_xy[j])
                        assigned[j] = True
                        found = True
                        break
                if not found:
                    track.append(None)
            for j, was_assigned in enumerate(assigned):
                if not was_assigned:
                    new_track = [None] * frame_idx
                    new_track.append(centers_xy[j])
                    led_tracks.append(new_track)
        
        # Ring center
        if len(centers_xy) >= 1:
            xs = np.array([pt[0] for pt in centers_xy])
            ys = np.array([pt[1] for pt in centers_xy])
            ring_centers.append((np.mean(xs), np.mean(ys)))
        else:
            ring_centers.append(None)

        # Save frame with detections
        frame_draw = draw_leds(frame, centers, ring_center=ring_centers[-1])
        cv2.imwrite(os.path.join(frames_folder, f'frame_{frame_idx:05d}.png'), frame_draw)

        if frame_idx % 50 == 0 or frame_idx == total_frames - 1:
            print(f"  Frame {frame_idx+1}/{total_frames}")
        frame_idx += 1

    cap.release()
    print(f"  Total frames processed: {frame_idx}")

    # ========================================================================
    # Stage 2: Aggregate and save LED size data
    # ========================================================================
    print("\n[Stage 2] Aggregating LED size data...")
    size_df = aggregate_led_sizes(all_frame_led_sizes)
    size_csv_path = os.path.join(video_out_folder, f'{video_name}_led_sizes.csv')
    size_df.to_csv(size_csv_path, index=False)
    print(f"  Saved: {size_csv_path}")
    
    # Save detailed per-LED size data
    detailed_size_path = os.path.join(video_out_folder, f'{video_name}_led_sizes_detailed.csv')
    save_detailed_size_data(all_frame_led_sizes, detailed_size_path)
    print(f"  Saved: {detailed_size_path}")
    
    # Plot LED sizes
    size_plot_path = os.path.join(plots_folder, f'{video_name}_led_sizes.png')
    plot_led_sizes(size_df, size_plot_path, video_name)
    print(f"  Saved: {size_plot_path}")

    # ========================================================================
    # Stage 3: Aggregate and save LED pair distance data
    # ========================================================================
    print("\n[Stage 3] Aggregating LED pair distance data...")
    distance_df = aggregate_pair_distances(all_frame_pair_distances)
    distance_csv_path = os.path.join(video_out_folder, f'{video_name}_led_pair_distances.csv')
    distance_df.to_csv(distance_csv_path, index=False)
    print(f"  Saved: {distance_csv_path}")
    
    # Save detailed per-pair distance data
    detailed_distance_path = os.path.join(video_out_folder, f'{video_name}_led_distances_detailed.csv')
    save_detailed_distance_data(all_frame_pair_distances, detailed_distance_path)
    print(f"  Saved: {detailed_distance_path}")
    
    # Plot LED distances
    distance_plot_path = os.path.join(plots_folder, f'{video_name}_led_pair_distances.png')
    plot_led_distances(distance_df, distance_plot_path, video_name)
    print(f"  Saved: {distance_plot_path}")

    # ========================================================================
    # Stage 4: Compute depth estimations
    # ========================================================================
    print("\n[Stage 4] Computing depth estimations...")
    
    # Method 1: Depth from LED size
    depth_from_size = estimate_depth_from_size(size_df)
    depth_from_size_smooth = smooth_signal(depth_from_size, window=11, poly=2)
    print(f"  Size-based depth: min={np.nanmin(depth_from_size_smooth):.3f}, max={np.nanmax(depth_from_size_smooth):.3f}")
    
    # Method 2: Depth from LED pair distance
    depth_from_distance = estimate_depth_from_distance(distance_df)
    depth_from_distance_smooth = smooth_signal(depth_from_distance, window=11, poly=2)
    print(f"  Distance-based depth: min={np.nanmin(depth_from_distance_smooth):.3f}, max={np.nanmax(depth_from_distance_smooth):.3f}")
    
    # Method 3: Depth from distance with HMM
    depth_from_hmm, hmm_states, hmm_model = estimate_depth_from_distance_hmm(distance_df, n_components=3)
    depth_from_hmm_smooth = smooth_signal(depth_from_hmm, window=11, poly=2)
    n_hmm_states = len(np.unique(hmm_states)) if hmm_states is not None else 0
    print(f"  HMM-based depth: {n_hmm_states} states detected")
    
    # Save depth data
    depth_df = pd.DataFrame({
        'frame': range(frame_idx),
        'depth_from_size': depth_from_size,
        'depth_from_size_smooth': depth_from_size_smooth,
        'depth_from_distance': depth_from_distance,
        'depth_from_distance_smooth': depth_from_distance_smooth,
        'depth_from_hmm': depth_from_hmm,
        'depth_from_hmm_smooth': depth_from_hmm_smooth,
        'hmm_state': hmm_states if hmm_states is not None else -1
    })
    depth_csv_path = os.path.join(video_out_folder, f'{video_name}_depth_estimation.csv')
    depth_df.to_csv(depth_csv_path, index=False)
    print(f"  Saved: {depth_csv_path}")
    
    # Plot depth comparison
    depth_plot_path = os.path.join(plots_folder, f'{video_name}_depth_comparison.png')
    plot_depth_comparison(depth_from_size_smooth, depth_from_distance_smooth, 
                          hmm_states, depth_plot_path, video_name)
    print(f"  Saved: {depth_plot_path}")

    # ========================================================================
    # Stage 5: Filter LED tracks and compute ring centers
    # ========================================================================
    print("\n[Stage 5] Filtering LED tracks...")
    filtered_led_tracks = filter_ambient_led_tracks(led_tracks, min_frames=10, 
                                                     min_disp=20, group_corr_thresh=0.5)
    print(f"  Filtered: {len(led_tracks)} → {len(filtered_led_tracks)} tracks")
    
    # Compute smoothed ring center trajectory
    centers_interp = interpolate_centers(ring_centers)
    centers_smooth = smooth_centers(centers_interp, window=9, poly=2)
    
    # Save ring center data
    ring_csv_path = os.path.join(video_out_folder, f'{video_name}_ring_center.csv')
    with open(ring_csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['frame', 'ring_x', 'ring_y'])
        for i, (x, y) in enumerate(centers_smooth):
            writer.writerow([i, x, y])
    print(f"  Saved: {ring_csv_path}")

    # ========================================================================
    # Stage 6: Compute 3D trajectories using different depth methods
    # ========================================================================
    print("\n[Stage 6] Computing 3D trajectories...")
    
    # Trajectory using LED size for depth
    traj_size_df = compute_3d_trajectory(centers_smooth, depth_from_size_smooth, camera_position)
    traj_size_df['depth_method'] = 'size'
    traj_size_csv = os.path.join(video_out_folder, f'{video_name}_trajectory_size.csv')
    traj_size_df.to_csv(traj_size_csv, index=False)
    print(f"  Saved: {traj_size_csv}")
    
    # Trajectory using LED pair distance for depth
    traj_dist_df = compute_3d_trajectory(centers_smooth, depth_from_distance_smooth, camera_position)
    traj_dist_df['depth_method'] = 'distance'
    traj_dist_csv = os.path.join(video_out_folder, f'{video_name}_trajectory_distance.csv')
    traj_dist_df.to_csv(traj_dist_csv, index=False)
    print(f"  Saved: {traj_dist_csv}")
    
    # Trajectory using HMM-based depth
    traj_hmm_df = compute_3d_trajectory(centers_smooth, depth_from_hmm_smooth, camera_position)
    traj_hmm_df['depth_method'] = 'hmm'
    traj_hmm_df['hmm_state'] = hmm_states if hmm_states is not None else -1
    traj_hmm_csv = os.path.join(video_out_folder, f'{video_name}_trajectory_hmm.csv')
    traj_hmm_df.to_csv(traj_hmm_csv, index=False)
    print(f"  Saved: {traj_hmm_csv}")
    
    # Plot 3D trajectories
    plot_3d_trajectory(traj_size_df, os.path.join(plots_folder, f'{video_name}_trajectory_3d_size.png'),
                       video_name + ' (Size)', camera_position)
    plot_3d_trajectory(traj_hmm_df, os.path.join(plots_folder, f'{video_name}_trajectory_3d_hmm.png'),
                       video_name + ' (HMM)', camera_position)

    # ========================================================================
    # Stage 7: Map trajectories to PIN pad and compare with actual PIN
    # ========================================================================
    print("\n[Stage 7] Mapping trajectories to PIN pad...")
    
    # Map size-based trajectory to PIN pad
    pinpad_traj_size = map_trajectory_to_pinpad(traj_size_df, camera_position)
    keypresses_size = detect_keypress_moments(pinpad_traj_size)
    print(f"  Size-based: detected {len(keypresses_size)} keypresses")
    
    # Map HMM-based trajectory to PIN pad
    pinpad_traj_hmm = map_trajectory_to_pinpad(traj_hmm_df, camera_position)
    keypresses_hmm = detect_keypress_moments(pinpad_traj_hmm)
    print(f"  HMM-based: detected {len(keypresses_hmm)} keypresses")
    
    # Plot trajectory on PIN pad using LED SIZE for depth
    pinpad_size_plot = os.path.join(plots_folder, f'{video_name}_pinpad_trajectory_size.png')
    plot_trajectory_on_pinpad(
        pinpad_traj_size, 
        actual_pin,
        f'{video_name}\nTrajectory using LED SIZE for depth\n(Camera: {camera_position})',
        pinpad_size_plot,
        keypresses_size
    )
    print(f"  Saved: {pinpad_size_plot}")
    
    # Plot trajectory on PIN pad using LED PAIR DISTANCE (HMM) for depth
    pinpad_hmm_plot = os.path.join(plots_folder, f'{video_name}_pinpad_trajectory_hmm.png')
    plot_trajectory_on_pinpad(
        pinpad_traj_hmm,
        actual_pin,
        f'{video_name}\nTrajectory using LED PAIR DISTANCE (HMM) for depth\n(Camera: {camera_position})',
        pinpad_hmm_plot,
        keypresses_hmm
    )
    print(f"  Saved: {pinpad_hmm_plot}")

    # ========================================================================
    # Stage 8: Save filtered frames
    # ========================================================================
    print("\n[Stage 8] Saving filtered frames...")
    filtered_ring_centers = []
    
    for f in range(frame_idx):
        frame = original_frames[f]
        
        filtered_centers = []
        for track in filtered_led_tracks:
            if f < len(track) and track[f] is not None:
                for cx, cy, r in detected_centers[f]:
                    if abs(cx - track[f][0]) < 2 and abs(cy - track[f][1]) < 2:
                        filtered_centers.append((cx, cy, r))
                        break
        
        if len(filtered_centers) >= 1:
            xs = np.array([x for (x, y, r) in filtered_centers])
            ys = np.array([y for (x, y, r) in filtered_centers])
            filtered_ring_centers.append((np.mean(xs), np.mean(ys)))
        else:
            filtered_ring_centers.append(None)
            
        frame_draw = draw_leds(frame, filtered_centers, ring_center=filtered_ring_centers[-1])
        cv2.imwrite(os.path.join(filtered_frames_folder, f'frame_{f:05d}.png'), frame_draw)

    # Save filtered ring center
    if filtered_ring_centers:
        centers_interp = interpolate_centers(filtered_ring_centers)
        centers_smooth = smooth_centers(centers_interp, window=9, poly=2)
        
        filtered_csv = os.path.join(video_out_folder, f'{video_name}_filtered_ring_center.csv')
        with open(filtered_csv, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['frame', 'ring_x', 'ring_y'])
            for i, (x, y) in enumerate(centers_smooth):
                writer.writerow([i, x, y])
        print(f"  Saved: {filtered_csv}")

    print(f"\n{'='*60}")
    print(f"Completed processing: {video_name}")
    print(f"{'='*60}")


# ============================================================================
# Main Entry Point
# ============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Track VR controller LEDs and estimate 3D trajectory for PIN pad analysis',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python track_90.py -i ../yaw_n90/input -o ../yaw_n90/output -c left
  python track_90.py -i ../yaw_p90/input -o ../yaw_p90/output -c right
  python track_90.py -i ../pitch_n90/input -o ../pitch_n90/output -c bottom
  python track_90.py -i ../pitch_p90/input -o ../pitch_p90/output -c top
        """
    )
    parser.add_argument('--input', '-i', required=True, 
                        help='Input directory containing video files')
    parser.add_argument('--output', '-o', required=True, 
                        help='Output directory for results')
    parser.add_argument('--camera-position', '-c', 
                        choices=['left', 'right', 'bottom', 'top'],
                        default='left',
                        help='Camera position: left (yaw -90), right (yaw +90), '
                             'bottom (pitch -90), top (pitch +90)')
    
    args = parser.parse_args()
    
    input_folder = args.input
    output_folder = args.output
    camera_position = args.camera_position
    
    print(f"\n{'#'*60}")
    print(f"# VR Controller LED Tracking with 3D Trajectory Estimation")
    print(f"{'#'*60}")
    print(f"Input: {input_folder}")
    print(f"Output: {output_folder}")
    print(f"Camera position: {camera_position}")
    
    os.makedirs(output_folder, exist_ok=True)
    
    video_files = [fname for fname in os.listdir(input_folder) 
                   if fname.lower().endswith(('.mp4', '.avi', '.mov'))]
    total_videos = len(video_files)
    
    print(f"Found {total_videos} video(s) to process")
    
    for idx, fname in enumerate(video_files, 1):
        video_path = os.path.join(input_folder, fname)
        process_video(video_path, output_folder, idx, total_videos, camera_position)
    
    print(f"\n{'#'*60}")
    print(f"# All videos processed!")
    print(f"{'#'*60}")
