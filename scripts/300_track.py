'''
python 300_track.py -i ../0_reg/4digit_100/input_e -o ../0_reg/4digit_100/output_e
'''


import cv2
import numpy as np
import os
import csv
import argparse
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from scipy.signal import savgol_filter


def detect_blue_circles(image):
    # Extract blue channel directly - blue LEDs will be brighter here
    b, g, r = cv2.split(image)
    
    # Apply stronger blur to help with faint spots
    blurred = cv2.GaussianBlur(b, (11, 11), 0)
    
    # Use a very low threshold to catch faint spots
    _, thresh = cv2.threshold(blurred, 25, 255, cv2.THRESH_BINARY)
    
    # Use larger kernel for morphological operations
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask_opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
    mask_closing = cv2.morphologyEx(mask_opening, cv2.MORPH_CLOSE, kernel, iterations=1)
    
    # Find contours in the mask
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
    """Draw detected LEDs on the frame with colored circles"""
    # Create a copy of the frame to avoid modifying the original
    frame_copy = frame.copy()
    
    # Draw each LED with green circle and red center
    for (x, y, r) in centers:
        cv2.circle(frame_copy, (x, y), r, (0, 255, 0), 2)
        cv2.circle(frame_copy, (x, y), 2, (0, 0, 255), 3)
    
    # Draw the ring center if available
    if ring_center is not None:
        x, y = int(ring_center[0]), int(ring_center[1])
        cv2.circle(frame_copy, (x, y), 8, (255, 0, 0), 2)
        cv2.circle(frame_copy, (x, y), 2, (255, 0, 0), 3)
    
    # Add LED count in bottom left corner
    led_count = len(centers)
    text = f"LEDs: {led_count}"
    
    # Create a semi-transparent background for text
    text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
    cv2.rectangle(frame_copy, 
                 (10, frame_copy.shape[0] - 10 - text_size[1] - 10), 
                 (10 + text_size[0] + 10, frame_copy.shape[0] - 10), 
                 (0, 0, 0), -1)
    
    # Add text with LED count
    cv2.putText(frame_copy, text, 
               (15, frame_copy.shape[0] - 15),
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    return frame_copy

def fit_circle(xs, ys):
    if len(xs) < 3:
        return None
    A = np.c_[2*xs, 2*ys, np.ones(len(xs))]
    b = xs**2 + ys**2
    c, resid, rank, s = np.linalg.lstsq(A, b, rcond=None)
    xc, yc = c[0], c[1]
    return (xc, yc)

def interpolate_centers(centers):
    centers = np.array([
        [c[0], c[1]] if c is not None else [np.nan, np.nan]
        for c in centers
    ])
    for i in range(2):  # x and y
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
    if len(centers) < window:
        return centers
    x = savgol_filter(centers[:,0], window, poly)
    y = savgol_filter(centers[:,1], window, poly)
    return np.stack([x, y], axis=1)

def filter_ambient_led_tracks(led_tracks, min_frames=10, min_disp=20, group_corr_thresh=0.5):
    """
    Remove LED tracks that are either too short, do not move enough, or do not move with the group.
    - min_frames: minimum number of frames the LED must be tracked
    - min_disp: minimum spatial displacement (in pixels) required to keep the track
    - group_corr_thresh: minimum correlation with group motion to keep the track
    """
    # Build array: (num_leds, num_frames, 2)
    max_len = max(len(track) for track in led_tracks)
    led_arr = np.full((len(led_tracks), max_len, 2), np.nan)
    for i, track in enumerate(led_tracks):
        for j, pt in enumerate(track):
            if pt is not None:
                led_arr[i, j] = pt

    # Compute group mean trajectory (ignore NaNs)
    group_mean = np.nanmean(led_arr, axis=0)  # shape: (num_frames, 2)

    filtered = []
    for i, track in enumerate(led_tracks):
        pts = np.array([pt for pt in track if pt is not None])
        if len(pts) < min_frames:
            continue
        
        # Max displacement check
        if len(pts) >= 2:
            dists = np.linalg.norm(pts[None, :, :] - pts[:, None, :], axis=2)
            max_disp = np.max(dists)
        else:
            max_disp = 0
            
        if max_disp < min_disp:
            continue
            
        # Correlation with group mean
        led_traj = led_arr[i]
        group_traj = group_mean
        
        # Only compare frames where this LED is present
        valid = ~np.isnan(led_traj[:,0])
        if np.sum(valid) < min_frames:
            continue
            
        led_xy = led_traj[valid]
        group_xy = group_traj[valid]
        
        # Compute correlation of x and y separately, then average
        corr_x = np.corrcoef(led_xy[:,0], group_xy[:,0])[0,1] if np.std(led_xy[:,0]) > 0 and np.std(group_xy[:,0]) > 0 else 0
        corr_y = np.corrcoef(led_xy[:,1], group_xy[:,1])[0,1] if np.std(led_xy[:,1]) > 0 and np.std(group_xy[:,1]) > 0 else 0
        mean_corr = np.nanmean([corr_x, corr_y])
        
        if mean_corr < group_corr_thresh:
            continue
            
        filtered.append(track)
    
    return filtered

def process_video(video_path, output_folder, video_idx, total_videos):
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    print(f"Processing video {video_idx}/{total_videos}: {video_path} ...")
    video_out_folder = os.path.join(output_folder, video_name)
    frames_folder = os.path.join(video_out_folder, 'frames')
    filtered_frames_folder = os.path.join(video_out_folder, 'filtered_frames')
    os.makedirs(frames_folder, exist_ok=True)
    os.makedirs(filtered_frames_folder, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    led_tracks = []
    ring_centers = []
    original_frames = []  # Store original frames for later reprocessing
    detected_centers = []  # Store all detected centers for each frame
    frame_idx = 0

    print("  Processing frames and tracking LEDs...")
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Store original frame
        original_frames.append(frame.copy())
        
        # Use the improved detection function that works with blue-purple LEDs
        centers = detect_blue_circles(frame)

        # Handle case where centers is None (no LEDs detected)
        if centers is None:
            centers = []
        
        # Store detected centers for this frame
        detected_centers.append(centers)
            
        centers_xy = [(x, y) for (x, y, r) in centers]

        # Tracking logic (simple nearest neighbor)
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
        
        # Ring center = average of all detected LEDs
        if len(centers_xy) >= 1:  # Need at least one LED
            xs = np.array([pt[0] for pt in centers_xy])
            ys = np.array([pt[1] for pt in centers_xy])
            center_x = np.mean(xs)
            center_y = np.mean(ys)
            ring_centers.append((center_x, center_y))
        else:
            ring_centers.append(None)

        # Draw and save frame with all detected LEDs
        frame_draw = draw_leds(frame, centers, ring_center=ring_centers[-1])
        frame_save_path = os.path.join(frames_folder, f'frame_{frame_idx:05d}.png')
        cv2.imwrite(frame_save_path, frame_draw)

        # Print frame progress
        if frame_idx % 50 == 0 or frame_idx == total_frames - 1:
            print(f"    Frame {frame_idx+1}/{total_frames} ...")
        frame_idx += 1

    cap.release()

    # Save raw LED tracks as CSV (before filtering)
    csv_path = os.path.join(video_out_folder, f'{video_name}_led_tracks_raw.csv')
    os.makedirs(video_out_folder, exist_ok=True)
    with open(csv_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        header = ['frame']
        for i in range(len(led_tracks)):
            header += [f'led{i+1}_x', f'led{i+1}_y']
        writer.writerow(header)
        for f in range(frame_idx):
            row = [f]
            for track in led_tracks:
                if f < len(track) and track[f] is not None:
                    row += [track[f][0], track[f][1]]
                else:
                    row += [None, None]
            writer.writerow(row)
    
    # Process and save ring center data
    centers_interp = interpolate_centers(ring_centers)
    centers_smooth = smooth_centers(centers_interp, window=9, poly=2)
    if len(centers_smooth) > 0:
        # Save smoothed ring center trajectory as CSV
        ring_center_csv_path = os.path.join(video_out_folder, f'{video_name}_ring_center.csv')
        with open(ring_center_csv_path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['frame', 'ring_x', 'ring_y'])
            for i, (x, y) in enumerate(centers_smooth):
                writer.writerow([i, x, y])
        
        # Calculate velocity for time-aware analysis
        if len(centers_smooth) > 1:
            velocities = np.linalg.norm(np.diff(centers_smooth, axis=0), axis=1)
            velocities_reshape = velocities.reshape(-1, 1)
            # Only run KMeans if there are at least 2 valid (non-NaN) velocities
            valid_velocities = velocities[~np.isnan(velocities)]
            if len(valid_velocities) >= 2:
                velocities_reshape = np.nan_to_num(velocities_reshape, nan=np.nanmedian(valid_velocities))
                kmeans = KMeans(n_clusters=2, random_state=0).fit(velocities_reshape)
                labels_kmeans = kmeans.labels_
            else:
                print("  Skipping KMeans: not enough valid velocities.")
    
    # Filter out ambient/static LED tracks
    filtered_led_tracks = filter_ambient_led_tracks(led_tracks, min_frames=10, min_disp=20, group_corr_thresh=0.5)
    print(f"  Filtered LED tracks: {len(led_tracks)} -> {len(filtered_led_tracks)}")
    
    # Save filtered LED tracks as CSV
    filtered_csv_path = os.path.join(video_out_folder, f'{video_name}_led_tracks_filtered.csv')
    with open(filtered_csv_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        header = ['frame']
        for i in range(len(filtered_led_tracks)):
            header += [f'led{i+1}_x', f'led{i+1}_y']
        writer.writerow(header)
        for f in range(frame_idx):
            row = [f]
            for track in filtered_led_tracks:
                if f < len(track) and track[f] is not None:
                    row += [track[f][0], track[f][1]]
                else:
                    row += [None, None]
            writer.writerow(row)
    
    # REDRAW FILTERED FRAMES - with only the filtered LEDs
    print("  Redrawing frames with filtered LEDs only...")
    filtered_ring_centers = []
    
    for f in range(frame_idx):
        # Get original frame
        frame = original_frames[f]
        
        # Get centers for this frame that match filtered tracks
        filtered_centers = []
        for track_idx, track in enumerate(filtered_led_tracks):
            if f < len(track) and track[f] is not None:
                # Find the matching center with radius
                for cx, cy, r in detected_centers[f]:
                    if abs(cx - track[f][0]) < 2 and abs(cy - track[f][1]) < 2:
                        filtered_centers.append((cx, cy, r))
                        break
        
        # Ring center = average of all filtered LEDs
        if len(filtered_centers) >= 1:
            xs = np.array([x for (x, y, r) in filtered_centers])
            ys = np.array([y for (x, y, r) in filtered_centers])
            center_x = np.mean(xs)
            center_y = np.mean(ys)
            filtered_ring_centers.append((center_x, center_y))
        else:
            filtered_ring_centers.append(None)

            
        # Draw filtered LEDs
        frame_draw = draw_leds(frame, filtered_centers, ring_center=filtered_ring_centers[-1])
        
        # Save filtered frame
        frame_save_path = os.path.join(filtered_frames_folder, f'frame_{f:05d}.png')
        cv2.imwrite(frame_save_path, frame_draw)
    
    # Save filtered ring centers data
    if filtered_ring_centers:
        centers_interp = interpolate_centers(filtered_ring_centers)
        centers_smooth = smooth_centers(centers_interp, window=9, poly=2)
        
        if len(centers_smooth) > 0:
            # Save smoothed filtered ring center trajectory as CSV
            filtered_ring_csv_path = os.path.join(video_out_folder, f'{video_name}_filtered_ring_center.csv')
            with open(filtered_ring_csv_path, 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(['frame', 'ring_x', 'ring_y'])
                for i, (x, y) in enumerate(centers_smooth):
                    writer.writerow([i, x, y])
            
    if filtered_led_tracks:  # Check if filtered_led_tracks is not empty
        max_len = max(len(track) for track in filtered_led_tracks)
        # Calculate ring center using filtered LEDs
        led_positions = []
        for track in filtered_led_tracks:
            arr = np.full((max_len, 2), np.nan)
            for i, pt in enumerate(track):
                if pt is not None:
                    arr[i] = pt
            led_positions.append(arr)
        led_positions = np.array(led_positions)  # shape: (num_leds, num_frames, 2)
        
        # Calculate ring center for each frame as mean of filtered LEDs
        ring_centers_filtered_leds = np.nanmean(led_positions, axis=0)  # shape: (num_frames, 2)
        
        # Save the filtered-LED ring center as CSV
        filtered_leds_center_csv_path = os.path.join(video_out_folder, f'{video_name}_all_leds_center.csv')
        with open(filtered_leds_center_csv_path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['frame', 'center_x', 'center_y'])
            for i, (x, y) in enumerate(ring_centers_filtered_leds):
                writer.writerow([i, x, y])
    else:
        print(f"  Warning: No LED tracks remaining after filtering for {video_name}. Skipping all-LED center calculation.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process videos to track LEDs')
    parser.add_argument('--input', '-i', required=True, help='Input directory containing video files')
    parser.add_argument('--output', '-o', required=True, help='Output directory for results')
    args = parser.parse_args()
    
    input_folder = args.input
    output_folder = args.output
    
    os.makedirs(output_folder, exist_ok=True)
    
    video_files = [fname for fname in os.listdir(input_folder) 
                  if fname.lower().endswith(('.mp4'))]
    total_videos = len(video_files)
    for idx, fname in enumerate(video_files, 1):
        video_path = os.path.join(input_folder, fname)
        process_video(video_path, output_folder, idx, total_videos)
    print("All videos processed.")
