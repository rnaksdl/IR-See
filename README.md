# IR-See

## Overview

IR-See analyzes infrared LED trajectories from VR controller recordings to infer PIN inputs on virtual keypads.

## Requirements
- The commands below are for linux systems
- Python 3.8+
- FFmpeg
- OpenCV, NumPy, Pandas, scikit-learn, SciPy, Matplotlib
- BeautifulSoup4, hmmlearn
- (Recording only) Raspberry Pi + PiCamera2

```bash
pip install opencv-python numpy pandas scikit-learn scipy matplotlib beautifulsoup4 hmmlearn
```

## Pipeline (0° Camera)

```bash
cd scripts

# 0. Generate PIN trajectory lookup tables
python 000_generate_pin_trajectories.py

# 1. Record IR video (only for Raspberry Pi)
python 100_record.py

# 2. Preprocess video
python 200_batch_adjust.py ../0_reg/4digit/input ../0_reg/4digit/input_e \
    --brightness -0.8 --saturation 3.0 --blackpoint 0.50 --temp-k 18000 --probe -v

# 3. Track IR LEDs
python 300_track.py -i ../0_reg/4digit/input_e -o ../0_reg/4digit/output_e

# 4. Infer PINs
python 400_guess.py -i ../0_reg/4digit/output_e -o ../0_reg/4digit/report_e

# 5. Generate summary
python 500_simple_report.py -i ../0_reg/4digit/report_e
```
#### For 5 digit pins
```
# Make sure to be in /IR-See/scripts and have trajectory pkl files generated
python 200_batch_adjust.py ../0_reg/5digit/input ../0_reg/5digit/input_e \
    --brightness -0.8 --saturation 3.0 --blackpoint 0.50 --temp-k 18000 --probe -v
python 300_track.py -i ../0_reg/5digit/input_e -o ../0_reg/5digit/output_e
python 400_guess.py -i ../0_reg/5digit/output_e -o ../0_reg/5digit/report_e --pin-length 5
python 500_simple_report.py -i ../0_reg/5digit/report_e
```
#### For 6 digit pins
```
# Make sure to be in /IR-See/scripts and have trajectory pkl files generated
python 200_batch_adjust.py ../0_reg/6digit/input ../0_reg/6digit/input_e \
    --brightness -0.8 --saturation 3.0 --blackpoint 0.50 --temp-k 18000 --probe -v
python 300_track.py -i ../0_reg/6digit/input_e -o ../0_reg/6digit/output_e
python 400_guess.py -i ../0_reg/6digit/output_e -o ../0_reg/6digit/report_e --pin-length 6
python 500_simple_report.py -i ../0_reg/6digit/report_e
```


## Angled Camera Variants

| Script | Use Case |
|--------|----------|
| `401_guess_angle.py` | Oblique angles (e.g., pitch 60°) |
| `402_guess_90.py` | 90° views (top/bottom/left/right) |

### Infer PINs for videos taken with some yaw or pitch
(We give some extra angles for error tolerance)
#### for pitch_p60
```bash
python 300_track.py -i ../1_angle/pitch_p60/input_e -o ../1_angle_/pitch_p60/output_e
python 401_guess_angle.py -i ../1_angle/pitch_p60/output_e -o ../1_angle_/pitch_p60/report_e --yaw 0 0 --pitch 30 90
python 500_simple_report.py -i ../1_angle/pitch_p60/report_e
```
#### for yaw_p45
```bash
python 300_track.py -i ../1_angle/yaw_p45/input_e -o ../1_angle_/yaw_p45/output_e
python 401_guess_angle.py -i ../1_angle/yaw_p45/output_e -o ../1_angle_/yaw_p45/report_e --yaw 15 75 --pitch 0 0
python 500_simple_report.py -i ../1_angle/yaw_p45/report_e
```
### Infer PINs for videos taken with 90 degrees yaw or pitch
#### for pitch_p90
```bash
python 302_track_90.py -i ../2_90/pitch_p90/input_e -o ../2_90/pitch_p90/output_e --camera-position top
python 402_guess_90.py -i ../2_90/pitch_p90/output_e -o ../2_90/pitch_p90/report_e --camera-position top
python 500_simple_report.py -i ../2_90/pitch_p90/report_e
```
#### for yaw_p90
```bash
python 302_track_90.py -i ../2_90/yaw_p90/input_e -o ../2_90/yaw_p90/output_e --camera-position right
python 402_guess_90.py -i ../2_90/yaw_p90/output_e -o ../2_90/yaw_p90/report_e --camera-position right
python 500_simple_report.py -i ../2_90/yaw_p90/report_e
```

## License

MIT
