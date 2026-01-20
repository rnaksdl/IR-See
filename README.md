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

### Generate PIN trajectory lookup tables
```bash
cd scripts

python 000_generate_pin_trajectories.py
```

#### Recroding (ONLY if you want to record new videos with Raspberry Pi)
```bash
python 100_record.py
```
#### to start a recording
press 1 or l
#### to stop and save the recording
press 2 or ;
#### to quit and NOT save the recording
press 3 or '

### Process videos

#### 4 digit pins
```bash
# 1. Preprocess video
python 200_batch_adjust.py ../0_reg/4digit/input ../0_reg/4digit/input_e --brightness -0.8 --saturation 3.0 --blackpoint 0.50 --temp-k 18000 --probe -v

# 2. Track IR LEDs
python 300_track.py -i ../0_reg/4digit/input_e -o ../0_reg/4digit/output_e

# 3. Infer PINs
python 400_guess.py -i ../0_reg/4digit/output_e -o ../0_reg/4digit/report_e

# 4. Generate summary
python 500_simple_report.py -i ../0_reg/4digit/report_e
```
#### 5 digit pins
```bash
python 200_batch_adjust.py ../0_reg/5digit/input ../0_reg/5digit/input_e --brightness -0.8 --saturation 3.0 --blackpoint 0.50 --temp-k 18000 --probe -v
python 300_track.py -i ../0_reg/5digit/input_e -o ../0_reg/5digit/output_e
python 400_guess.py -i ../0_reg/5digit/output_e -o ../0_reg/5digit/report_e --pin-length 5
python 500_simple_report.py -i ../0_reg/5digit/report_e
```
#### 6 digit pins
```bash
python 200_batch_adjust.py ../0_reg/6digit/input ../0_reg/6digit/input_e --brightness -0.8 --saturation 3.0 --blackpoint 0.50 --temp-k 18000 --probe -v
python 300_track.py -i ../0_reg/6digit/input_e -o ../0_reg/6digit/output_e
python 400_guess.py -i ../0_reg/6digit/output_e -o ../0_reg/6digit/report_e --pin-length 6
python 500_simple_report.py -i ../0_reg/6digit/report_e
```

#### Pitch +60°
```bash
python 200_batch_adjust.py ../1_angle/pitch_p60/input ../1_angle/pitch_p60/input_e --brightness -0.8 --saturation 3.0 --blackpoint 0.50 --temp-k 18000 --probe -v
python 300_track.py -i ../1_angle/pitch_p60/input_e -o ../1_angle/pitch_p60/output_e
python 400_guess.py -i ../1_angle/pitch_p60/output_e -o ../1_angle/pitch_p60/report_e
python 500_simple_report.py -i ../1_angle/pitch_p60/report_e
```
#### yaw + 45°
```bash
python 200_batch_adjust.py ../1_angle/yaw_p45/input ../1_angle/yaw_p45/input_e --brightness -0.8 --saturation 3.0 --blackpoint 0.50 --temp-k 18000 --probe -v
python 300_track.py -i ../1_angle/yaw_p45/input_e -o ../1_angle/yaw_p45/output_e
python 400_guess.py -i ../1_angle/yaw_p45/output_e -o ../1_angle/yaw_p45/report_e
python 500_simple_report.py -i ../1_angle/yaw_p45/report_e
```
#### for pitch_p90
```bash
python 200_batch_adjust.py ../2_90/pitch_p90/input ../2_90/pitch_p90/input_e --brightness -0.8 --saturation 3.0 --blackpoint 0.50 --temp-k 18000 --probe -v
python 302_track_90.py -i ../2_90/pitch_p90/input_e -o ../2_90/pitch_p90/output_e --camera-position top
python 401_guess_90.py -i ../2_90/pitch_p90/output_e -o ../2_90/pitch_p90/report_e --camera-position top
python 500_simple_report.py -i ../2_90/pitch_p90/report_e
```
#### for yaw_p90
```bash
python 200_batch_adjust.py ../2_90/yaw_p90/input ../2_90/yaw_p90/input_e --brightness -0.8 --saturation 3.0 --blackpoint 0.50 --temp-k 18000 --probe -v
python 302_track_90.py -i ../2_90/yaw_p90/input_e -o ../2_90/yaw_p90/output_e --camera-position right
python 401_guess_90.py -i ../2_90/yaw_p90/output_e -o ../2_90/yaw_p90/report_e --camera-position right
python 500_simple_report.py -i ../2_90/yaw_p90/report_e
```

## License

MIT
