# IR-See

Artifact for **"IR-See: Inferring PIN Entry on Virtual Keyboards via IR LED Tracking"** (ACM CCS 2025).

## Overview

IR-See analyzes infrared LED trajectories from VR controller recordings to infer PIN inputs on virtual keypads.

## Requirements

- Python 3.8+
- OpenCV, NumPy, Pandas, scikit-learn, SciPy, Matplotlib
- BeautifulSoup4, hmmlearn
- (Recording only) Raspberry Pi + PiCamera2

```bash
pip install opencv-python numpy pandas scikit-learn scipy matplotlib beautifulsoup4 hmmlearn
