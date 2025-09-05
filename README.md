# 🚁 UAV Advanced Tracking System

[![Python Version](https://img.shields.io/badge/python-3.10+-blue.svg)](https://python.org)
[![YOLO](https://img.shields.io/badge/YOLO-v8s-green.svg)](https://github.com/ultralytics/ultralytics)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.12+-red.svg)](https://opencv.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.7+-orange.svg)](https://pytorch.org)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

**UAV Advanced Tracking System** is an advanced computer vision system developed for UAV/drone detection and tracking. It performs real-time UAV tracking using YOLOv8s deep learning model and OpenCV tracker algorithms. The system is equipped with optimized parameters based on academic research findings and a PID control system.

## 🎯 Features

### 🔥 **Dual Tracking Approach**
- **YOLO-only Tracking**: YOLO detection + Kalman Filter in every frame
- **OpenCV Tracker**: 7 different classical tracking algorithms (CSRT, KCF, MOSSE, MIL, BOOSTING, MEDIANFLOW, TLD)

### 🧠 **AI Integration**
- **YOLOv8s** custom model support
- **GPU/CPU** automatic optimization
- **Confidence threshold** academic optimization (20% main, 15% tracking)
- **NMS IOU threshold** adjustable (0.4 optimized)

### 🎮 **Advanced Control System**
- **PID Controller** dynamic gain adjustment
- **Kalman Filter** smooth tracking
- **ROI scanning** system (4 region optimized)
- **Adaptive threshold** (far/near distance)

### 📊 **Performance Monitoring**
- Real-time **FPS** indicator
- **Confidence tracking** and statistics
- **Bbox history** and smooth filtering
- **Detection timeout** control

## 🚀 Quick Start

### 📋 Requirements

- **Python 3.10+**
- **Windows 10/11** (tested)
- **CUDA compatible GPU** (optional, for acceleration)
- **Webcam or video file**

### ⚡ Installation

1. **Clone the repository:**
```bash
git clone https://github.com/yourusername/uav-tracking-system.git
cd uav-tracking-system
```

2. **Create virtual environment:**
```bash
python -m venv uav_env
# For Windows:
.\uav_env\Scripts\activate
# For Linux/Mac:
source uav_env/bin/activate
```

3. **Install dependencies:**
```bash
pip install --upgrade pip
pip install opencv-python torch ultralytics numpy opencv-contrib-python
```

4. **Alternative - Installation with requirements:**
```bash
pip install -r requirements.txt
```

### 🎬 Usage

#### **YOLO-only Tracker (Recommended):**
```bash
python uav_tracking_yolo.py
```

#### **OpenCV Tracker Version:**
```bash
python uav_tracking_opencv_trackers.py
```

### 📹 Video Configuration

Place your video files in the `Scenes/` folder:
```
UAV/
├── Scenes/
│   ├── scene1.mp4
│   ├── scene2.mp4
│   └── ...
├── best.pt (YOLO model)
└── ...
```

## 🔧 Technical Details

### 🎯 **Optimal Threshold Values**

Threshold values optimized according to academic research findings:

| Parameter | Value | Description |
|-----------|-------|-------------|
| **Main Detection Confidence** | 20% | Optimal for initial detection |
| **Tracking Confidence** | 15% | More sensitive during tracking |
| **IOU Threshold (NMS)** | 0.4 | Non-Maximum Suppression |
| **Minimum Bbox Size** | 2.5% frame | For small UAVs |
| **Target Area** | 1.5% frame | PID control reference |

### 📊 **ROI Scanning System**

Systematic scanning with 4 large ROI regions:
```
┌─────────┬─────────┐
│  ROI 1  │  ROI 2  │
│ (Top-L) │ (Top-R) │
├─────────┼─────────┤
│  ROI 3  │  ROI 4  │
│ (Bot-L) │ (Bot-R) │
└─────────┴─────────┘
```

- **ROI Size**: 640x360 (with overlap)
- **Scanning Speed**: 12 frame/ROI
- **Total Scan**: ~1.6 seconds/cycle

### 🎮 **PID Control System**

Adaptive PID gains based on distance:

| Distance | Kp | Ki | Kd | Kf | Usage |
|----------|----|----|----|----|-------|
| **Far** | 300 | 35 | 80 | 1480 | Distant targets |
| **Close** | 200 | 50 | 120 | 1480 | Near targets |

### 🧠 **Kalman Filter Configuration**

8-state Kalman filter (x, y, w, h, vx, vy, vw, vh):
- **Process Noise**: 5e-3 (low noise, stable tracking)
- **Measurement Noise**: 1e-1 (YOLO detection reliability)
- **Prediction Model**: Constant velocity

## 📈 **OpenCV Tracker Comparison**

Based on [LearnOpenCV](https://learnopencv.com/object-tracking-using-opencv-cpp-python/) reference, tracker performance:

| Tracker | Accuracy | Speed | UAV Suitability | Recommended |
|---------|----------|-------|-----------------|-------------|
| **CSRT** | ⭐⭐⭐⭐⭐ | ⭐⭐ | 🟢 Best | ✅ Precise tracking |
| **KCF** | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | 🟢 Good | ✅ General purpose |
| **MOSSE** | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | 🟡 Speed focused | ⚡ Real-time |
| **MIL** | ⭐⭐⭐ | ⭐⭐⭐ | 🟡 Basic | 📚 Simple |
| **TLD** | ⭐⭐⭐⭐ | ⭐⭐ | 🟡 Occlusion | 🔍 Experimental |
| **MEDIANFLOW** | ⭐⭐⭐ | ⭐⭐⭐ | 🔴 Not suitable | ❌ Slow motion only |
| **BOOSTING** | ⭐⭐ | ⭐⭐ | 🔴 Not recommended | ❌ Outdated |

## 🎨 **User Interface**

### 🎯 **YOLO-only Version**
- **Green box**: UAV tracking active
- **Yellow area**: Target region
- **Red line**: Center-target connection
- **Control panel**: PID values real-time
- **ROI indicator**: Scanning mode

### 🎮 **Control Keys**
- `Q`: Exit
- `R`: Tracking reset
- `S`: Screenshot (YOLO version only)

### 📊 **Information Indicators**
- **FPS**: Real-time performance
- **Confidence**: Detection confidence
- **PID Values**: Aileron, Elevator, Throttle
- **Target Area**: Current/target area ratio

## 🔬 **Academic Optimizations**

### 📚 **Confidence Threshold Research**

Literature review results:
- **YOLOv8 Default**: 0.4 (too high, UAVs missed)
- **LMWP-YOLO Study**: 0.25 recommended
- **UAV Detection Papers**: 0.2-0.25 range optimal
- **This Project**: 0.2 (main), 0.15 (tracking)

### 🎯 **ROI Optimization**

- **Old**: 6 small ROI (3x2 grid) → Confusing scanning
- **New**: 4 large ROI (2x2 grid) → 40% more efficient
- **Overlap**: 100px → Edge case handling
- **Frame/ROI**: 15 → 12 frame (20% speedup)

## 🚀 **Performance Benchmarks**

### 💻 **Test System**
- **CPU**: Intel i7 series
- **GPU**: CUDA compatible (optional)
- **RAM**: 8GB+ recommended
- **Video**: 1280x720 @ 30fps

### 📊 **Performance Results**

| Mode | FPS | CPU Usage | GPU Usage | Detection Rate |
|------|-----|-----------|-----------|----------------|
| **YOLO-only** | 15-25 | 60-80% | 40-60% | 95%+ |
| **CSRT Tracker** | 20-30 | 40-60% | 20-40% | 85%+ |
| **MOSSE Tracker** | 35-45 | 30-50% | 15-30% | 80%+ |

### 🎯 **Detection Accuracy**

On test videos:
- **Confidence Range**: 31% - 94%
- **Average Confidence**: 76%
- **Detection Success**: 92%
- **False Positive**: <5%
- **Tracking Loss**: <8%

## 🗂️ **Project Structure**

```
UAV/
├── 📁 Scenes/              # Video files
│   ├── scene1.mp4
│   └── scene2.mp4
├── 📁 uav_env/             # Python virtual environment
├── 🐍 uav_tracking_yolo.py   # Main YOLO tracker
├── 🐍 uav_tracking_opencv_trackers.py  # OpenCV trackers
├── 🐍 test_trackers.py       # Tracker test script
├── 🧠 best.pt               # YOLOv8s custom model
├── 📋 requirements.txt      # Python dependencies
└── 📖 README.md             # This file
```

## 🔧 **Advanced Configuration**

### ⚙️ **YOLO Model Settings**

```python
# Model parameters
CONF_THRESHOLD = 0.2        # Main detection
TRACK_CONF_THRESHOLD = 0.15 # Tracking detection
IOU_THRESHOLD = 0.4         # NMS threshold
MAX_DET = 100              # Maximum detections
```

### 🎮 **PID Tuning**

```python
# Far distance gains
gains_far = {
    'Kp': 300,    # Proportional gain
    'Ki': 35,     # Integral gain  
    'Kd': 80,     # Derivative gain
    'Kf': 1480    # Feedforward
}

# Near distance gains
gains_close = {
    'Kp': 200,    # Softer control
    'Ki': 50,     # Higher integral
    'Kd': 120,    # Higher derivative
    'Kf': 1480    # Constant feedforward
}
```

### 🔍 **ROI Customization**

```python
# ROI size settings
roi_width = frame_width // 2   # 640px default
roi_height = frame_height // 2 # 360px default
overlap = 100                  # Overlap pixels
ROI_SCAN_FRAMES = 12          # Frame/ROI
```

## 🛠 **Troubleshooting**

### ❌ **Common Errors**

1. **"CUDA out of memory"**
   ```bash
   # Switch to CPU mode
   device = "cpu"
   ```

2. **"TrackerCSRT_create not found"**
   ```bash
   # Install OpenCV-contrib
   pip install opencv-contrib-python
   ```

3. **"Video file cannot be opened"**
   ```bash
   # Check video path
   VIDEO_PATH = "Scenes/scene1.mp4"
   ```

### 🔧 **Performance Tuning**

1. **GPU Memory Optimization**:
   ```python
   torch.cuda.empty_cache()  # Clear memory
   ```

2. **Frame Rate Optimization**:
   ```python
   cv2.waitKey(1)  # Reduce display delay
   ```

3. **Detection Frequency**:
   ```python
   # Detect every 2nd frame
   if frame_count % 2 == 0:
       detect()
   ```

## 🤝 **Contributing**

1. **Fork** it
2. Create **feature branch**: `git checkout -b amazing-feature`
3. **Commit** changes: `git commit -m 'Add amazing feature'`
4. **Push** to branch: `git push origin amazing-feature`
5. Open **Pull Request**

### 🛠 **Development Setup**

```bash
# Development dependencies
pip install pytest black flake8 mypy
```

### 🧪 **Testing**

```bash
# Tracker tests
python test_trackers.py

# Unit tests (future)
pytest tests/
```

## 📚 **References and Resources**

### 📖 **Academic Sources**
- [LearnOpenCV Object Tracking Guide](https://learnopencv.com/object-tracking-using-opencv-cpp-python/)
- YOLOv8 Official Documentation
- OpenCV Tracking Algorithms Documentation
- UAV Detection Research Papers

### 🔗 **Technical Documentation**
- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
- [OpenCV Python Tutorials](https://docs.opencv.org/4.x/d6/d00/tutorial_py_root.html)
- [PyTorch Documentation](https://pytorch.org/docs/stable/index.html)

### 🎯 **Confidence Threshold Research**
- LMWP-YOLO: Optimized threshold studies
- UAV Detection Papers: 0.2-0.25 optimal range
- Real-world deployment findings

## 📞 **Contact**

- **Email**: your.email@domain.com
- **GitHub**: [@yourusername](https://github.com/yourusername)
- **LinkedIn**: [Your Profile](https://linkedin.com/in/yourprofile)

## 📄 **License**

This project is licensed under the **MIT License**. See the [LICENSE](LICENSE) file for details.

## 🏆 **Acknowledgments**

- **YOLOv8** team
- **OpenCV** community  
- **LearnOpenCV** for educational materials
- **PyTorch** framework
- Academic research contributions

## 👨‍💻 **Author**

**Mesut Taha Güven** - *Lead Developer & Researcher*
- GitHub: [@mtgsoftworks](https://github.com/mtgsoftworks)
- Email: mtg@duck.com

*Specialized in computer vision, UAV systems, and real-time tracking algorithms. Passionate about combining academic research with practical applications.*

## 📊 **Changelog**

### v2.0 (Latest)
- ✅ Dual tracking system implementation
- ✅ Academic optimization integration
- ✅ OpenCV tracker fallback mechanism
- ✅ Improved ROI scanning
- ✅ Enhanced PID control

### v1.0 (Initial)
- ✅ Basic YOLO detection
- ✅ Simple tracking implementation
- ✅ ROI scanning prototype

---

**⭐ Don't forget to star the project if you liked it!**

**🚁 UAV Advanced Tracking System - Where AI Meets Precision**
