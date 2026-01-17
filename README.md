# AutoClap
<img src="./assets/logo.png" alt="logo" width="200"/>

:👏An automated clapperboard detection tool that locates clap points in videos to streamline video editing workflows.

## 🧠 Motivation
Manual identification of clapper points in raw footage is time-consuming and repetitive, especially when handling large volumes of video. AutoClap was created to automate this process using computer vision, reducing editing time and enabling a more efficient post-production workflow.

This project also serves as a hands-on exploration of building a modular, extensible video-processing pipeline rather than focusing solely on model performance.

## 🎯 Goals
- Automatically detect clapperboards in video footage

- Convert detected timestamps into precise frame indices

- Design a detector pipeline that is easily extensible to new models

- Emphasize clean architecture over one-off scripts

## 🔍 System Architecture
```
Video Input
↓
Video Sampler (uniform / custom)
↓
Detector Pipeline
↓
Detector Output
↓
OCR Pipeline
↓
Result
```

## ⚙️ Installation
`poetry`
```Bash
# Clone repository
git clone https://github.com/Eumgill98/AutoClap.git
cd autoclap

# Install dependencies
poetry install
```

`pip`
```Bash
pip install git+https://github.com/Eumgill98/AutoClap.git
```

## 📁 Project Structure
```
autoclap
├──core
│    ├──output
│    ├──sampler
├──detector
│    ├──base
│    ├──yolov8
│    └──detector_pipeline.py
├──ocr
│    ├──base
│    ├──paddleocr
│    └──ocr_pipeline.py
└──README.md
```

## 📌 Example
```Python
from autoclap.detector import DetectorPipeline, YOLOv8Detector
from autoclap.core.sampler import TimeSlidingVideoSampler

from autoclap.ocr import OCRPipeline, PaddleOCRModel

VIDEO_FILE = YOUR_VIDEO_FILE

video_sampler = TimeSlidingVideoSampler(
    video = VIDEO_FILE,
    batch_size=16,
)

model = YOLOv8Detector(
    weight_path=PRETRAINED_MODEL_WEIGHT,
)

d_pipeline = DetectorPipeline(
    model= model
)

result = d_pipeline.run(
    video_sampler
)

print(result)
print()


# check ocr
result = result[0]
clapper_zone = result.get_clapperboard_zone()

ocr_pipeline = OCRPipeline(model=PaddleOCRModel(device='cpu'))
for (s_time, end_time, presence, bboxes) in clapper_zone:
    if presence:
        frame = result.get_frame_by_time(s_time)
        frame_bbox = (frame, bboxes)

        t = ocr_pipeline.run(frames_bboxes=frame_bbox, pad=5)
        print(t)
```

## ⭐ Pretrained Model Weight
- To be added
