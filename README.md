# Anti-Gravity Object Detection Model

A YOLOv8-based object detection model for military vehicles

## Directory Structure
```
Object-Detection-Model/
├── data/                   # Dataset (train, valid, test)
├── notebooks/              # Original Jupyter notebooks
├── src/                    # Source code
│   ├── config.py           # Configuration settings
│   ├── data_setup.py       # Data configuration generator
│   ├── dataset.py          # Custom dataset utils (legacy)
│   ├── predict.py          # Inference script
│   ├── train.py            # Training script
│   └── visualization.py    # Visualization utils
└── requirements.txt        # Dependencies
```

## Setup

1. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Prepare Data Configuration**
   This script generates the `data.yaml` file required by YOLO, using absolute paths from your local environment.
   ```bash
   python src/data_setup.py
   ```

## Usage

### Training
To train the model from scratch (or fine-tune):
```bash
python src/train.py --epochs 50 --batch-size 16 --model yolov8n.pt
```
Arguments:
- `--epochs`: Number of training epochs (default: 50)
- `--batch-size`: Batch size (default: 16)
- `--model`: Base model to use (default: `yolov8n.pt`)

### Inference (Prediction)
To run object detection on an image, video, or webcam:
```bash
python src/predict.py --source path/to/image.jpg
```
Arguments:
- `--source`: Path to image/video or `0` for webcam.
- `--model`: Path to trained model weights (defaults to best training run or `yolov8n.pt`).
- `--conf`: Confidence threshold (default: 0.25).

## Class Names
The model detects the following classes:
- TANK, IFV, APC, EV, AH, TH, AAP, TA, AA, TART, SPART
