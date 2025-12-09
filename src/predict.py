from ultralytics import YOLO
import argparse
import os
import sys

# Ensure src is in python path to import config if run directly
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import config

def predict(source, model_path='yolov8n.pt', save=True, conf=0.25):
    """
    Runs inference on the source using the specified model.
    """
    print(f"Starting inference...")
    print(f"Source: {source}")
    print(f"Model: {model_path}")
    
    if not os.path.exists(model_path):
        # Trying to find the best model from default runs directory if specific path not provided
        default_model_path = os.path.join(config.BASE_DIR, 'runs/detect/yolov8_military_vehicles/weights/best.pt')
        if os.path.exists(default_model_path):
            print(f"Model not found at {model_path}, using best model from training: {default_model_path}")
            model_path = default_model_path
        elif model_path == 'yolov8n.pt': 
             pass # Use pretrained path
        else:
            print(f"Error: Model not found at {model_path} and no trained model found.")
            return

    # Load a model
    model = YOLO(model_path)

    # Run inference
    results = model.predict(source=source, save=save, conf=conf, project=os.path.join(config.BASE_DIR, 'runs/detect'), name='predict')
    
    print("Inference completed.")
    print(f"Results saved to {results[0].save_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run inference with YOLOv8 model')
    parser.add_argument('--source', type=str, required=True, help='Path to image, video directory, or 0 for webcam')
    parser.add_argument('--model', type=str, default='yolov8n.pt', help='Path to model file (default: yolov8n.pt or best.pt if exists)')
    parser.add_argument('--conf', type=float, default=0.25, help='Confidence threshold')
    
    args = parser.parse_args()
    
    predict(source=args.source, model_path=args.model, conf=args.conf)
