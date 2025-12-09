import yaml
import os
import sys

# Ensure src is in python path to import config if run directly
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import config

def create_data_yaml(output_path='data.yaml'):
    """
    Creates a data.yaml file for YOLOv8 training.
    Uses absolute paths from config.py to ensure Ultralytics can find the data.
    """
    
    print(f"Checking data directories...")
    # Check if directories exist
    all_exist = True
    for name, p in [('train', config.TRAIN_DIR), ('val', config.VAL_DIR), ('test', config.TEST_DIR)]:
        if not os.path.exists(p):
            print(f"WARNING: {name} directory not found at: {p}")
            all_exist = False
        else:
            print(f"Found {name}: {p}")

    if not all_exist:
        print("Warning: Some data directories are missing. Training might fail.")

    data = {
        'train': config.TRAIN_DIR,
        'val': config.VAL_DIR,
        'test': config.TEST_DIR,
        'nc': len(config.CLASSES),
        'names': config.CLASSES
    }
    
    # Write to file
    with open(output_path, 'w') as file:
        yaml.dump(data, file, default_flow_style=False)
    
    print(f"Successfully created YOLO configuration at: {output_path}")

if __name__ == "__main__":
    # Create data.yaml in the project root by default
    yaml_path = os.path.join(config.BASE_DIR, 'data.yaml')
    create_data_yaml(yaml_path)
