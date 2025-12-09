import os
import glob
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

class MilitaryVehiclesDataset(Dataset):
    def __init__(self, images_dir, labels_dir, img_size=640, transform=None):
        self.images_dir = images_dir
        self.labels_dir = labels_dir
        self.img_size = img_size
        self.transform = transform
        self.image_files = [f for f in os.listdir(images_dir) if f.endswith('.jpg') or f.endswith('.png')]
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        img_path = os.path.join(self.images_dir, self.image_files[idx])
        label_path = os.path.join(self.labels_dir, self.image_files[idx].replace('.jpg', '.txt').replace('.png', '.txt'))
        
        # Load image using OpenCV
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Resize image to img_size
        img = cv2.resize(img, (self.img_size, self.img_size))
        
        # Normalize image to range [0, 1]
        img = img.astype(np.float32) / 255.0
        
        # Load bounding boxes and labels
        boxes, labels = self.load_annotations(label_path)
        
        # Apply transformations if provided
        if self.transform:
            img, boxes = self.transform(img, boxes)
        
        # Convert everything to tensors
        img = torch.tensor(img, dtype=torch.float32).permute(2, 0, 1)  # CHW format
        boxes = torch.tensor(boxes, dtype=torch.float32)
        labels = torch.tensor(labels, dtype=torch.long)
        
        return img, boxes, labels
    
    def load_annotations(self, label_path):
        boxes = []
        labels = []
        
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    label = int(parts[0])
                    bbox = [float(x) for x in parts[1:]]
                    
                    # Append to list
                    boxes.append(bbox)
                    labels.append(label)
        
        return boxes, labels
    

def collate_fn(batch):
    """
    Custom collate function to handle varying number of bounding boxes per image.
    """
    imgs = []
    boxes_list = []
    labels_list = []

    for img, boxes, labels in batch:
        imgs.append(img)
        boxes_list.append(boxes)
        labels_list.append(labels)

    imgs = torch.stack(imgs, dim=0) if imgs else None
    return imgs, boxes_list, labels_list


def get_data_loader(images_dir, labels_dir, batch_size=4, shuffle=True, img_size=640, transform=None):
    dataset = MilitaryVehiclesDataset(images_dir, labels_dir, img_size=img_size, transform=transform)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn)
