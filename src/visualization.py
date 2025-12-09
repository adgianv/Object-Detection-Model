import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

def plot_examples(data_loader, class_names, num_examples=5, images_per_row=4):
    images, boxes_list, labels_list = next(iter(data_loader))
    
    # Calculate number of rows needed
    num_rows = (num_examples + images_per_row - 1) // images_per_row  # This ensures we round up

    fig, axs = plt.subplots(num_rows, images_per_row, figsize=(20, 5 * num_rows))
    axs = axs.flatten()  # Flatten to easily index the axes

    for i in range(num_examples):
        image = images[i].permute(1, 2, 0).numpy()
        boxes = boxes_list[i]
        labels = labels_list[i]
        
        image = image * 255  # Denormalize to [0, 255]
        image = image.astype(np.uint8)
        
        axs[i].imshow(image)
        axs[i].axis('off')
        
        for box, label in zip(boxes, labels):
            x_center, y_center, width, height = box
            x_min = (x_center - width / 2) * image.shape[1]
            y_min = (y_center - height / 2) * image.shape[0]
            width *= image.shape[1]
            height *= image.shape[0]
            
            rect = patches.Rectangle((x_min, y_min), width, height, linewidth=2, edgecolor='r', facecolor='none')
            axs[i].add_patch(rect)
            axs[i].text(x_min, y_min, class_names[label], bbox=dict(facecolor='yellow', alpha=0.5))
    
    # Hide any unused subplots
    for j in range(num_examples, len(axs)):
        axs[j].axis('off')

    plt.tight_layout()
    plt.show()
