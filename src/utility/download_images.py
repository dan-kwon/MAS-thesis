"""
Script used to download dataset from huggingface
"""

import os

import pandas as pd
import glob
from datasets import load_dataset


# download images
def download_data(dataset_name, folder_name, split):
    dataset = load_dataset(dataset_name, split=split)
    
    # Iterate over the dataset and save images
    label_counts = [0]*4
    for idx, (image, label) in enumerate(zip(dataset['image'], dataset['label'])):
        
        if label == 0:
            label_name = 'MildDemented'
        elif label == 1:
            label_name = 'ModerateDemented'
        elif label == 2:
            label_name = 'NonDemented'
        elif label == 3:
            label_name = 'VeryMildDemented'

        save_dir = f"data/{folder_name}/{split}/{label_name}"
        os.makedirs(save_dir, exist_ok=True)
        filename = os.path.join(save_dir, f"{label_name}_{label_counts[label]}.png")    
        image.save(filename)
        label_counts[int(label)] += 1

        if idx % 100 == 0:  # Print progress every 100 images
            print(f"Saved {idx} images")

    print("All images saved successfully!")


download_data(
    dataset_name='Falah/Alzheimer_MRI',
    folder_name='alzheimer_mri',
    split='train'
)
download_data(
    dataset_name='Falah/Alzheimer_MRI',
    folder_name='alzheimer_mri',
    split='test'
)
