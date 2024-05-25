import os

import pandas as pd
import glob
from datasets import load_dataset


# download images
def download_data(dataset_name, folder_name, split):
    dataset = load_dataset(dataset_name, split=split)
    save_dir = f"data/{folder_name}/{split}"
    os.makedirs(save_dir, exist_ok=True)
    # Iterate over the dataset and save images
    for idx, (image, label) in enumerate(zip(dataset['image'], dataset['label'])):
        filename = os.path.join(save_dir, f"{idx}_{label}.png")
        image.save(filename)

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


# make csv
def make_csv(folder_name, split, file_ext='png'):
    filepath = f"data/{folder_name}/{split}/"
    filenames = [str(i[len(filepath):]) for i in glob.glob(f"{filepath}*.{file_ext}")]
    categories = [int(i[-(len(file_ext)+2)]) for i in filenames]
    df = pd.DataFrame(
        {
            'img_path':filenames,
            'label':categories
        }
    )
    df.to_csv(f'data/{folder_name}/{split}.csv', index=False)


make_csv(folder_name='alzheimer_mri', split='train')
make_csv(folder_name='alzheimer_mri', split='test')
