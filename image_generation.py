import os

import numpy as np
from io import BytesIO
from PIL import Image
from openai import OpenAI
import requests
import time
import shutil




client = OpenAI()


def generate_images(folder_name, sample_size):
    """
    Utility function that will sample images from a given folder and pass to the
    OpenAI API to generate similar images
    """
    image_filenames = [
        name for name in os.listdir(folder_name) 
        if (os.path.isfile(f'{folder_name}/{name}'))
        & (np.random.uniform(0,1) <= sample_size)
    ]

    # Copy original images over to new folder as well
    folder_root = '/'.join(folder_name.split('/')[:2])
    label_name = folder_name.split('/')[-1]
    save_dir = f"{folder_root}/synthetic/percent_{sample_size*100:.0f}/{label_name}"
    # Write image to folder
    os.makedirs(save_dir, exist_ok=True)
    for img in image_filenames:
        shutil.copy(f'{folder_name}/{img}', f'{save_dir}/{img}')

    # Read the image file from disk and convert to RGB
    # (OpenAI API need RGB, so we'll convert back to greyscale after)
    num_files_written = 0
    while num_files_written < len([
        name for name in os.listdir(folder_name) 
        if (os.path.isfile(f'{folder_name}/{name}'))]):
        
        for n,f in enumerate(image_filenames):
            
            image = Image.open(f'{folder_name}/{f}')
            image = image.convert('RGB')

            # Convert the image to a BytesIO object
            byte_stream = BytesIO()
            image.save(byte_stream, format='PNG')
            byte_array = byte_stream.getvalue()

            # Make API call 
            response = client.images.create_variation(
                image=byte_array,
                n=1,
                model="dall-e-2",
                size="1024x1024"
            )

            image_url = response.data[0].url

            # Download image from URL provided from API
            img_data = requests.get(image_url).content
            
            # Write image
            with open(f'{save_dir}/{label_name}_generated{n}.png', 'wb') as handler:
                handler.write(img_data)

            # Wait 15 seconds to avoid rate limiting
            time.sleep(5)
            
            # Increment number of files
            num_files_written += 1

generate_images(
    folder_name='data/alzheimer_mri/train/ModerateDemented',
    sample_size=0.5
)
