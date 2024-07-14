"""
Module used to generate synthetic images using OpenAI's API
"""

import os

from io import BytesIO
from PIL import Image
from openai import OpenAI
import requests
import time
import shutil


client = OpenAI()


def generateImages(image_path):
    """
    Given an image, generates a similar synthetic image and saves both in the
    specified folder
    """
    label = image_path.split('/')[-2]
    folder_name = '/'.join(image_path.split('/')[:-1])+'/'
    destination_folder = folder_name.replace('train','synthetic_train')
    os.makedirs(destination_folder, exist_ok=True)
    
    shutil.copyfile(image_path, destination_folder + image_path.split('/')[-1])
    
    image_filenames = [
        name for name in os.listdir(destination_folder) 
        if (os.path.isfile(f'{folder_name}/{name}'))
    ]
    
    n = len(image_filenames)+1

    # Read the image file from disk and convert to RGB
    # (OpenAI API need RGB, so we'll convert back to greyscale after)
    image = Image.open(image_path)
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
    
    # Download image from URL provided from API
    image_url = response.data[0].url
    r = requests.get(image_url, stream=True)
    if r.status_code == 200:
        
        # Download image from URL and resize to 128x128
        synthetic_img = Image.open(BytesIO(r.content))
        synthetic_img = synthetic_img.resize((128,128))
        
        # Save image to file
        synthetic_image_filepath = f'{destination_folder}/{label}_generated{n+1}.png'
        synthetic_img.save(synthetic_image_filepath)
    
    # Wait a few seconds to avoid rate limiting
    time.sleep(2)
