"""
Module used to generate synthetic images using OpenAI's API
"""

import os

from io import BytesIO
from PIL import Image
from openai import OpenAI
import requests
import random
import shutil
import time

client = OpenAI()


def generateImages(image_path, destination_directory):
    """
    Given an image, generates a similar synthetic image and saves both in the
    specified folder
    """
    label = image_path.split('/')[-2]
    os.makedirs(destination_directory, exist_ok=True)
    
    image_filenames = [
        name for name in os.listdir(destination_directory)
    ]
    
    n = len(image_filenames)

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
        
        # Save synthetic image to file
        synthetic_image_filepath = f'{destination_directory}/{label}_generated{n+1}.png'
        synthetic_img.save(synthetic_image_filepath)
        
        # Wait a second to avoid rate limiting
        time.sleep(5)


def makeSyntheticTrain(train_directory, synthetic_directory, train_percentage, synthetic_percentage):

    # Remove any existing images in directory
    try:
        shutil.rmtree(synthetic_directory)
    except:
        print("directory does not exist")

    # Loop through subfolders, generate synthetic images
    subfolders = [f for f in os.listdir(train_directory)]

    for s in subfolders:
        os.makedirs(f"{synthetic_directory}/{s}", exist_ok=True)
        
        subfolder_path = f"{train_directory}/{s}"
        files = os.listdir(subfolder_path)
        sample_files = random.sample(files, round(len(files)*train_percentage))
        
        # Move sample files to synthetic directory
        for f in sample_files:
            
            image_path = f"{subfolder_path}/{f}"
            destination_directory = f"{synthetic_directory}/{s}/"
            shutil.copyfile(image_path, destination_directory + image_path.split('/')[-1])
            
        # Create synthetic images
        source_files = random.sample(sample_files, round(len(sample_files)*synthetic_percentage))
        
        for f in source_files:
            
            image_path = f"{subfolder_path}/{f}"
            destination_directory = f"{synthetic_directory}/{s}/"
            generateImages(image_path, destination_directory)
