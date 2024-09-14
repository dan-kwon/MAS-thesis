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
    image_num = image_path.split('/')[-1].split('.')[0].split('_')[-1]
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
        synthetic_image_filepath = f'{destination_directory}/{label}_{image_num}_generated{n+1}.png'
        synthetic_img.save(synthetic_image_filepath)
        
        # Wait a second to avoid rate limiting
        time.sleep(10)


def makeSyntheticTrain(train_directory, synthetic_directory, train_percentage, synthetic_percentage):

    # Remove any existing images in directory
    try:
        shutil.rmtree(synthetic_directory)
    except:
        print("directory does not exist")

    # Loop through subfolders, generate synthetic images
    subfolders = [f for f in os.listdir(train_directory)]

    for s in subfolders:
        # for each subfolder in the train directory, make the same in the synthetic train directory
        os.makedirs(f"{synthetic_directory}/{s}", exist_ok=True)
        
        # get a random sample from each subfolder
        subfolder_path = f"{train_directory}/{s}"
        files = os.listdir(subfolder_path)
        sample_files = random.sample(files, round(len(files)*train_percentage))
        
        # create synthetic sample based on sampled original images
        synthetic_subfolder_path = subfolder_path.replace('train','synthetic')
        synthetic_files = [f for f in os.listdir(synthetic_subfolder_path) if int(f.replace('.png','').split('_')[1]) in [int(f.replace('.png','').split('_')[1]) for f in sample_files]]
        synthetic_sample_files = random.sample(synthetic_files, round(len(files)*synthetic_percentage))
        
        # Move sample files to synthetic directory
        for f in sample_files:
            
            image_path = f"{subfolder_path}/{f}"
            destination_directory = f"{synthetic_directory}/{s}/"
            shutil.copyfile(image_path, destination_directory + image_path.split('/')[-1])

        # Move synthetic sample files to synthetic directory
        for f in synthetic_sample_files:

            image_path = f"{synthetic_subfolder_path}/{f}"
            destination_directory = f"{synthetic_directory}/{s}/"
            shutil.copyfile(image_path, destination_directory + image_path.split('/')[-1])