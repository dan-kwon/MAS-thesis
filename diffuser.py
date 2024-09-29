import PIL
import requests
from torchvision import transforms
from torchvision.transforms import v2
import torch
from diffusers import StableDiffusionInstructPix2PixPipeline, EulerAncestralDiscreteScheduler

model_id = "timbrooks/instruct-pix2pix"
pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(model_id, torch_dtype=torch.float16, safety_checker=None)
pipe.to("cpu")
pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)

image = PIL.Image.open("data/alzheimer_mri/train/MildDemented/MildDemented_0.png")
image = PIL.ImageOps.exif_transpose(image)
#image = image.convert("RGB")
tform = transforms.Compose([
    #transforms.ToTensor(),
    v2.Grayscale(num_output_channels=3),
    transforms.Resize(
        (224, 224)),
    ])

inp = tform(image).to('cpu').unsqueeze(0)

prompt = "create a similar image"
images = pipe(prompt, image=tform(image), num_inference_steps=10, image_guidance_scale=15).images
images[0]