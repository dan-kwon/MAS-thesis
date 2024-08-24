from torchvision import transforms
from torchvision.transforms import v2
from diffusers import StableDiffusionImageVariationPipeline
from PIL import Image

device = "cpu"

sd_pipe = StableDiffusionImageVariationPipeline.from_pretrained(
    "lambdalabs/sd-image-variations-diffusers",
    revision="v2.0",
    )

sd_pipe = sd_pipe.to(device)

im = Image.open("data/alzheimer_mri/train/MildDemented/MildDemented_0.png")

tform = transforms.Compose([
    transforms.ToTensor(),
    v2.Grayscale(num_output_channels=3),
    transforms.Resize(
        (224, 224),
        interpolation=transforms.InterpolationMode.BICUBIC,
        antialias=False,
        ),
    transforms.Normalize(
        [0.48145466, 0.4578275, 0.40821073],
        [0.26862954, 0.26130258, 0.27577711]
        ),
    ])

inp = tform(im).to(device).unsqueeze(0)

out = sd_pipe(inp, guidance_scale=3)

out["images"][0].save("result.jpg")
