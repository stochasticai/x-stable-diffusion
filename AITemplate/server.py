
from fastapi import FastAPI
from typing import List
from pydantic import BaseModel
from pipeline_stable_diffusion_ait import StableDiffusionAITPipeline
import torch
from tqdm import tqdm
from PIL import Image
import io
from fastapi import Response

height = 512                        
width = 512
num_inference_steps = 30
guidance_scale = 7.5
batch_size = 1
UNET_INPUTS_CHANNEL=4
torch_device = torch.device("cuda:0")

class Item(BaseModel):
    input_text: List[str]
app = FastAPI()

pipe = StableDiffusionAITPipeline()

@app.post("/predict/")
async def predict(input_api: Item):
    with torch.autocast("cuda"):
        images = pipe(input_api.input_text,num_inference_steps=30)
    if images.ndim == 3:
        images = images[None, ...]
    image = (images[0] * 255).round().astype("uint8")
    image = Image.fromarray(image)
    
    # save image to an in-memory bytes buffer
    with io.BytesIO() as buf:
        image.save(buf, format='PNG')
        im_bytes = buf.getvalue()
    headers = {'Content-Disposition': 'inline; filename="test.png"'}
    return Response(im_bytes, headers=headers, media_type='image/png')
