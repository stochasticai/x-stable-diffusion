from fastapi import FastAPI
from typing import List, Union
from pydantic import BaseModel
from pipeline_stable_diffusion_ait import StableDiffusionAITPipeline
import torch
from tqdm import tqdm
from PIL import Image
import io
from fastapi import Response

torch_device = torch.device("cuda:0")


class Item(BaseModel):
    prompt: Union[str, List[str]]
    img_height: int = 512
    img_width: int = 512
    num_inference_steps: int = 50
    guidance_scale: float = 7.5


app = FastAPI()

pipe = StableDiffusionAITPipeline()


@app.post("/predict/")
async def predict(input_api: Item):
    with torch.autocast("cuda"):
        images = pipe(
            input_api.prompt,
            height=input_api.img_height,
            width=input_api.img_width,
            num_inference_steps=input_api.num_inference_steps,
            guidance_scale=input_api.guidance_scale,
        )
    if images.ndim == 3:
        images = images[None, ...]
    image = (images[0] * 255).round().astype("uint8")
    image = Image.fromarray(image)

    # save image to an in-memory bytes buffer
    with io.BytesIO() as buf:
        image.save(buf, format="PNG")
        im_bytes = buf.getvalue()
    headers = {"Content-Disposition": 'inline; filename="test.png"'}
    return Response(im_bytes, headers=headers, media_type="image/png")
