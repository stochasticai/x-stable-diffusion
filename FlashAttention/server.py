from huggingface_hub import HfApi
from huggingface_hub.commands.user import _login

_login(HfApi(), token="")
from fastapi import FastAPI
from typing import List, Union
from pydantic import BaseModel
from diffusers import StableDiffusionPipeline
import torch
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
pipe = StableDiffusionPipeline.from_pretrained(
    "CompVis/stable-diffusion-v1-4",
    revision="fp16",
    torch_dtype=torch.float16,
    use_auth_token=True,
).to("cuda")


@app.post("/predict/")
async def predict(input_api: Item):
    with torch.inference_mode(), torch.autocast("cuda"):
        images = pipe(input_api.prompt)
    im = images.images[0]

    # save image to an in-memory bytes buffer
    with io.BytesIO() as buf:
        im.save(buf, format="PNG")
        im_bytes = buf.getvalue()
    headers = {"Content-Disposition": 'inline; filename="test.png"'}
    return Response(im_bytes, headers=headers, media_type="image/png")
