from huggingface_hub import HfApi
from huggingface_hub.commands.user import _login

_login(HfApi(), token="")
from fastapi import FastAPI
from typing import List, Union
from pydantic import BaseModel
from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import AutoencoderKL
from diffusers import LMSDiscreteScheduler
import torch
from torch import autocast
from tqdm import tqdm
from PIL import Image
import io
from fastapi import Response
import gc

torch_device = torch.device("cuda:0")


class Item(BaseModel):
    prompt: Union[str, List[str]]
    img_height: int = 512
    img_width: int = 512
    num_inference_steps: int = 50
    guidance_scale: float = 7.5


app = FastAPI()

unet = torch.jit.load("./unet_jit.pt").to(torch_device)
vae = AutoencoderKL.from_pretrained(
    "CompVis/stable-diffusion-v1-4", subfolder="vae", use_auth_token=True
).to(torch_device)
tokenizer = CLIPTokenizer.from_pretrained(
    "CompVis/stable-diffusion-v1-4", subfolder="tokenizer", use_auth_token=True
)
text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14").to(
    torch_device
)
scheduler = LMSDiscreteScheduler(
    beta_start=0.00085,
    beta_end=0.012,
    beta_schedule="scaled_linear",
    num_train_timesteps=1000,
)


@app.post("/predict/")
async def predict(input_api: Item):
    batch_size = 1
    text_input = tokenizer(
        input_api.prompt[0],
        padding="max_length",
        max_length=64,
        truncation=True,
        return_tensors="pt",
    )
    with torch.inference_mode(), autocast("cuda"):
        text_embeddings = text_encoder(text_input.input_ids.to(torch_device))[0]
    uncond_input = tokenizer(
        [""] * batch_size, padding="max_length", max_length=64, return_tensors="pt"
    )
    with torch.inference_mode(), autocast("cuda"):
        uncond_embeddings = text_encoder(uncond_input.input_ids.to(torch_device))[0]
    text_embeddings = torch.cat([uncond_embeddings, text_embeddings])

    latents = torch.randn(
        (batch_size, 4, input_api.img_height // 8, input_api.img_width // 8)
    ).to(torch_device)
    scheduler.set_timesteps(input_api.num_inference_steps)

    latents = latents * scheduler.sigmas[0]
    for i, t in tqdm(enumerate(scheduler.timesteps)):
        gc.collect()
        latent_model_input = torch.cat([latents] * 2)
        sigma = scheduler.sigmas[i]
        latent_model_input = latent_model_input / ((sigma**2 + 1) ** 0.5)

        noise_pred = unet(
            latent_model_input, torch.tensor([t]).to(torch_device), text_embeddings
        )[0]
        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + input_api.guidance_scale * (
            noise_pred_text - noise_pred_uncond
        )
        with torch.inference_mode(), autocast("cuda"):
            with torch.no_grad():
                latents = scheduler.step(noise_pred, i, latents)["prev_sample"]

    # # scale and decode the image latents with vae
    latents = 1 / 0.18215 * latents
    with torch.inference_mode(), autocast("cuda"):
        image = vae.decode(latents).sample
    image = (image / 2 + 0.5).clamp(0, 1)
    image = image.detach().cpu().permute(0, 2, 3, 1).numpy()
    images = (image * 255).round().astype("uint8")
    im = Image.fromarray(images[0])

    # save image to an in-memory bytes buffer
    with io.BytesIO() as buf:
        im.save(buf, format="PNG")
        im_bytes = buf.getvalue()
    headers = {"Content-Disposition": 'inline; filename="test.png"'}
    return Response(im_bytes, headers=headers, media_type="image/png")
