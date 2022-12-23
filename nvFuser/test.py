import torch
import argparse
from diffusers import UNet2DConditionModel
from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import AutoencoderKL
from diffusers import LMSDiscreteScheduler
from torch import autocast
from tqdm import tqdm
from time import time

device = torch.device("cuda")
sd_fused = torch.jit.load("unet_jit.pt")
sd_fused = sd_fused.to(device)
tokenizer = CLIPTokenizer.from_pretrained(
    "CompVis/stable-diffusion-v1-4", subfolder="tokenizer", use_auth_token=True
)
prompt = "Super Mario learning to fly in an airport, Painting by Leonardo Da Vinci"
text_input = tokenizer(
    prompt, padding="max_length", max_length=64, truncation=True, return_tensors="pt"
).input_ids.cuda()
uncond_input = tokenizer(
    [""] * 1, padding="max_length", max_length=64, return_tensors="pt"
).input_ids.cuda()
batch_size = 1
img_size = (512, 512)
latents = torch.randn((batch_size, 4, img_size[0] // 8, img_size[1] // 8)).cuda()

for _ in tqdm(range(5)):
    out = sd_fused(text_input, uncond_input, latents)
torch.cuda.synchronize()
start = time.perf_counter()
for i in tqdm(range(100)):
    out = sd_fused(text_input, uncond_input, latents)
torch.cuda.synchronize()
