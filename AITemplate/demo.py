#  Copyright (c) Meta Platforms, Inc. and affiliates.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#
import click
import torch

from aitemplate.testing.benchmark_pt import benchmark_torch_function
from pipeline_stable_diffusion_ait import StableDiffusionAITPipeline
from PIL import Image


@click.command()
@click.option("--prompt", default="A vision of paradise, Unreal Engine", help="prompt")
@click.option(
    "--benchmark", type=bool, default=False, help="run stable diffusion e2e benchmark"
)
@click.option(
    "--batch_size", type=int, default=1, help="batch size"
)
def run(prompt, benchmark, batch_size):
    pipe = StableDiffusionAITPipeline()
    height = 512
    width = 512
    num_inference_steps = 50
    with torch.autocast("cuda"):
        images = pipe([prompt]*batch_size)
        if benchmark:
            t = benchmark_torch_function(10, pipe, [prompt]*batch_size,height,width,num_inference_steps)
            print(f"sd e2e: {t} ms")
    
    images = (images * 255).round().astype("uint8")
    pil_images = [Image.fromarray(image) for image in images]
    pil_images[0].save("example_ait.png")


if __name__ == "__main__":
    run()
