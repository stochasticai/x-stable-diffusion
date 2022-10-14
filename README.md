<p align="center">
  <img src=".github/stochastic_logo_light.svg#gh-light-mode-only" width="600" alt="Stochastic.ai"/>
  <img src=".github/stochastic_logo_dark.svg#gh-dark-mode-only" width="600" alt="Stochastic.ai"/>
</p>

<br>
<br>

# Realtime inference for Stable Diffusion



Stochastic.ai provides this repository to perform fast, almost realtime inference solutions for Stable Diffusion. This repository includes 4 folders corresponding to 4 optimization techiques:

- AITemplate: [Latest optimization framework of Meta](https://github.com/facebookincubator/AITemplate)
- TensorRT: [Nvidia TensorRT framework](https://github.com/NVIDIA/TensorRT)
- Nvfuser: [Nvfuser with Pytorch](https://pytorch.org/blog/introducing-nvfuser-a-deep-learning-compiler-for-pytorch/)
- Flash Attention: [Flash Attention intergration in Xformers](https://github.com/facebookresearch/xformers)

## Benchmark result

Here are some benchmark resuls on 1x40 A100, cuda11.6:

All benchmarks are doing by averaging 50 iterations run:
```
Running args {'max_seq_length': 64, 'num_inference_steps':35, 'image_size':(512,512)}
```
Throughput in sec on 1x40GB gpu - batch size = 1:

| project                | Latency (s) | GPU VRAM  |
| :--------------------- | :---------- | :------   |
| Pytorch           fp16 |  5.77       |  10.3     |
| AITemplate        fp16 |  1.38       |  4.83     |
| TensorRT          fp16 |  1.68       |  8.1      |
| Nvfuser           fp16 |  3.15       |  ---      |
| Flash Attention   fp16 |  2.8        |  ---      |

## Batched Version

| project           \ bs |      1        |     4         |    8          |    16             |   24              | 
| :--------------------- | :------------ | :------------ | :------------ | :---------------- | :---------------- |
| Pytorch           fp16 | 5.77s/10.3GB  | 19s/18.5GB    | 36s/26.7GB    |                   |                   |
| AITemplate        fp16 |               |               |               |                   |                   |
| TensorRT          fp16 |               |               |               |                   |                   |
| Flash Attention   fp16 |               |               |               |                   |                   |


## Reference

- [Diffusers](https://github.com/huggingface/diffusers)
- [AITemplate](https://github.com/facebookincubator/AITemplate)
- [Flash Attention](https://www.photoroom.com/tech/stable-diffusion-100-percent-faster-with-memory-efficient-attention/)
