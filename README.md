<p align="center">
  <img src=".github/stochastic_logo_light.svg#gh-light-mode-only" width="600" alt="Stochastic.ai"/>
  <img src=".github/stochastic_logo_dark.svg#gh-dark-mode-only" width="600" alt="Stochastic.ai"/>
</p>

<br>

# ⚡️ Realtime inference for Stable Diffusion

Stochastic provides this repository to perform fast, almost realtime inference with `Stable Diffusion`. The goal of this repo is to collect and document as many optimization techniques for Stable Diffusion as possible. 

Currently this repository includes 4 optimization techiques with more in the pipeline. Feel free to open a PR to submit a new optimization technique to the folder.

<!-- TOC -->
Table of contents:
- [Optimizations](#-optimizations)
- [Benchmarks](#benchmark-result)
- [Deploy](#deployment)
    - [Quickstart](#-quickstart)
    - [Manual](#manual)
- [Stochastic](#-stochastic)
    - [Features](#features)
- [Reference](#reference)
<!-- /TOC -->

## 🔥 Optimizations

- AITemplate: [Latest optimization framework of Meta](https://github.com/facebookincubator/AITemplate)
- TensorRT: [NVIDIA TensorRT framework](https://github.com/NVIDIA/TensorRT)
- nvFuser: [nvFuser with Pytorch](https://pytorch.org/blog/introducing-nvfuser-a-deep-learning-compiler-for-pytorch/)
- FlashAttention: [FlashAttention intergration in Xformers](https://github.com/facebookresearch/xformers)

## Benchmark result

Here are some benchmark resuls on 1x40 A100, cuda11.6:

All benchmarks are doing by averaging 50 iterations run:
```
Running args {'max_seq_length': 64, 'num_inference_steps':35, 'image_size':(512,512)}
```
Throughput in sec on 1x40GB gpu - batch size = 1:

| Optimization           | Latency (s) | GPU VRAM  |
| :--------------------- |:----------- | :------   |
| PyTorch FP16           | 5.77        |  10.3     |
| AITemplate FP16        | 1.38        |  4.83     |
| TensorRT FP16          | 1.68        |  8.1      |
| nvFuser FP16           | 3.15        |  ---      |
| FlashAttention FP16    | 2.8         |  ---      |

## Batched Version

| Optimization      \ bs |      1        |     4         |    8          |    16             |   24              | 
| :--------------------- | :------------ | :------------ | :------------ | :---------------- | :---------------- |
| PyTorch FP16           | 5.77s/10.3GB  | 19s/18.5GB    | 36s/26.7GB    |                   |                   |
| AITemplate FP16        |               |               |               |                   |                   |
| TensorRT FP16          |               |               |               |                   |                   |
| FlashAttention FP16    |               |               |               |                   |                   |

## 🚀 Quickstart

<>

## Manual deployment

<>

## ✅ Stochastic

Stochastic was founded with a vision to make deep learning optimization and deployment effortless. We make it easy to ship state-of-the-art AI models with production-grade performance.

### Features
- Auto-optimization of deep learning models
- Benchmarking of models and hardware on different evaluation metrics
- Auto-scaling hosted and on-prem accelerated inference for models like BLOOM 176B, Stable Diffusion, GPT-J [Get in touch →](https://stochastic.ai/contact)
- Support for AWS, GCP, Azure clouds and Kubernetes clusters

### [Get early access ->](https://www.stochastic.ai/)


## Reference

- [HuggingFace Diffusers](https://github.com/huggingface/diffusers)
- [AITemplate](https://github.com/facebookincubator/AITemplate)
- [Make stable diffusion up to 100% faster with Memory Efficient Attention](https://www.photoroom.com/tech/stable-diffusion-100-percent-faster-with-memory-efficient-attention/)
- [Making stable diffusion 25% faster using TensorRT](https://www.photoroom.com/tech/stable-diffusion-25-percent-faster-and-save-seconds/)
