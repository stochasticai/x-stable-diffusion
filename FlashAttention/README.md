## Flash-Attention Stable Diffusion Example

### Build Dependencies

Require python 3.9 or python 3.10, Pytorch 1.12.1-cuda11.6.

```
conda create -n diffusion_fa python=3.10
conda activate diffusion_fa
pip install torch==1.12.1+cu116 torchvision==0.13.1+cu116 --extra-index-url https://download.pytorch.org/whl/cu116
pip install git+https://github.com/facebookresearch/xformers@51dd119#egg=xformers
cd diffusers
pip install -e .
```

Install libraries

```
pip install -r requirements.txt
```
### Benchmark

You need to register in HuggingFace hub. Get your access token from [Hugging Face account settings](https://huggingface.co/settings/tokens). Then login using `huggingface-cli login` command.

```
USE_MEMORY_EFFICIENT_ATTENTION=1 python3 demo.py --benchmark
```