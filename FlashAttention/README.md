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
pip install torch==1.12.1+cu116 torchvision==0.13.1+cu116 torchaudio==0.12.1 -f https://download.pytorch.org/whl/torch_stable.html
pip install -r requirements.txt
```
### Benchmark

You need to register in HuggingFace hub. Get your access token from [Hugging Face account settings](https://huggingface.co/settings/tokens). Then login using `huggingface-cli login` command.

```
USE_MEMORY_EFFICIENT_ATTENTION=1 python3 demo.py --benchmark
```

### Deploy as rest-api end-point

Requirement: Make sure that you enable Nvidia runtime when building docker image as Xformers requires GPU to turn on some flags.

You need provide the HuggingFace token in file `server.py`.

```
docker build -t fa_diffusion .
docker run -p 5000:5000 -ti --gpus=all fa_diffusion
```

Note: Building Xformers takes about 35 mins - be patient

### Test API

```
python3 client.py
```

Check the resulted image: `output_api.png`