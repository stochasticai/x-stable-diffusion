## Nvfuser fp16 Stable Diffusion Example

### Build Dependencies

Install libraries

```
pip install torch==1.12.1+cu116 torchvision==0.13.1+cu116 torchaudio==0.12.1 -f https://download.pytorch.org/whl/torch_stable.html
pip install -r requirements.txt
```

### Convert Unet model to Nvfuser torchscript fp16

You also need to register in HuggingFace hub. Get your access token from [Hugging Face account settings](https://huggingface.co/settings/tokens). Then login using `huggingface-cli login` command.

```
python3 convert_unet_to_tensorrt.py
```

Unet Nvfuser fp16 model is store in `./unet_jit.pt`

### Benchmark

```
python3 demo.py --benchmark
```

### Deploy as rest-api end-point

You need provide the HuggingFace token in file `server.py`.

```
docker build -t nvfuser_diffusion .
docker run -p 5000:5000 -ti --gpus=all nvfuser_diffusion
```

### Test API

```
python3 client.py
```

Check the resulted image: `output_api.png`