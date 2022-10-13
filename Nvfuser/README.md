## Nvfuser fp16 Stable Diffusion Example

### Build Dependencies

Install libraries

```
pip install -r requirements.txt
```

### Convert Unet model to Nvfuser torchscript fp16

You also need to register in HuggingFace hub. Get your access token from [Hugging Face account settings](https://huggingface.co/settings/tokens). Then login using `huggingface-cli login` command.

```
python3 create_unet_nvfuser_model.py
```

Unet Nvfuser fp16 model is store in `./unet_jit.pt`

### Benchmark

```
python3 demo.py --benchmark
```
