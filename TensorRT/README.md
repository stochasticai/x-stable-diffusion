## TensorRT Stable Diffusion Example

### Build Dependencies

Install TensorRT

```
Download TensorRT 8.2 from [Nvidia TensorRT](https://developer.nvidia.com/nvidia-tensorrt-8x-download) and install.
```

Install libraries

```
pip install -r requirements.txt
```

Verify the library versions. We have tested transformers 4.22, diffusers 0.3 and torch 1.12.

### Convert Unet Onnx model to TensorRT model

You need to download Unet onnx model before converting. You can download from [HuggingFace hub](https://huggingface.co/kamalkraj/stable-diffusion-v1-4-onnx/resolve/main/models.tar.gz). Extract tar file and Unet onnx model is stored in `./models/unet/unet.onnx`.

You also need to register in HuggingFace hub. Get your access token from [Hugging Face account settings](https://huggingface.co/settings/tokens). Then login using `huggingface-cli login` command.

```
python3 convert_unet_to_tensorrt.py
```

Unet TensorRT model is store in `./unet.engine`

### Benchmark

```
python3 demo.py --benchmark
```