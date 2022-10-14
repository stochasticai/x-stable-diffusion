## TensorRT Stable Diffusion Example

### Build Dependencies

Install TensorRT 8.4.2.2.4

```
wget wget https://developer.download.nvidia.com/compute/redist/nvidia-tensorrt/nvidia_tensorrt-8.4.2.4-cp39-none-linux_x86_64.whl
pip install nvidia_tensorrt-8.4.2.4-cp39-none-linux_x86_64.whl
```

Install libraries

```
pip install torch==1.12.1+cu116 torchvision==0.13.1+cu116 torchaudio==0.12.1 -f https://download.pytorch.org/whl/torch_stable.html
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

### Deploy as rest-api end-point

You need provide the HuggingFace token in file `server.py`.

```
docker build -t tensorrt_diffusion .
docker run -p 5000:5000 -ti --gpus=all tensorrt_diffusion
```

### Test API

```
python3 client.py
```

Check the resulted image: `output_api.png`