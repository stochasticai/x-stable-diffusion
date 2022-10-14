## AITemplate Stable Diffusion Example

### Build Dependencies

Install AITemplate

```
git clone --recursive https://github.com/facebookincubator/AITemplate
cd python
python setup.py bdist_wheel
pip install dist/*.whl --force-reinstall
```

Install libraries

```
pip install torch==1.12.1+cu116 torchvision==0.13.1+cu116 torchaudio==0.12.1 -f https://download.pytorch.org/whl/torch_stable.html
pip install -r requirements.txt
```

Verify the library versions. We have tested transformers 4.22, diffusers 0.4 and torch 1.12.

### Compile AITemplate models

You need to register in HuggingFace hub. Get your access token from [Hugging Face account settings](https://huggingface.co/settings/tokens). Then login using `huggingface-cli login` command.

```
python3 compile.py
```

Compiled models are store in `./tmp` folder

### Benchmark

```
python3 demo.py --benchmark
```

Check the resulted image: `example_ait.png`

### Deploy as rest-api end-point

```
docker build -t ait_diffusion .
docker run -p 5000:5000 -ti --gpus=all ait_diffusion
```

### Test API

```
python3 client.py
```

Check the resulted image: `output_api.png`