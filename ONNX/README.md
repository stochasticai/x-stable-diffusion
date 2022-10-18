# ONNX Stable Diffusion Example

## 1. Requirements
The Stable Diffusion model will be downloaded from the Hugging Face Hub. That's why before running any of the scripts (`demo.py` or `server.py`) you will have to login in the Hugging Face Hub using the following command:

``` 
huggingface-cli login
```

If not, you can download the same model from the following path: `https://downloads.stochastic.ai/stable-diffusion/onnx_model.zip`

### 1.1. Docker execution
[Install Docker](https://docs.docker.com/engine/install/)


### 1.2. Python execution
[Install Python](https://www.python.org/downloads/) and the required libraries:
```
pip install -r requirements.txt
```

## 2. REST API

### 2.1. Docker execution

1. Build the Docker image
```
docker build --build-arg model_dir_path=/path/to/stable_diffusion/model -f Dockerfile -t stable_diffusion_img .
```

2. Execute the Docker Container
```
sudo docker run --gpus all -p 5000:5000 stable_diffusion_img
```

### 2.2. Python execution

To deploy the Stable Diffusion model as an API, execute the following command:
```
uvicorn server:app --host 0.0.0.0 --port 5000
```

## 3. Demo App

To generate images as a command line tool, execute the following command:
```
python demo.py --prompt "an astronaut riding a horse"
```

Check all the options of the command line tool with `python demo.py --help`
