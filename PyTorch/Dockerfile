FROM python:3.9-slim

# Download the HuggingFace model in your local machine and specify the directory path
ARG model_dir_path

WORKDIR /code
ENV MODEL_DIR_PATH=/code/model
COPY requirements.txt requirements.txt
RUN python -m pip install --upgrade pip && \
    pip install torch==1.12.1+cu116 torchvision==0.13.1+cu116 torchaudio==0.12.1 -f https://download.pytorch.org/whl/torch_stable.html && \
    pip install -r /code/requirements.txt
COPY $model_dir_path /code/model
COPY server.py model.py ./
EXPOSE 5000
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "5000", "--workers", "1"]

# Build Docker image example
# sudo docker build --build-arg model_dir_path=stable_diffusion_torch_model -f Dockerfile -t stable_diffusion_img .

# Run Docker image example
# sudo docker run --gpus all -p 5000:5000 stable_diffusion_img