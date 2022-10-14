
FROM nvidia/cuda:11.6.0-devel-ubuntu20.04

RUN apt-get update && apt-get install --no-install-recommends -y curl && apt-get -y install git

ENV CONDA_AUTO_UPDATE_CONDA=false \
    PATH=/opt/miniconda/bin:$PATH
RUN curl -sLo ~/miniconda.sh https://repo.anaconda.com/miniconda/Miniconda3-py39_4.12.0-Linux-x86_64.sh \
    && chmod +x ~/miniconda.sh \
    && ~/miniconda.sh -b -p /opt/miniconda \
    && rm ~/miniconda.sh \
    && sed -i "$ a PATH=/opt/miniconda/bin:\$PATH" /etc/environment

RUN python3 -m pip --no-cache-dir install --upgrade pip

WORKDIR /code

RUN pip install torch==1.12.1+cu116 torchvision==0.13.1+cu116 -f https://download.pytorch.org/whl/torch_stable.html

COPY requirements.txt /code/requirements.txt

RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

RUN apt-get update && apt-get -y install wget

RUN pip install git+https://github.com/facebookresearch/xformers@51dd119#egg=xformers

COPY . /code/

RUN cd diffusers && pip install -e .

EXPOSE 5000

ENV USE_MEMORY_EFFICIENT_ATTENTION=1

CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "5000", "--workers", "1"]