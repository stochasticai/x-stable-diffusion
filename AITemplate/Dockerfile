FROM python:3.9-slim

WORKDIR /code

RUN pip install torch==1.12.1+cu116 torchvision==0.13.1+cu116 torchaudio==0.12.1 -f https://download.pytorch.org/whl/torch_stable.html

COPY requirements.txt /code/requirements.txt

RUN apt-get update && apt-get -y install curl && apt -y install git

RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

RUN git clone --recursive https://github.com/facebookincubator/AITemplate

RUN cd AITemplate/python && python3 setup.py bdist_wheel && pip install dist/*.whl --force-reinstall

COPY . /code/

EXPOSE 5000

CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "5000", "--workers", "1"]