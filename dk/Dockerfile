# v100
FROM pytorch/pytorch:1.13.1-cuda11.6-cudnn8-devel

RUN sed -i "s@/archive.ubuntu.com/@/mirrors.aliyun.com/@g" /etc/apt/sources.list && \
	sed -i "s@/security.ubuntu.com/@/mirrors.aliyun.com/@g" /etc/apt/sources.list && \
	apt-get update && \
	apt-get install -y wget git ffmpeg libsm6 libxext6 libgl1 unzip vim

RUN ln -sf /usr/share/zoneinfo/Asia/Shanghai /etc/localtime && echo "Asia/Shanghai" > /etc/timezone

RUN pip install -i https://pypi.tuna.tsinghua.edu.cn/simple numpy scipy ftfy wandb safetensors mediapipe fire OmegaConf matplotlib

RUN pip install -i https://pypi.tuna.tsinghua.edu.cn/simple huggingface_hub==0.11.1 transformers==4.27.1
RUN pip install -i https://pypi.tuna.tsinghua.edu.cn/simple diffusers==0.15.1
RUN pip install -i https://pypi.tuna.tsinghua.edu.cn/simple accelerate==0.17.1 modelcards==0.1.6

RUN pip install -i https://pypi.tuna.tsinghua.edu.cn/simple opencv-python
RUN apt-get install libgl1-mesa-glx -y

RUN pip install -i https://pypi.tuna.tsinghua.edu.cn/simple sentencepiece sacremoses datasets
RUN pip install -i https://pypi.tuna.tsinghua.edu.cn/simple Jinja2 bitsandbytes
RUN pip install -i https://pypi.tuna.tsinghua.edu.cn/simple deepspeed
RUN pip install -i https://pypi.tuna.tsinghua.edu.cn/simple --upgrade torchvision==0.14.1
RUN pip install -i https://pypi.tuna.tsinghua.edu.cn/simple gradio setuptools
RUN pip install -i https://pypi.tuna.tsinghua.edu.cn/simple tensorboard
RUN pip install -i https://pypi.tuna.tsinghua.edu.cn/simple xformers==0.0.16
RUN pip install -i https://pypi.tuna.tsinghua.edu.cn/simple triton==2.0.0.dev20221120

RUN pip install -i https://pypi.tuna.tsinghua.edu.cn/simple protobuf==3.19.3
RUN pip install -i https://pypi.tuna.tsinghua.edu.cn/simple icetk cpm_kernels
RUN pip install -i https://pypi.tuna.tsinghua.edu.cn/simple rouge_chinese nltk jieba
RUN pip install -i https://pypi.tuna.tsinghua.edu.cn/simple streamlit==1.17.0
RUN pip install -i https://pypi.tuna.tsinghua.edu.cn/simple packaging psutil pyyaml
RUN pip install -i https://pypi.tuna.tsinghua.edu.cn/simple peft==0.3.0
WORKDIR /home

## --upgrade
## docker build -t wgs-torch:chat -f ./dk/Dockerfile .
## docker run --rm -v `pwd`:/data -it python:3.7 sh -c "chmod 777 -R /data/*"

# ------------------------------

###FROM wgs-torch:control_diffusion
#FROM wgs-torch:chat
#RUN pip install -i https://pypi.tuna.tsinghua.edu.cn/simple transformers==4.29.1
#
#WORKDIR /home




