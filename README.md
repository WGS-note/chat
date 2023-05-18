# 开源 LLMs 部署及微调（持续更新）

目前支持的开源 LLMs 部署搭建：

+ MOSS
+ ChatGLM

微调 LLMs：

+ 微调 ChatGLM
  + P-tuning（单卡、多卡）
  + Lora（单卡、多卡）

注意：

+ 133 路径为：/data/wgs/chat
+ v100 路径为：/data/donews/wangguisen/chat



# Dockerfile

./dk/Dockerfile



# MOSS

+ https://github.com/OpenLMLab/MOSS
+ https://huggingface.co/fnlp/moss-moon-003-sft-plugin-int4

```shell
git clone https://github.com/OpenLMLab/MOSS.git

git clone https://github.com/fpgaminer/GPTQ-triton.git
cd GPTQ-triton
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple .

cd ./weights
git lfs clone https://huggingface.co/fnlp/moss-moon-003-sft-int4
```

+ 命令行对话运行：

```shell
docker run --rm -it --gpus '"device=2"' --name chat_moss_demo\
                   --shm-size 15G \
                   -v /data/wgs/chat:/home \
                   wgs-torch:chat \
                   sh -c "python /home/MOSS/moss_cli_demo.py"
```

+ 推理运行：

```shell
docker run --rm -it --gpus '"device=2"' --name chat_moss\
                   --shm-size 15G \
                   -v /data/wgs/chat:/home \
                   wgs-torch:chat \
                   sh -c "python /home/run_moss.py >>/home/log/run_moss.log 2>>/home/log/run_moss.err"
```

+ 例子见：./doc/moss_test.md



# ChatGLM

+ https://github.com/THUDM/ChatGLM-6B

```shell
cd ./weights
git lfs clone https://huggingface.co/THUDM/chatglm-6b
```

+ 命令行对话运行：

```shell
docker run --rm -it --gpus '"device=2"' --name chat_glm_demo\
                   --shm-size 15G \
                   -v /data/wgs/chat:/home \
                   wgs-torch:chat \
                   sh -c "python /home/chatglm_cli.py"
```

+ 推理运行：

```shell
docker run --rm -it --gpus '"device=2"' --name chat_glm\
                   --shm-size 15G \
                   -v /data/wgs/chat:/home \
                   wgs-torch:chat \
                   sh -c "python /home/run_chatglm.py >>/home/log/run_chatglm.log 2>>/home/log/run_chatglm.err"
```





# LLaMA-Alpaca

+ https://github.com/ymcui/Chinese-LLaMA-Alpaca





#  Ziya-LLaMA-13B-v1

+ https://huggingface.co/IDEA-CCNL/Ziya-LLaMA-13B-v1

```shell
cd ./weights
git lfs clone https://huggingface.co/IDEA-CCNL/Ziya-LLaMA-13B-v1
```

+ 命令行对话运行：

```shell
```

+ 推理运行：

```shell
docker run --rm -it --gpus '"device=3"' --name chat_ziya_demo\
                   --shm-size 15G \
                   -v /data/donews/wangguisen/chat:/home \
                   wgs-torch:chat \
                   sh -c "python /home/run_ziya_llama.py >>/home/log/run_ziya_llama.log 2>>/home/log/run_ziya_llama.err"
```



# Finetune

见：./finetune/README.md

+ finetune ChatGLM
  + p-tuning（单卡、多卡）
  + Lora（单卡、多卡）





# Tools

欢迎关注我的公众号：

![](./examples/tmp.jpg)

