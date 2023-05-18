# coding:utf-8
# @Email: wangguisen@donews.com
# @Time: 2023/4/27 10:14
# @File: chatglm_cli.py
'''
命令行 chatglm 对话

https://github.com/THUDM/ChatGLM-6B
'''
from transformers import AutoTokenizer, AutoModel
from transformers.generation.utils import logger
from huggingface_hub import snapshot_download
import warnings
import time, os
import platform

logger.setLevel("ERROR")
warnings.filterwarnings("ignore")

# model_path = "/home/weights/chatglm-6b"
# model_path = "/home/weights/ptuning_glm/model_best"
model_path = "/home/weights/lora_glm/model_best"
if not os.path.exists(model_path):
    # model_path = snapshot_download(model_path)
    raise ValueError("model path is not exist")

print("Waiting for all devices to be ready, it may take a few minutes...")

tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
model = AutoModel.from_pretrained(model_path, trust_remote_code=True).half().cuda()

print('model load done...')

def clear():
    os.system('cls' if platform.system() == 'Windows' else 'clear')

def main():
    '''

    :return:
    '''
    print("欢迎使用 ChatGLM-6B 人工智能助手！输入内容即可进行对话。输入 clear 以清空对话历史，输入 stop 以终止对话。")
    print("您想体验【单轮对话】还是【多轮对话】？")
    query = input("请输入（1 代表单轮，2 代表多轮）：")
    if query == '1':
        tmpbool = True
    elif query == '2':
        tmpbool = False
    else:
        raise ValueError('error')

    history = []
    while True:
        query = input("<|Human|>: ")
        if query.strip() == "stop":
            print("期待下次和您见面，再见")
            break
        if query.strip() == "clear":
            clear()
            history.clear()
            continue

        prompt = query

        print('> begin generated, wait a moment...')
        starttime__ = time.time()

        if tmpbool:
            response, history = model.chat(tokenizer, prompt, history=[])
        else:
            response, history = model.chat(tokenizer, prompt, history=history)

        print('<|ChatGLM-6B|>:', response)
        print('> use time: {}'.format(time.time() - starttime__))
        print('\n')

if __name__ == '__main__':
    '''
    python /home/chatglm_cli.py
    
    docker run --rm -it --gpus '"device=2"' --name chat_glm_demo\
                   --shm-size 15G \
                   -v /data/wgs/chat:/home \
                   wgs-torch:chat \
                   sh -c "python /home/chatglm_cli.py"
    
    docker run --rm -it --gpus '"device=3"' --name chat_glm_demo\
                   --shm-size 15G \
                   -v /data/donews/wangguisen/chat:/home \
                   wgs-torch:chat \
                   sh -c "python /home/chatglm_cli.py"
    '''
    main()

