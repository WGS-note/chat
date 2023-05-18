# coding:utf-8
# @Email: wangguisen@donews.com
# @Time: 2023/4/27 10:01
# @File: run_chatglm.py
'''
https://github.com/THUDM/ChatGLM-6B
'''
from transformers import AutoTokenizer, AutoModel

model_path = "./weights/chatglm-6b"

tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
model = AutoModel.from_pretrained(model_path, trust_remote_code=True).half().cuda()

response, history = model.chat(tokenizer, "你好", history=[])
print(response)

response, history = model.chat(tokenizer, "晚上睡不着应该怎么办", history=history)
print(response)

if __name__ == '__main__':
    '''
    docker run --rm -it --gpus '"device=2"' --name chat_glm\
                   --shm-size 15G \
                   -v /data/wgs/chat:/home \
                   wgs-torch:chat \
                   bash
                   
    python /home/run_chatglm.py >>/home/log/run_chatglm.log 2>>/home/log/run_chatglm.err
    
    docker run --rm -it --gpus '"device=2"' --name chat_glm\
                   --shm-size 15G \
                   -v /data/wgs/chat:/home \
                   wgs-torch:chat \
                   sh -c "python /home/run_chatglm.py >>/home/log/run_chatglm.log 2>>/home/log/run_chatglm.err"
    '''

