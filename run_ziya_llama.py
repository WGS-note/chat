# coding:utf-8
# @Email: wangguisen@donews.com
# @Time: 2023/5/18 10:41
# @File: run_ziya_llama.py
'''
https://huggingface.co/IDEA-CCNL/Ziya-LLaMA-13B-v1
'''
from transformers import AutoTokenizer
from transformers import LlamaForCausalLM
import torch
import time

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model_path = "./weights/Ziya-LLaMA-13B-v1"

ssstime = time.time()
model = LlamaForCausalLM.from_pretrained(model_path, device_map="auto")
tokenizer = AutoTokenizer.from_pretrained(model_path)
print('load model done: ', time.time() - ssstime)

query = "帮我写一份去西安的旅游计划"

inputs = '<human>:' + query.strip() + '\n<bot>:'

ssstime = time.time()

input_ids = tokenizer(inputs, return_tensors="pt").input_ids.to(device)
generate_ids = model.generate(
    input_ids,
    max_new_tokens=1024,
    do_sample=True,
    top_p=0.85,
    temperature=1.0,
    repetition_penalty=1.,
    eos_token_id=2,
    bos_token_id=1,
    pad_token_id=0)

output = tokenizer.batch_decode(generate_ids)[0]

print(output)
print('inference done: ', time.time() - ssstime)

'''
docker run --rm -it --gpus '"device=3"' --name chat_ziya_demo\
                   --shm-size 15G \
                   -v /data/donews/wangguisen/chat:/home \
                   wgs-torch:chat \
                   sh -c "python /home/run_ziya_llama.py >>/home/log/run_ziya_llama.log 2>>/home/log/run_ziya_llama.err"
'''