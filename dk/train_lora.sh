#!/bin/bash
# LoRA Finetune

prjPath="/data/donews/wangguisen/chat"
#prjPath="/data/wgs/chat"

model_path="./weights/chatglm-6b"
train_path="./data/test.jsonl"
dev_path="./data/test_eval.jsonl"
save_dir="./weights/lora_glm"
img_log_dir=${save_dir}"/imglog/"
img_log_name="ChatGLM_LoRA"

lora_rank=4
batch_size=2
num_train_epochs=250
learning_rate=2e-4
save_freq=500
is_save_frep="no"
logging_steps=100
# 300
max_source_seq_len=230
max_target_seq_len=230

docker run --rm -d --gpus '"device=2"' --name lora_glm \
                   --shm-size 15G \
                   -v ${prjPath}:/home \
                   -it wgs-torch:chat \
                   sh -c "python ./finetune/train.py \
                            --use_lora True \
                            --model_path ${model_path}  \
                            --train_path ${train_path}  \
                            --dev_path ${dev_path} \
                            --lora_rank ${lora_rank} \
                            --batch_size ${batch_size} \
                            --num_train_epochs ${num_train_epochs} \
                            --save_freq ${save_freq} \
                            --is_save_frep ${is_save_frep} \
                            --learning_rate ${learning_rate} \
                            --logging_steps ${logging_steps} \
                            --max_source_seq_len ${max_source_seq_len} \
                            --max_target_seq_len ${max_target_seq_len} \
                            --save_dir ${save_dir} \
                            --img_log_dir ${img_log_dir} \
                            --img_log_name ${img_log_name} \
                            --device cuda:0 \
                            >>/home/log/lora_glm.log 2>>/home/log/lora_glm.err"

# lora_rank 4

# sh ./dk/train_lora.sh

#docker run --rm --gpus '"device=3"' --name ddd \
#                   --shm-size 15G \
#                   -v /data/donews/wangguisen/chat:/home \
#                   -it wgs-torch:chat \
#                   bash
