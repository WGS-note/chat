#!/bin/bash
# P-Tuning

#PATH="/data/donews/wangguisen/chat"
prjPath="/data/wgs/chat"

model_path="./weights/chatglm-6b"
train_path="./data/test.jsonl"
dev_path="./data/test_eval.jsonl"
save_dir="./weights/ptuning_glm"
img_log_dir=${save_dir}"/imglog/"
img_log_name="ChatGLM_P-Tuning"

pre_seq_len=128
batch_size=1
num_train_epochs=500
learning_rate=2e-4
save_freq=500
is_save_frep="no"
logging_steps=100
#max_source_seq_len=400
#max_target_seq_len=300
max_source_seq_len=300
max_target_seq_len=300
quantization_bit=4

docker run --rm -d --gpus '"device=2"' --name ptuning_glm \
                   --shm-size 15G \
                   -v ${prjPath}:/home \
                   -it wgs-torch:chat \
                   sh -c "python ./finetune/train.py \
                             --use_ptuning True \
                             --model_path ${model_path}  \
                             --train_path ${train_path}  \
                             --dev_path ${dev_path} \
                             --pre_seq_len ${pre_seq_len} \
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
                             --quantization_bit ${quantization_bit} \
                             --device cuda:0 \
                             >>/home/log/ptuning_glm.log 2>>/home/log/ptuning_glm.err"

# --quantization_bit ${quantization_bit} \

# save_freq
# logging_steps

# img_log_dir：损失图

# docker run --rm -v `pwd`:/data -it python:3.7 sh -c "chmod 777 -R /data/*"
# sh ./dk/train_ptuning.sh