#!/bin/bash
# P-Tuning

#PATH="/data/donews/wangguisen/chat"
prjPath="/data/wgs/chat"

model_path="./weights/chatglm-6b"
train_path="./data/test.jsonl"
dev_path="./data/test_eval.jsonl"
save_dir="./weights/ptuning_glm_multi"
img_log_dir=${save_dir}"/imglog/"
img_log_name="ChatGLM_P-Tuning_multi"

pre_seq_len=128
batch_size=1
num_train_epochs=500
learning_rate=2e-4
save_freq=500
is_save_frep="no"
logging_steps=100
max_source_seq_len=300
max_target_seq_len=300
#max_source_seq_len=50
#max_target_seq_len=50
quantization_bit=4

#accelerate config default

docker run --rm -d --gpus '"device=2,3"' --name ptuning_glm_multi \
                   --shm-size 15G \
                   -v ${prjPath}:/home \
                   -it wgs-torch:chat \
                   sh -c "accelerate launch --multi_gpu --num_processes=2 --num_machines=1 \
                                                               ./finetune/train_multi_gpu.py \
                                                               --use_ptuning True \
                                                               --model_path ${model_path}  \
                                                               --train_path ${train_path} \
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
                            >>/home/log/ptuning_glm_multi.log 2>>/home/log/ptuning_glm_multi.err"

# --quantization_bit ${quantization_bit} \
# --dynamo_backend
# --mixed_precision=fp16

# CUDA_VISIBLE_DEVICES=2,3
# sh ./dk/train_ptuning_multi.sh
