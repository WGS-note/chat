#!/bin/bash

cd /data/donews/wangguisen/chat/weights

git lfs clone https://huggingface.co/THUDM/chatglm-6b

git lfs clone https://huggingface.co/fnlp/moss-moon-003-sft-int4


# nohup sh /data/donews/wangguisen/chat/dk/down.sh &