#!/bin/bash

cd /data/wgs/chat

docker run --rm -it --gpus '"device=2"' --name chat_moss_demo\
                   --shm-size 15G \
                   -v /data/wgs/chat:/home \
                   wgs-torch:chat \
                   sh -c "python /home/MOSS/moss_cli_demo.py"
