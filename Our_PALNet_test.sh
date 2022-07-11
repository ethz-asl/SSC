#!/bin/bash

python ./test.py \
--model='palnet_ours' \
--dataset=nyu \
--batch_size=4 \
--resume='pretrained_models/weights/PALNet_ours.pth.tar' 2>&1 |tee test_PALNet_ours_NYU.log


