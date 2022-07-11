#!/bin/bash

python ./test.py \
--model='palnet' \
--dataset=nyu \
--batch_size=4 \
--resume='pretrained_models/weights/PALNet.pth.tar' 2>&1 |tee test_PALNet_NYU.log


