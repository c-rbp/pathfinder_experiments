#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES=0,1,2,3
python mainclean.py data/train_images_14_SEG.txt data/test_images_14_SEG.txt --print-freq 20 --lr 1e-03 --epochs 20 -b 128 --algo rbp --penalty --name RBP_penalty_hgru_PF14SEG_20ts_15iter_4GPU_128Batch --parallel --log
