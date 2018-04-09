#!/usr/bin/env sh 
CUDA_VISIBLE_DEVICES=2,3  python eval_ijba.py 2>&1 | tee eval_res.txt
