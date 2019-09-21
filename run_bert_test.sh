#!/usr/bin/env bash
# python ./data/preprocess.py;
export CUDA_VISIBLE_DEVICES=0
python run_bert.py \
--model_type bert \
--model_name_or_path chinese_RoBERTa_zh_Large_pytorch \
--do_train \
--do_eval \
--do_test \
--data_dir ./data/data_0 \
--output_dir ./output_test/fold_0 \
--classifier 'guoday' \
--max_seq_length 10 \
--split_num 4 \
--lstm_hidden_size 512 \
--lstm_layers 1 \
--lstm_dropout 0.1 \
--per_gpu_train_batch_size 6 \
--gradient_accumulation_steps 2 \
--warmup_steps 0 \
--per_gpu_eval_batch_size 32 \
--learning_rate 5e-5 \
--adam_epsilon 1e-6 \
--weight_decay 0 \
--train_steps 200