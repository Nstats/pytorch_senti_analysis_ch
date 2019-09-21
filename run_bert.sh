#!/usr/bin/env bash
python ./data/preprocess.py;
export CUDA_VISIBLE_DEVICES=0
for((i=0;i<5;i++));

do
python run_bert.py \
--model_type bert \
--model_name_or_path chinese_RoBERTa_zh_Large_pytorch \
--do_train \
--do_eval \
--do_test \
--data_dir ./data/data_$i \
--output_dir ./output_RoBERTa_large_10epo_3split/fold_$i \
--classifier 'guoday' \
--max_seq_length 512 \
--split_num 3 \
--lstm_hidden_size 512 \
--lstm_layers 1 \
--lstm_dropout 0.1 \
--eval_steps 200 \
--per_gpu_train_batch_size 64 \
--gradient_accumulation_steps 32 \
--warmup_steps 500 \
--per_gpu_eval_batch_size 32 \
--learning_rate 5e-5 \
--adam_epsilon 1e-6 \
--weight_decay 0 \
--train_steps 30000

done

:<<annotation
--classifier:'guoday' or 'MLP' or 'GRU_MLP'
training_steps = epoch*num_training_examples/(per_gpu_train_batch_size/gradient_accumulation_steps)
epoch = training_steps*(per_gpu_train_batch_size/gradient_accumulation_steps)/num_training_examples
python combine.py --model_prefix ./output_RoBERTa_large_3epo/fold_
annotation
