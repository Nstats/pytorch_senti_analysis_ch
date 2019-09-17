#!/usr/bin/env bash
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
--output_dir ./output_RoBERTa_large/fold_$i \
--max_seq_length 512 \
--split_num 3 \
--lstm_hidden_size 512 \
--lstm_layers 1 \
--lstm_dropout 0.1 \
--eval_steps 200 \
--per_gpu_train_batch_size 64 \
--gradient_accumulation_steps 32 \
--warmup_steps 1000 \
--per_gpu_eval_batch_size 32 \
--learning_rate 5e-5 \
--adam_epsilon 1e-6 \
--weight_decay 0 \
--train_steps 20000

done

:<<annotation
epoch=10 for Roberta_large and 20 for BERT_base
training_steps=epoch*num_training_examples/(per_gpu_train_batch_size*gradient_accumulation_steps)
annotation
