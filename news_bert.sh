#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=0

python run_bert.py \
--model_type bert \
--model_name_or_path chinese_roberta_wwm_large_pytorch_hit \
--optimizer 'Adam' \
--label_name 'label_usd_1' \
--do_train 'no' \
--do_eval 'no' \
--do_test 'yes' \
--do_label_smoothing 'no' \
--data_dir ./data/news_data \
--output_dir ./out_news_classification_label_usd_1 \
--classifier 'MLP' \
--draw_loss_steps 10 \
--max_seq_length 512 \
--split_num 1 \
--lstm_hidden_size 512 \
--lstm_layers 1 \
--dropout 0.1 \
--eval_steps 10 \
--per_gpu_train_batch_size 8 \
--gradient_accumulation_steps 8 \
--warmup_steps 0 \
--per_gpu_eval_batch_size 32 \
--learning_rate 2e-5 \
--adam_epsilon 1e-6 \
--weight_decay 0 \
--train_steps 500

