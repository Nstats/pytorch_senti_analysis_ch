#!/usr/bin/env bash
# python ./data/preprocess_original_balanced_except_eval.py;
# python ./data/preprocess_original_balanced_except_eval_random_drop.py;
python ./data/preprocess_original_balanced_except_eval_random_drop_v2.py;
export CUDA_VISIBLE_DEVICES=0
for((i=0;i<1;i++));

do
python run_bert_continue.py \
--model_type bert \
--model_name_or_path ./chinese_roberta_wwm_large_pytorch_hit \
--optimizer 'Adam' \
--do_train 'yes' \
--do_eval 'no' \
--do_test 'no' \
--do_label_smoothing 'yes' \
--data_dir ./data/data_$i \
--output_dir ./test1_continue_on_v2_out_random_drop_roberta_large_hit_new_dataloaderv2_del_w_data_2epo_3split_64bs_lr5e-6_guoday_balanced_except_eval/fold_$i \
--classifier 'guoday' \
--max_seq_length 100 \
--split_num 1 \
--lstm_hidden_size 512 \
--lstm_layers 1 \
--dropout 0.1 \
--eval_steps 200 \
--per_gpu_train_batch_size 16 \
--gradient_accumulation_steps 16 \
--warmup_steps 0 \
--per_gpu_eval_batch_size 32 \
--learning_rate 5e-6 \
--adam_epsilon 1e-6 \
--weight_decay 0 \
--train_steps 100;

python run_bert_continue.py \
--model_type bert \
--model_name_or_path ./chinese_roberta_wwm_large_pytorch_hit \
--optimizer 'Adam' \
--do_train 'yes' \
--do_eval 'no' \
--do_test 'no' \
--do_label_smoothing 'yes' \
--data_dir ./data/data_$i \
--output_dir ./test2_continue_on_v2_out_random_drop_roberta_large_hit_new_dataloaderv2_del_w_data_2epo_3split_64bs_lr5e-6_guoday_balanced_except_eval/fold_$i \
--classifier 'guoday' \
--max_seq_length 100 \
--split_num 1 \
--lstm_hidden_size 512 \
--lstm_layers 1 \
--dropout 0.1 \
--eval_steps 200 \
--per_gpu_train_batch_size 16 \
--gradient_accumulation_steps 16 \
--warmup_steps 0 \
--per_gpu_eval_batch_size 32 \
--learning_rate 5e-6 \
--adam_epsilon 1e-6 \
--weight_decay 0 \
--train_steps 100

done

:<<annotation
--optimizer 'RAdam' or 'Adam'
--classifier:'guoday' or 'MLP' or 'GRU_MLP' or 'GRU_highway'
training_steps = epoch*num_training_examples/(per_gpu_train_batch_size/gradient_accumulation_steps)
epoch = training_steps*(per_gpu_train_batch_size/gradient_accumulation_steps)/num_training_examples
python combine.py --model_prefix ./output_RoBERTa_large_3epo/fold_
RoBErTa_large:
8docs/step 1split 512 27g guoday
2docs/step 3split 512 27g guoday
8docs/step 1split 512 25g GRU_MLP
2docs/step 3split 512 19g GRU_MLP
annotation
