#!/usr/bin/env bash
# python pregenerate_lmtraining_data.py \
# --train_corpus ./data/data_for_lmfinetune.txt \
# --bert_model bert-large-uncased \
# --do_lower_case \
# --output_dir lm_training/ \
# --epochs_to_generate 3 \
# --max_seq_len 256;

python lm_finetune_on_pregenerated.py \
--pregenerated_data lm_training/ \
--bert_model bert-large-uncased \
--do_lower_case \
--output_dir finetuned_lm/ \
--epochs 3
