#!/bin/bash
set -euo pipefail

lang=$1
vocab_size=$2
tokenizer_type=$3
model_type=$4
seed=$5

model_name="$model_type"_"$lang"_"$tokenizer_type"_"$vocab_size"_"$seed"

python generate_config.py \
  --lang "$lang" \
  --tokenizer_type "$tokenizer_type" \
  --vocab "$vocab_size" \
  --model_type "$model_type" \
  --pretokenized_file


python train_tokenizer.py -c configs/$model_type/${lang}_${tokenizer_type}_${vocab_size}_${seed}.json
python save_config.py -c configs/$model_type/${lang}_${tokenizer_type}_${vocab_size}_${seed}.json
python train_clm.py configs/$model_type/${lang}_${tokenizer_type}_${vocab_size}_${seed}.json