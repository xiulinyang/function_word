#!/bin/bash
set -euo pipefail

data_name=$1
seed=$5

model_name="GPT2"_"$data_name"_"$seed"

python generate_config.py \
  --data_name $data_name \
  --model_name $model_name \
  --seed $seed \
  --pretokenized_file


python train_tokenizer.py -c configs/"$model_name".json
python save_config.py -c configs/"$model_name".json
python train_clm.py configs/"$model_name".json