#!/bin/bash

python src/clm/evaluation/sas_probe.py -d blimp/five_function_blimp -m xiulinyang/GPT2_five_function_53 -o sas_prob/
python src/clm/evaluation/sas_head.py -m xiulinyang/GPT2_five_function_53 -f five_function


python src/clm/evaluation/sas_probe.py -d blimp/natural_function_blimp -m xiulinyang/GPT2_natural_function_53 -o sas_prob/
python src/clm/evaluation/sas_head.py -m xiulinyang/GPT2_natural_function_53 -f natural_function
