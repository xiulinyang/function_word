#!/bin/bash

python src/clm/evaluation/sas_probe.py -d blimp/no_function_blimp -m xiulinyang/GPT2_no_function_42 -o sas_prob/
python src/clm/evaluation/sas_head.py -m xiulinyang/GPT2_no_function_42 -f no_function

python src/clm/evaluation/sas_probe.py -d blimp/random_function_blimp -m xiulinyang/GPT2_random_function_42 -o sas_prob/
python src/clm/evaluation/sas_head.py -m xiulinyang/GPT2_random_function_42 -f random_function


python src/clm/evaluation/sas_probe.py -d blimp/within_boundary_blimp -m xiulinyang/GPT2_within_boundary_42 -o sas_prob/
python src/clm/evaluation/sas_head.py -m xiulinyang/GPT2_within_boundary_42 -f within_boundary
