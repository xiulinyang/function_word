#!/bin/bash

python benchmark_eval.py xiulinyang/GPT2_no_function_53 blimp/no_function_blimp
python benchmark_eval.py xiulinyang/GPT2_no_function_42 blimp/no_function_blimp
python benchmark_eval.py xiulinyang/GPT2_no_function_67 blimp/no_function_blimp


python benchmark_eval.py xiulinyang/GPT2_five_function_53 blimp/five_function_blimp
python benchmark_eval.py xiulinyang/GPT2_five_function_42 blimp/five_function_blimp
python benchmark_eval.py xiulinyang/GPT2_five_function_67 blimp/five_function_blimp

