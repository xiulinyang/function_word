#!/bin/bash

bash benchmark_eval.py xiulinyang/GPT2_natural_function_53 no_function --best
bash benchmark_eval.py xiulinyang/GPT2_natural_function_53 five_function --best
bash benchmark_eval.py xiulinyang/GPT2_natural_function_53 more_function --best
bash benchmark_eval.py xiulinyang/GPT2_natural_function_53 within_boundary --best
bash benchmark_eval.py xiulinyang/GPT2_natural_function_53 random_function --best
bash benchmark_eval.py xiulinyang/GPT2_natural_function_53 bigram_function --best