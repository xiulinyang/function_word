from pathlib import Path
from glob import glob
import json
import os

updated_p ='normalized_blimp'
os.makedirs(updated_p, exist_ok=True)
filter_f = Path(f'blimp/no_function_blimp/principle_A_reconstruction.jsonl').read_text().strip().split('\n')
original_f = [json.loads(x)['original'] for x in filter_f]
all_versions = ['bigram_function_blimp', 'five_function_blimp', 'natural_function_blimp','more_function_blimp','random_function_blimp', 'within_boundary_blimp']
cat = 'blimp/natural_function_blimp/principle_A_reconstruction.jsonl'
f = Path(cat).read_text().strip().split('\n')
cat_name = cat.split('/')[-1]
for v in all_versions:
    os.makedirs(f'{updated_p}/{v}', exist_ok=True)
    with open(f'{updated_p}/{v}/{cat_name}', 'w') as o:
        v_f = Path(f'blimp/{v}/{cat_name}').read_text().strip().split('\n')
        cat_v = [json.loads(x) for x in v_f]
        for x in cat_v:
            if 'natural' in v:
                original = x['sentence_good']
            else:
                original = x['original']
            if original not in original_f:
                print(v)
                print(cat_name)
                print(x)
            else:
                o.write(json.dumps(x))
                o.write('\n')





