from pathlib import Path
from glob import glob
from tqdm import tqdm
for f in tqdm(glob('/Users/xiulinyang/Desktop/conll/data/natural_function/*.txt')):
    lines = Path(f).read_text(encoding='utf-8').strip().split('\n')
    lower_lines = [line.lower() for line in lines]
    new_f = f+'.lower'
    Path(new_f).write_text('\n'.join(lower_lines), encoding='utf-8')