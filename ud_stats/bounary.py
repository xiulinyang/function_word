from conllu import parse
from glob import glob
from tqdm import tqdm
from pathlib import Path
open_class = {'ADJ', 'ADV', 'INTJ', 'NOUN', 'PROPN', 'VERB','NUM'}
closed_class = {'ADP', 'AUX', 'CCONJ', 'DET', 'SCONJ','PRON'}
other = {'PUNCT', 'SYM', 'X'}

uds = sorted(glob('/Users/xiulinyang/Downloads/ud-treebanks-v2.17/*'))
with open('close_vs_open_count.tsv' ,'w') as co:
    for l in tqdm(uds):
        name = l.split('/')[-1]
        lang = l.split('/')[-1].split('-')[0][3:]
        c_w = []
        o_w = []
        ps = glob(f'{l}/*.conllu')
        for con in ps:
            texts = Path(con).read_text().strip().split('\n\n')
            for sent in texts:
                sent_lines = [x for x in sent.split('\n') if not x.startswith('#')]
                closed_words = [x.split()[1] for x in sent_lines if x.split()[3] in closed_class]
                open_words = [x.split()[1] for x in sent_lines if x.split()[3] in open_class]

                c_w.extend(closed_words)
                o_w.extend(open_words)
            ratio= len(set(c_w))/(len(set(o_w))+len(set(c_w)))
            c_ratio =1-ratio
            co.write(f'{name}\t{lang}\t{len(set(c_w))}\t{len(set(o_w))}\t{len(set(o_w))+len(set(c_w))}\t{ratio}\t{1-ratio}\n')
            # print(f'{lang}\t{len(set(c_w))}\t{len(set(o_w))}\t{len(set(o_w))+len(set(c_w))}\t{ratio}\n')
