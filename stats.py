from pathlib import Path
from tqdm import tqdm
from collections import Counter
from glob import glob
import json
# blimps = glob('blimp/*.jsonl')
# for f in tqdm(blimps):
#     data = []
#     f = Path(f).read_text().strip().split('\n')
#     for line in f:
#         data.append(json.loads(line))
#
#     all_sents = []
#     for d in data:
#         if 'some' in d['sentence_good'].split() or 'any' in d['sentence_good'].split():
#             print(d['sentence_good'])





DET = ["the","this","a","an","no","all","another","each","that","any","those","these","both","every","either","neither"]
CCONJ = ["and","but","or","yet"]
SCONJ = ["that","if","although","after","whereas","while","before","as","though","until","because", "since","once","whether","unless","albeit","till","whilst"]
AUX = ["will","be","had","were","being","is","would","was","do","could","are","have","been","has","did","should","might","can","does","'s","may","must","ca","'s","am","shall","art","ar","re","ought","need"]
ADP = ["at","in","of","near","for","by","to","with","on","from","behind","into","within","despite","against","as","over","than","during","about","between","among","except","through","around","after","like","off","without","under","before","throughout","unlike","across","toward","along","above","aboard","until","upon","via","beneath","unto","beyond","per","below","amongst","till","beside","amid","onto","towards","underneath","alongside"]

train = Path('/Users/xiulinyang/Desktop/conll/data/natural_function/train.txt').read_text().strip().split('\n')
all_words = []
for sent in tqdm(train):
    words = sent.strip().split()
    all_words.extend(words)

most_freq = Counter(all_words)
function_words = set(DET + CCONJ + SCONJ + AUX + ADP)


func_word_freq = sum([freq for word, freq in most_freq.items() if word in function_words and word in most_freq])

print(func_word_freq/len(all_words))
for w in function_words:
    print(w, most_freq[w])

