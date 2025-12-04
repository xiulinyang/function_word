from io import open
from conllu import parse
from pathlib import Path
from tqdm import tqdm
input_path = "long_distance.conll"
sents = Path(input_path).read_text(encoding="utf-8").strip().split("\n\n")
sum_len = 0
for sentence in tqdm(sents):
    sentence = parse(sentence)[0]
    words = [word["form"] for word in sentence]
    sum_len += len(words)

print("average length:", sum_len / len(sents))
print("total sentences:", len(sents))
print("total words:", sum_len)

# DET = ["the","this","a","an","some","no","all","another","each","that","any","those","these","both","every","either","neither"]
# CCONJ = ["and","but","or","yet"]
# SCONJ = ["that","if","although","after","whereas","while","before","as","though","until","because", "since","once","whether","unless","albeit","till","whilst","tho","insofar","cuz"]
# AUX = ["will","be","had","were","being","is","would","was","do","could","are","have","been","has","did","should","might","can","does","'s","may","must","'m","ca","'s","'d","'ll","am","'ve","'re","wilt","shall","hav","art","ve","ar","re","ll","ought","need"]
# ADP = ["at","in","of","near","for","by","to","with","on","from","behind","into","within","despite","against","as","over","than","during","about","between","among","except","through","around","after","like","off","without","under","before","throughout","unlike","across","toward","along","above","aboard","until","upon","via","beneath","unto","beyond","per","below","amongst","thru","till","beside","amid","onto","towards","underneath","alongside"]
# FUNCTION_WORDS = set(DET + CCONJ + SCONJ + AUX + ADP)
# print(len(FUNCTION_WORDS))