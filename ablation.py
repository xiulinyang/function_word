import os
from transformers import GPT2LMHeadModel, AutoTokenizer, AutoModelForCausalLM
from minicons import scorer
from tqdm import tqdm
import pandas as pd
from glob import glob

empty_categories = ['superlative_quantifiers_1', 'determiner_noun_agreement_irregular_2',
                    'determiner_noun_agreement_with_adj_2','superlative_quantifiers_2',
                    'determiner_noun_agreement_with_adj_irregular_2','determiner_noun_agreement_2',
                    'matrix_npi']

NATURAL_FUNCTION_HEADS = [(4, 7)]
def register_head_ablation_hooks(model, heads_to_ablate):
    hooks = []
    layer2heads = {}
    for layer, head in heads_to_ablate:
        layer2heads.setdefault(layer, []).append(head)

    for layer_idx, head_list in layer2heads.items():
        attn_module = model.transformer.h[layer_idx].attn

        def make_hook(head_list):
            def hook(module, inputs, output):
                if isinstance(output, tuple):
                    attn_output = output[0]
                    others = output[1:]
                else:
                    attn_output = output
                    others = ()
                B, T, C = attn_output.shape
                n_heads = module.num_heads
                head_dim = C // n_heads

                attn_4d = attn_output.view(B, T, n_heads, head_dim)
                for h in head_list:
                    attn_4d[:, :, h, :] = 0.0
                attn_output = attn_4d.view(B, T, C)

                if others:
                    return (attn_output,) + others
                else:
                    return attn_output
            return hook

        h = attn_module.register_forward_hook(make_hook(head_list))
        hooks.append(h)

    return hooks

def read_data(data_path):
    test_set = {}
    phenomenon_paths = glob(f'{data_path}/*.jsonl')
    for p in tqdm(phenomenon_paths):
        phenomenon_n = p.split('/')[-1].split('.')[0]
        if phenomenon_n in empty_categories:
            continue
        phenomenon = pd.read_json(p, lines=True).to_dict(orient='records')
        sent_pair = [(x['sentence_bad'], x['sentence_good']) for x in phenomenon]
        test_set[phenomenon_n] = sent_pair
    return test_set

def eval_sent_pair(ilm_model, tokenizer, test_set):
    results = {}
    distributions = {}
    for phe, sents in tqdm(test_set.items()):
        correct = 0
        distribution = []
        for sent in sents:
            sent = list(sent)
            num_token0 = len(tokenizer.encode(sent[0],add_special_tokens=False))
            num_token1 = len(tokenizer.encode(sent[1],add_special_tokens=False))
            nll0, nll1 = ilm_model.sequence_score(sent, reduction=lambda x: -x.sum(0).item())
            ppl0 = nll0/num_token0
            ppl1 = nll1/num_token1
            distribution.append(f'{sent[0]}\t{ppl0}\t{sent[1]}\t{ppl1}')
            if ppl0 > ppl1:
                correct+=1
        acc = correct/len(sents)
        results[phe] = acc
        distributions[phe] = '|||'.join(distribution)
        print(phe, acc)
    return results, distributions



if __name__=='__main__':
    tokenizer = AutoTokenizer.from_pretrained("xiulinyang/GPT2_natural_function_53",revision='epoch-10')
    model = AutoModelForCausalLM.from_pretrained("xiulinyang/GPT2_natural_function_53",revision='epoch-10')
    BLIMP_DIR = "blimp/natural_function_blimp/"
    OUT_PREFIX = "blimp_ablation_epoch10_5head"
    os.makedirs(OUT_PREFIX, exist_ok=True)
    test_set = read_data(BLIMP_DIR)
    model.eval()
    results={}
    hooks = register_head_ablation_hooks(model, [(4, 7),(3,6),(1,5),(0,0),(11,1)])
    ilm_model = scorer.IncrementalLMScorer(model, device='cpu', tokenizer=tokenizer)
    acc, dist = eval_sent_pair(ilm_model, tokenizer,test_set)

    results['epoch-10'] = acc
    pd.DataFrame(results).to_csv(f'{OUT_PREFIX}/results_GPT2_natural_function_53_epoch-10.csv')

    for h in hooks:
        h.remove()