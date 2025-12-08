#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from minicons import scorer
from tqdm import tqdm
import pandas as pd
from glob import glob
import argparse

empty_categories = [
    "superlative_quantifiers_1",
    "determiner_noun_agreement_irregular_2",
    "determiner_noun_agreement_with_adj_2",
    "superlative_quantifiers_2",
    "determiner_noun_agreement_with_adj_irregular_2",
    "determiner_noun_agreement_2",
    "matrix_npi",
]

DET = ["the","this","a","an","no","all","another","each","that","any","those","these","both","every","either","neither"]
CCONJ = ["and","but","or","yet"]
SCONJ = ["that","if","although","after","whereas","while","before","as","though","until","because",
         "since","once","whether","unless","albeit","till","whilst"]
AUX = ["will","be","had","were","being","is","would","was","do","could","are","have","been","has",
       "did","should","might","can","does","'s","may","must","ca","'s","am","shall","art","ar","re","ought","need"]
ADP = ["at","in","of","near","for","by","to","with","on","from","behind","into","within","despite","against","as",
       "over","than","during","about","between","among","except","through","around","after","like","off",
       "without","under","before","throughout","unlike","across","toward","along","above","aboard","until",
       "upon","via","beneath","unto","beyond","per","below","amongst","till","beside","amid","onto","towards",
       "underneath","alongside"]

FUNCTION_WORDS = set(DET + CCONJ + SCONJ + AUX + ADP)


def build_function_token_ids(tokenizer):
    func_ids = set()
    vocab_size = tokenizer.vocab_size
    for tid in range(vocab_size):
        txt = tokenizer.decode([tid]).strip().lower()
        if txt in FUNCTION_WORDS:
            func_ids.add(tid)
    return func_ids


def register_function_token_mask_hooks(model, function_token_ids, mask_value=-1e4):
    hooks = []
    ctx = {"func_mask": None}
    function_token_ids = set(function_token_ids)

    def embed_pre_hook(module, args, kwargs):
        input_ids = args[0]
        with torch.no_grad():
            mask = torch.zeros_like(input_ids, dtype=torch.bool)
            for tid in function_token_ids:
                mask |= (input_ids == tid)
        ctx["func_mask"] = mask
        return args, kwargs

    emb_hook = model.transformer.wte.register_forward_pre_hook(
        embed_pre_hook, with_kwargs=True
    )
    hooks.append(emb_hook)

    def make_attn_pre_hook(layer_idx):
        def attn_pre_hook(module, args, kwargs):
            attention_mask = kwargs.get("attention_mask", None)
            func_mask = ctx.get("func_mask", None)

            if func_mask is not None:
                fw_mask = func_mask.to(module.c_attn.weight.device).unsqueeze(1).unsqueeze(1)
                fw_mask = fw_mask * mask_value
                if attention_mask is None:
                    attention_mask_new = fw_mask
                else:
                    attention_mask_new = attention_mask + fw_mask
                kwargs["attention_mask"] = attention_mask_new

            return args, kwargs

        return attn_pre_hook

    for layer_idx, block in enumerate(model.transformer.h):
        attn_mod = block.attn
        h = attn_mod.register_forward_pre_hook(
            make_attn_pre_hook(layer_idx), with_kwargs=True
        )
        hooks.append(h)

    return hooks


def read_data(data_path):
    test_set = {}
    phenomenon_paths = glob(f"{data_path}/*.jsonl")
    for p in tqdm(phenomenon_paths):
        phenomenon_n = p.split("/")[-1].split(".")[0]
        if phenomenon_n in empty_categories:
            continue
        phenomenon = pd.read_json(p, lines=True).to_dict(orient="records")
        sent_pair = [(x["sentence_bad"], x["sentence_good"]) for x in phenomenon]
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
            num_token0 = len(tokenizer.encode(sent[0], add_special_tokens=False))
            num_token1 = len(tokenizer.encode(sent[1], add_special_tokens=False))
            nll0, nll1 = ilm_model.sequence_score(sent, reduction=lambda x: -x.sum(0).item())
            ppl0 = nll0 / num_token0
            ppl1 = nll1 / num_token1
            distribution.append(f"{sent[0]}\t{ppl0}\t{sent[1]}\t{ppl1}")
            if ppl0 > ppl1:
                correct += 1
        acc = correct / len(sents)
        results[phe] = acc
        distributions[phe] = "|||".join(distribution)
        print(phe, acc)
    return results, distributions


if __name__ == "__main__":
    args = argparse.ArgumentParser('eval language models')
    args.add_argument('model_name', type=str, help='model name')
    args = args.parse_args()
    lang_name = args.model_name
    tokenizer = AutoTokenizer.from_pretrained(f"xiulinyang/GPT2_{lang_name}_53", revision="epoch-10")
    model = AutoModelForCausalLM.from_pretrained(f"xiulinyang/GPT2_{lang_name}_53", revision="epoch-10")
    BLIMP_DIR = f"blimp/{lang_name}_blimp/"
    OUT_PREFIX = "blimp_ablation_epoch10_fw_mask"
    os.makedirs(OUT_PREFIX, exist_ok=True)
    test_set = read_data(BLIMP_DIR)
    model.eval()
    func_ids = build_function_token_ids(tokenizer)
    hooks = register_function_token_mask_hooks(model, func_ids)
    ilm_model = scorer.IncrementalLMScorer(model, device="cpu", tokenizer=tokenizer)
    results = {}
    acc, dist = eval_sent_pair(ilm_model, tokenizer, test_set)
    results["epoch-10"] = acc
    pd.DataFrame(results).to_csv(f"{OUT_PREFIX}/results_GPT2_{lang_name}_53_epoch-10.csv")
    for h in hooks:
        h.remove()