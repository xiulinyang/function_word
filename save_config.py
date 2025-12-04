
from clm.utils.tokenizer_and_config import load_config, autoreg_config
import argparse
import pathlib
from transformers import AutoConfig, AutoTokenizer
from transformers import AutoTokenizer
import argparse, json, os, pathlib
from transformers import PreTrainedTokenizerFast
from tokenizers import Tokenizer


def load_json_cfg(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)



def main(args):
    model_cfg = load_json_cfg(args.config)

    # tok_cfg = load_json_cfg(args.config)['tokenizer']

    model_cfg = model_cfg['model']
    hidden_size = model_cfg['n_embd']
    n_head = model_cfg['n_head']
    n_layer = model_cfg['num_layers']
    max_len = model_cfg['n_positions']

    model_name = model_cfg.get('model_name')
    model_type = model_cfg.get('model_type')
    tokenizer = AutoTokenizer.from_pretrained(model_cfg.get('tokenizer_name'))

    vocab = len(tokenizer)
    cfg = autoreg_config(
        model_type,
        model_name,
        tokenizer,
        vocab,
        hidden_size,
        n_head,
        n_layer,
        max_len,
    )

    pathlib.Path(f"models/{model_name}").mkdir(parents=True, exist_ok=True)
    cfg._name_or_path = model_name
    cfg.save_pretrained(f"models/{model_name}")
    print(f'Saved model config to models/{model_name}')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", type=str, help="Path to config file")
    args = parser.parse_args()
    main(args)