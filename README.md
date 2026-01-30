# Function Words as Statistical Cues for Language Learning

This repository contains code and resources for the paper:  
[*From Linear Input to Hierarchical Structure: Function Words as Statistical Cues for Language Learning*](https://arxiv.org/abs/2601.21191)  
by Xiulin Yang, Heidi Getz, and Ethan Gotlieb Wilcox

## Repository Structure

- `blimp/`: Contains manipulated BLiMP minimal pairs for each language; also includes code to generate the corresponding datasets (Wikipedia data can be downloaded from [Hugging Face](https://huggingface.co/datasets/wikimedia/wikipedia))
- `ablation_experiments/`: Contains code for ablation experiments in Experiment 3
- `r_scripts/`: Contains R scripts for data analysis and visualization
- `test_set_preference/`: Contains code for test set preference experiments (not included in the paper; exploratory analysis)
- `src/`: Contains source code for training language models
- `ud_stats/`: Contains code to extract function word statistics from UD treebanks
- `scripts/`: Contains bash scripts to run experiments
- `results/` and `overall_results/`: Contains results from experiments reported in the paper

## To replicate experiments in the paper, please follow these steps:

### 1. Clone this repository and install required packages

```bash
git clone https://github.com/xiulinyang/function_word
cd function_word

# Create and activate a virtual environment (optional but recommended)
conda create -n function_word_env python=3.11
conda activate function_word_env

# Install required packages
pip install -r requirements.txt
pip install -e . --no-dependencies
```
### 2. Train language models
```bash
bash script/train_model.sh $language $seed
```

### 3. Evaluate models on BLiMP minimal pairs
```bash
bash eval.sh # replace the model name and minimal pairs correspondingly in the script
```

### 4. Run ablation experiments
```bash
# Attention probing
bash scripts/sas_probing.sh

# Masking function words
python ablation_experiments/ablation_mask.py <model_name> <seed>

# Extract function heads
python ablation_experiments/ablation_function_head.py
# removing function words
# you can use the same eval.sh but replace the minimal pairs with blimp_no_function 

```

## Contact
For questions or issues, please open an issue on GitHub or contact Xiulin Yang (xiulin.yang.compling@gmail.com)
