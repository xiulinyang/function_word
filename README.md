# causal-lm-training
## Instructions
- Step1: clone necessary repositories and create environments

```bash
# install dependencies for others in another environment
conda create -n lmt python==3.10
git clone https://github.com/xiulinyang/causal-lm-training.git
cd causal-lm-training
pip install -e . --no-dependencies
pip install -r requirements.txt
```

- Step2: put training/dev/test data under the ```data``` folder. They should have the same name as the dataset name. E.g., wiki/train/wiki.txt
- Step3: train models using the following script. Note you can customize training/model/data hyperparameters in ```generate_config.py```
```bash
bash train_model.sh $LANG $vocab_size $tokenizer_type $model_type
```
