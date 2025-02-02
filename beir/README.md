# IFIR-Eval with BEIR

This repository contains the beir implementation to evaluate IFIR-Eval. We strongly recommend to use with faiss because some embeddings take a lot of time to build indices. And within the faiss, we can reuse the indices.

## Installation
```sh
pip install -r requirements.txt
```

## Usage
### Evaluate with analytic metrics
```sh
ln -s ../datasets datasets # link the datasets folder to current directory
python eval_with_faiss.py --model [model_name] --dataset [dataset_name] --topk [topk]
```
A folder named `results` will be created in the current directory. The results will be saved in the `results` folder.
And a folder named `qa` will be created in the current directory. 

### Evaluate with InstFol metrics
As we have build the `qa` folder in the previous step, you can use following command to evaluate the InstFol metrics.
```sh
cp ../src/llm_eval_pre.py llm_eval_pre.py 
cp ../src/llm_eval.py llm_eval.py
python llm_eval_pre.py --model [model_name] --dataset [dataset_name] --topk [topk]
python llm_eval.py --model [model_name] --dataset [dataset_name] --topk [topk]
```