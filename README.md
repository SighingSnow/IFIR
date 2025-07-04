# IFIR: A Comprehensive Benchmark for Evaluating Instruction-Following in Expert-Domain Information Retrieval
[![IFIR Paper (NAACL 2025)](https://img.shields.io/badge/arXiv-2503.04644-red)](https://arxiv.org/abs/2503.04644)
[![HuggingFace](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Datasets-blue)](https://huggingface.co/datasets/songtingyu/IFIR/)

### Table of Contents
- [Updates](#updates)
- [Overview](#overview)
- [Mteb](#mteb)
- [Preparation](#preparation)
- [Experiments](#experiments)
- [Citation](#citation)



<a name="updates"></a>
## :fire: Updates
- [06/2025] :tada: We are delighted that IFIR has been integrated into the [MTEB benchmark](https://mteb.org/benchmarks/ifir) and is now available for evaluation.
- [03/2025] Paper **IFIR: A Comprehensive Benchmark for Evaluating Instruction-Following in Expert-Domain Information Retrieval** is released. 
- [03/2025] Datasets for IFIR are available on HuggingFace. [![HuggingFace](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Datasets-blue)](https://huggingface.co/datasets/songtingyu/IFIR/)


  
<a name="preparation"></a>
## :mega: Overview
<center>
  <img src="./assets/motivation.png" width="50%">
</center>

<a name="mteb"></a>
## :rocket: MTEB

IFIR has been integrated into the [MTEB benchmark](https://mteb.org/benchmarks/ifir) v2.0.0 branch and is now available for evaluation. You can use the following code to evaluate your model on IFIR:

```python
import mteb
from sentence_transformers import SentenceTransformer
import torch

# Define the sentence-transformers model name
def main(args):
    model_name = args.model_name
    model = SentenceTransformer(
         model_name,
         model_kwargs={'torch_dtype': torch.float16},
         device=args.device,
         trust_remote_code=True
    )
    tasks = mteb.get_tasks(tasks=["IFIRFiQA", "IFIRFire", "IFIRAila",  "IFIRNFCorpus", "IFIRCds", "IFIRPm", "IFIRScifact"])
    evaluation = mteb.MTEB(tasks=tasks)
    model_name = args.model_name.split("/")[-1]
    results = evaluation.run(model, output_folder=f"results/{model_name}", encode_kwargs={"batch_size":2}, co2_tracker=False)

if __name__ == "__main__":
    from argparse import ArgumentParser
    parser = ArgumentParser(description="Run MTEB evaluation with a specified model.")
    parser.add_argument("--model_name", type=str, help="Name of the sentence-transformers model to evaluate.")
    parser.add_argument("--device", type=str, help="device")
    args = parser.parse_args()
    main(args)
```

<a name="preparation"></a>
## :hammer: Preparation

### Enviroment
```sh
pip install -r requirements.txt
```

### Dataset
Download from huggingface and extract it to `datasets` folder in the root dir within following structure:

You can use the following script to download the dataset:
```bash
git clone https://huggingface.co/datasets/songtingyu/IFIR/
mv IFIR datasets
```

```sh
datasets
├── fiqa
│   ├── fiqa-query.json
│   ├── fiqa-corpus.jsonl
│   ├── fiqa-qrel.jsonl
│   ├── test_data.json
├── aila
│   ├── aila-query.json
│   ├── aila-corpus.jsonl
│   ├── aila-qrel.jsonl
│   ├── test_data.json
├── fire
│   ├── fire-query.json
│   ├── fire-corpus.jsonl
│   ├── fire-qrel.jsonl
│   ├── test_data.json
├── ...
```

<a name="experiments"></a>
## :computer: Experiments
### Experiments
If you want to use BEIR for evaluation, please follow the README in the `beir` folder. And to reproduce the results in the paper, please follow the following steps:

1. Start the elasticsearch
```sh
wget -O elasticsearch-8.11.1.tar.gz https://artifacts.elastic.co/downloads/elasticsearch/elasticsearch-8.11.1-linux-x86_64.tar.gz
tar -xzf elasticsearch-8.11.1.tar.gz
cd elasticsearch-8.11.1
./bin/elasticsearch
```

2. Embed the dataset and evaluate the model
```sh
sh scripts/embed.sh [model_name] [dataset_name] [batch_size] [device] # embed the dataset for the model
sh scripts/eval.sh [model_name] [dataset_name] [top_k] [device] # calculate the statistical metrics
```

For InstFol metrics, please follow the following steps:
```
export OPENAI_API_KEY=[your_openai_api_key]
sh scripts/llm_eval_pre.sh [model_name] [dataset_name] [top_k] [device] # prepare the intermediate results for InstFol metrics
sh scripts/llm_eval.sh [model_name] [dataset_name] [top_k] [device] # calculate the InstFol metrics
```

The statistical metrics will be saved in `results` folder in the root dir. 
And the InstFol metrics will be saved in `llm_results` folder in the root dir. Additionally, `qa` folder and `evaluation` folder will be created in root dir to store the intermediate results. 

<a name="citation"></a>
## :page_facing_up: Citation
If you find this repository helpful, feel free to cite our paper:

```bibtex
@misc{song2025ifir,
      title={IFIR: A Comprehensive Benchmark for Evaluating Instruction-Following in Expert-Domain Information Retrieval}, 
      author={Tingyu Song and Guo Gan and Mingsheng Shang and Yilun Zhao},
      year={2025},
      eprint={2503.04644},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2503.04644}, 
}
```