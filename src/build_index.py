import json
import argparse
from tqdm import tqdm
from utils.dataset import load_corpus, get_dataset_len
from utils.constant import get_es

def build_index(model: str,  dataset: str, device: str, batch_size: int = 256, domain: bool = False):
    db = get_es(model, dataset, domain, device)

    total_len = get_dataset_len(dataset) // batch_size + 1
    if get_dataset_len(dataset) % batch_size == 0:
        total_len -= 1
    idx = 0
    for corpus in tqdm(load_corpus(dataset, batch_size), total = total_len):
        db.add_documents(corpus)
        idx = idx+1
    print(f"Indexing {dataset} complete with {model}")

def parse_args(args):
    model = args.model
    dataset = args.dataset.lower()
    batch_size = args.batch_size
    device = args.device
    domain = args.domain
    build_index(model, dataset, device, batch_size, domain)
    return 

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True) # bm25, contriever, bert
    parser.add_argument("--dataset", type=str, required=True) # fiqa, nfcorpus, 
    parser.add_argument("--batch_size", type=int, default=256) # batch size
    parser.add_argument("--device", type=str, default="cuda:0") # device
    parser.add_argument("--domain", type=bool, default=False) # for instructor inst
    args = parser.parse_args()
    parse_args(args)