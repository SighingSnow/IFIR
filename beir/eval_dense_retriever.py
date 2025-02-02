import os
import sys
import argparse
from typing import List
from beir import util, LoggingHandler
from beir.retrieval import models
from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval.evaluation import EvaluateRetrieval
from beir.retrieval.search.dense import DenseRetrievalExactSearch as DRES
from utils.models import (
    Contriever, ColBERT, GTR, Instructor,OpenAIEmbedding
)
from utils.inst_llms import E5, GritLM, Promptriever, NVEmbed
import logging
import pathlib, os
import random

#### Just some code to print debug information to stdout
logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])
#### /print debug information to stdout

def load_model(model_name: str = "contriever", dataset: str = "cds", encode_batch_size: int = 32, domain: bool = False, device: str = "cuda:0"):
    corpus_chunk_size = 512*99999
    if model_name == 'contriever':
        model = DRES(
            Contriever(device=device), 
            batch_size=encode_batch_size,
            corpus_chunk_size=corpus_chunk_size
        )
    elif model_name == 'colbert':
        model = DRES(
            ColBERT(device=device), 
            batch_size=encode_batch_size,
            corpus_chunk_size=corpus_chunk_size
        )
    elif model_name.find('gtr') != -1:
        model = DRES(
            GTR(name=model_name, device=device), 
            batch_size=encode_batch_size,
            corpus_chunk_size=corpus_chunk_size
        )
    elif model_name.find('instructor') != -1:
        model = DRES(
            Instructor(name=model_name, dataset=dataset, device=device, domain=domain), 
            batch_size=encode_batch_size,
            corpus_chunk_size=corpus_chunk_size
        )
    elif model_name == 'e5':
        model = DRES(
            E5(dataset=dataset, device=device, domain=domain),
            batch_size=encode_batch_size,
            corpus_chunk_size=corpus_chunk_size
        )
    elif model_name == 'gritlm':
        model = DRES(
            GritLM(dataset=dataset, device=device, domain=domain), 
            batch_size=encode_batch_size,
            corpus_chunk_size=corpus_chunk_size
        )
    elif model_name.find('openai') != -1:
        model = DRES(
            OpenAIEmbedding(name=model_name), 
            batch_size=encode_batch_size,
            corpus_chunk_size=corpus_chunk_size
        )
    elif model_name == 'promptriever':
        model = DRES(
            Promptriever(device=device), 
            batch_size=encode_batch_size, 
            corpus_chunk_size=corpus_chunk_size
        )
    elif model_name == "nv":
        model = DRES(
            NVEmbed(device=device),
            batch_size=encode_batch_size,
            corpus_chunk_size=corpus_chunk_size
        )
    return model

#### Evaluate your retrieval using NDCG@k, MAP@K ...
def eval(model, queries, corpus, qrels, score_func: str = "cos_sim", topk: List[int] = [20] ):
    retriever = EvaluateRetrieval(model, score_function=score_func)
    results = retriever.retrieve(corpus, queries)
    #### Retrieve dense results (format of results is identical to qrels)
    logging.info(f"Retriever evaluation for {topk}")
    ndcg, _map, recall, precision = retriever.evaluate(qrels, results, topk)
    print(f"ndcg: {ndcg}, map: {_map}, recall: {recall}, precision: {precision}")

def load_data(dataset: str = "nfcorpus", data_path: str = "datasets"):
    data_folder = f"{data_path}/{dataset}"
    corpus, queries, qrels = GenericDataLoader(
        data_folder=data_folder, 
        query_file=f"{dataset}-query.jsonl",
        qrels_file=f"{data_folder}/{dataset}-qrel.tsv",
        ).load_custom()
    return corpus, queries, qrels

def main(args):
    model = load_model(
        model_name=args.model,
        dataset=args.dataset,
        encode_batch_size=args.batch_size,
        domain=args.domain,
        device=args.device)
    corpus, queries, qrels = load_data(args.dataset)
    eval(model, queries, corpus, qrels)

if __name__ == "__main__":
    args = argparse.ArgumentParser(description='Evaluate Dense Retrieval')
    args.add_argument('--dataset', type=str, default='nfcorpus', help='dataset to be used')
    args.add_argument('--model', type=str, default='instructor-base', help='model to be used')
    args.add_argument('--device', type=str, default='cuda:0', help='device to be used')
    # args.add_argument('--topk', type=int, default=20, help='top k documents to be retrieved')
    args.add_argument('--batch_size', type=int, default=32, help='batch size for embedding')
    args.add_argument('--scores_func', type=str, default='cos_sim', help='scoring function to be used')
    args.add_argument('--domain', type=bool, default=False, help='include prompt when embedding')

    args = args.parse_args()
    main(args)