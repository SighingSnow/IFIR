# This file is a modified version of BEIR examples.
import os
import sys
import json
import argparse
from typing import List
from beir import util, LoggingHandler
from beir.retrieval import models
from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval.evaluation import EvaluateRetrieval
from beir.retrieval.search.dense import DenseRetrievalExactSearch as HNSW
from beir.retrieval.search.dense import HNSWFaissSearch as HNSW
from utils.models import (
    Contriever, ColBERT, GTR, Instructor, OpenAIEmbedding
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
        model = HNSW(
            Contriever(device=device), 
            batch_size=encode_batch_size,
            corpus_chunk_size=corpus_chunk_size
        )
    elif model_name == 'colbert':
        model = HNSW(
            ColBERT(device=device), 
            batch_size=encode_batch_size,
            corpus_chunk_size=corpus_chunk_size
        )
    elif model_name.find('gtr') != -1:
        model = HNSW(
            GTR(name=model_name, device=device), 
            batch_size=encode_batch_size,
            corpus_chunk_size=corpus_chunk_size
        )
    elif model_name.find('instructor') != -1:
        model = HNSW(
            Instructor(name=model_name, dataset=dataset, device=device, domain=domain), 
            batch_size=encode_batch_size,
            corpus_chunk_size=corpus_chunk_size
        )
    elif model_name == 'e5':
        model = HNSW(
            E5(dataset=dataset, device=device, domain=domain),
            batch_size=encode_batch_size,
            corpus_chunk_size=corpus_chunk_size
        )
    elif model_name == 'gritlm':
        model = HNSW(
            GritLM(dataset=dataset, device=device, domain=domain), 
            batch_size=encode_batch_size,
            corpus_chunk_size=corpus_chunk_size
        )
    elif model_name.find('openai') != -1:
        model = HNSW(
            OpenAIEmbedding(name=model_name), 
            batch_size=encode_batch_size,
            corpus_chunk_size=corpus_chunk_size
        )
    elif model_name == 'promptriever':
        model = HNSW(
            Promptriever(device=device), 
            batch_size=encode_batch_size, 
            corpus_chunk_size=corpus_chunk_size
        )
    elif model_name == "nv":
        model = HNSW(
            NVEmbed(device=device),
            batch_size=encode_batch_size,
            corpus_chunk_size=corpus_chunk_size
        )
    return model

def save_index(model, model_name, dataset, domain):
    index_path = f"{model_name}-{dataset}-domain-index" if domain else f"{model_name}-{dataset}"
    ext = "hnsw"
    if not os.path.exists("indices"):
        os.makedirs("indices")
    if not os.path.exists(os.path.join("indices", index_path)):
        model.save("indices", index_path, ext=ext)
    else:
        logging.info(f"Index {index_path} already exists. Skipping save.")
    return

def split_queries(queries):
    queries_dict = {}
    instructions = {}
    for query_id, query in queries.items():
        index = query_id.rfind('_')
        level = str(query_id[index+1])
        if level not in queries_dict:
            queries_dict[level] = {}
        queries_dict[level][query_id] = query
        if level != '0':
            instructions[query_id] = query
    return queries_dict, instructions

#### Evaluate your retrieval using NDCG@k, MAP@K ...
def eval(model, queries, corpus, qrels, score_func: str = "cos_sim", topk: List[int] = [20]):
    retriever = EvaluateRetrieval(model, score_function=score_func)
    results = retriever.retrieve(corpus, queries)
    
    #### Retrieve dense results (format of results is identical to qrels)
    logging.info(f"Retriever evaluation for {topk}")
    ndcg, _map, recall, precision = retriever.evaluate(qrels, results, topk)
    mrr = retriever.evaluate_custom(qrels, results, retriever.k_values, metric="mrr")
    print(f"ndcg: {ndcg}, map: {_map}, recall: {recall}, precision: {precision}")
    return results, {"ndcg": ndcg, "mrr": mrr}

def load_data(dataset: str = "nfcorpus", data_path: str = "datasets"):
    data_folder = f"{data_path}/{dataset}"
    corpus, queries, qrels = GenericDataLoader(
        data_folder=data_folder, 
        query_file=f"{dataset}-query.jsonl",
        qrels_file=f"{data_folder}/{dataset}-qrel.tsv",
        ).load_custom()
    return corpus, queries, qrels

def save_eval_results(args, results, queries):
    if not os.path.exists("qa"):
        os.makedirs("qa")
    # query_id: {doc_id: score}
    d = {}
    for query_id, doc in results.items():
        index = query_id.rfind('_')
        level = query_id[index+1]
        if level == '0':
            level = 'origin'
        true_query_id = query_id[:index]
        if true_query_id not in d:
            d[true_query_id] = {
                '_id' : true_query_id,
            }
        d[true_query_id][level] = {
            'query' : queries[query_id],
            "corpus" : [{"_id" : doc_id} for doc_id in doc.keys()]
        }
    
    filename = f"domain-{args.model}-{args.dataset}-{args.topk}.json" if args.domain else f"{args.model}-{args.dataset}-{args.topk}.json"
    with open(f"qa/{filename}", 'w') as f:
        json.dump(list(d.values()), f, indent=4)
    return

def main(args):
    prefix = f"{args.model}-{args.dataset}-domain" if args.domain else f"{args.model}-{args.dataset}"
    ext = "hnsw"
    index_name = f"{prefix}.{ext}.faiss"
    model = load_model(
            model_name=args.model,
            dataset=args.dataset,
            encode_batch_size=args.batch_size,
            domain=args.domain,
            device=args.device)
    if os.path.exists(os.path.join("indices", index_name)):
        model.load("indices", prefix=prefix, ext=ext)
    
    corpus, queries, qrels = load_data(args.dataset)
    queries_dict, instructions = split_queries(queries) # We first split the queries into different levels
    # results is for final export and record ndcg results
    eval_res = {
        'model' : args.model,
        'dataset' : args.dataset,
        'domain' : args.domain,
        'level' : {}
    }
    for level, query in queries_dict.items():
        print(f"retrieving for level {level}")
        _, to_measure = eval(model, query, corpus, qrels)
        if level == '0':
            eval_res['no_inst'] = to_measure
        else:
            eval_res['level'][str(level)] = to_measure

    print(f"retrieving for all instructions")
    _, to_measure = eval(model, instructions, corpus, qrels)
    eval_res['inst'] = to_measure
    
    filename = f"results/domain-{args.model}-{args.dataset}-{args.topk}.json" if args.domain else f"results/{args.model}-{args.dataset}-{args.topk}.json"
    with open(filename, 'w') as f:
        json.dump(eval_res, f, indent=4)

    results, _ = eval(model, queries, corpus, qrels)
    
    save_index(model=model, model_name = args.model, dataset = args.dataset, domain = args.domain)
    # Save eval results for further analysis
    save_eval_results(args, results, queries) # save retrieval results to evaluation methods

if __name__ == "__main__":
    args = argparse.ArgumentParser(description='Evaluate Dense Retrieval')
    args.add_argument('--dataset', type=str, default='nfcorpus', help='dataset to be used')
    args.add_argument('--model', type=str, default='instructor-base', help='model to be used')
    args.add_argument('--device', type=str, default='cuda:0', help='device to be used')
    args.add_argument('--topk', type=int, default=20, help='top k documents to be retrieved')
    args.add_argument('--batch_size', type=int, default=32, help='batch size for embedding')
    args.add_argument('--scores_func', type=str, default='cos_sim', help='scoring function to be used')
    args.add_argument('--domain', type=bool, default=False, help='include prompt when embedding')
    args = args.parse_args()
    if not os.path.exists('results'):
        os.makedirs('results')
    main(args)