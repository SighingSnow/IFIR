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
from beir.retrieval.search.lexical import BM25Search as BM25
import logging
import pathlib, os
import random
from elasticsearch import Elasticsearch

#### Just some code to print debug information to stdout
logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])
#### /print debug information to stdout

def load_model(model_name: str = "contriever",batch_size: int = 32, dataset: str = "cds", domain: bool = False):
    index = f"{model_name}-{dataset}-domain-index" if domain else f"{model_name}-{dataset}"
    host = "localhost"
    
    if dataset == 'aila' or dataset == 'fire' or dataset == "nfcorpus":
        number_of_shards = 1
    else:
        number_of_shards = 5
    
    es = Elasticsearch("http://localhost:9200")
    # if es.indices.exists(index):
    #     return BM25(index_name=index, hostname=host, initialize=False, batch_size= number_of_shards=number_of_shards)
    # else:
    return BM25(
        index_name=index,
        hostname=host, 
        initialize=True, 
        batch_size=batch_size, 
        number_of_shards=number_of_shards,
        sleep_for=4
    )

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
def eval(model, queries, corpus, qrels, topk: List[int] = [20]):
    retriever = EvaluateRetrieval(model)
    results = retriever.retrieve(corpus, queries)
    
    #### Retrieve dense results (format of results is identical to qrels)
    logging.info(f"Retriever evaluation for {topk}")
    ndcg, _map, recall, precision = retriever.evaluate(qrels, results, topk)
    print(f"ndcg: {ndcg}, map: {_map}, recall: {recall}, precision: {precision}")
    return results, {"ndcg": ndcg}

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
    model = load_model(
        model_name=args.model,
        dataset=args.dataset,
        domain=args.domain,
        batch_size=args.batch_size
    )
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
        _, ndcg = eval(model, query, corpus, qrels)
        if level == '0':
            eval_res['no_inst'] = ndcg
        else:
            eval_res['level'][str(level)] = ndcg

    print(f"retrieving for all instructions")
    _, ndcg = eval(model, instructions, corpus, qrels)
    eval_res['inst'] = ndcg
    
    filename = f"results/domain-{args.model}-{args.dataset}-{args.topk}.json" if args.domain else f"results/{args.model}-{args.dataset}-{args.topk}.json"
    with open(filename, 'w') as f:
        json.dump(eval_res, f, indent=4)

    results, _ = eval(model, queries, corpus, qrels)
    
    # Save eval results for further analysis
    save_eval_results(args, results, queries) # save retrieval results to evaluation methods

if __name__ == "__main__":
    args = argparse.ArgumentParser(description='Evaluate Dense Retrieval')
    args.add_argument('--dataset', type=str, default='nfcorpus', help='dataset to be used')
    args.add_argument('--model', type=str, default='bm25', help='model to be used')
    args.add_argument('--device', type=str, default='cuda:0', help='device to be used')
    args.add_argument('--topk', type=int, default=20, help='top k documents to be retrieved')
    args.add_argument('--domain', type=bool, default=False, help='include prompt when embedding')
    args.add_argument('--batch_size', type=int, default=512, help='batch size for embedding')
    args = args.parse_args()
    if not os.path.exists('results'):
        os.makedirs('results')
    main(args)