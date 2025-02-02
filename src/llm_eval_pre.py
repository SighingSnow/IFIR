import os
import json
import time
import random
import asyncio
import argparse
import numpy as np
from typing import List, Dict, Any
from tqdm import tqdm
from copy import deepcopy
from typing import Union
from utils.dataset import load_query
from utils.llm_metrics.GEval.prompts import GEvalPrompts
from utils.llm_metrics.evaluator import OpenAIEvaluator, LLamaEvaluator
from utils.constant import get_es, remove_extra_fields
reason_datasets = ['fiqa', 'pm', 'scifact_open', 'aila']
datasets = ['nfcorpus',  'pm', 'scifact_open', 'cds']
record_path = ""
evaluation_path = ""

# we only export ids because of the size of the data
def export_qa_ids(model: str, dataset: str, topk: int = 20, domain: bool = False, hybrid: bool = False, device: str = "cuda:0", ):
    print('-----------------------------------------------------------------------')
    print(f'Exporting results of {model} on {dataset} with topk={topk}')
    
    if os.path.exists(record_path):
        print(f"Results for {model} on {dataset} with topk={topk} already exists. Skipping...")
        return
    ir_data = load_query(dataset)
    db = get_es(model=model, dataset=dataset, domain=domain, device=device)
    # origin
    d_list = []
    for x in tqdm(ir_data):
        query = x['text']
        d = {'_id': x['_id']}
        responses = db.similarity_search_with_score(query=query, k=topk)
        results = [x[0] for x in responses] # get the responses
        query_ids = [{'_id': x.metadata['_id']} for x in results] # get the corpus

        corpus_ids = [c['_id'] for c in x['corpus']] # the ids of the corpus
        d['origin'] = {'query': query, 'corpus': query_ids}
        
        for inst in x['instructions']:
            level = inst['level']
            # w/o instruct
            inst_ids = [corpus_ids[x] for x in inst['rel']]
            # w/ instruct
            results = db.similarity_search(query=query+" "+inst['instruction'], k=topk)
            inst_corpus_ids = [{'_id': x.metadata['_id']} for x in results]
            d[str(level)] = {'query': query+" "+inst['instruction'], 'corpus': inst_corpus_ids}

        d_list.append(d)

    with open(record_path, "w") as f:
        json.dump(d_list, f, indent=4)

def generate_qa_history(model: str, dataset: str, sample_ids: List[Any], topk: int = 20, hybrid: bool = False, domain: bool = False):
    if not os.path.exists(record_path):
        print(f"Results for {model} on {dataset} with topk={topk} does not exist.")
        return
    
    with open(record_path, "r") as f:
        ids = json.load(f)
    
    ids = [x for x in ids if x['_id'] in sample_ids]
    ids = sorted(ids, key=lambda x: x['_id'])

    corpus_dict = {}
    with open(f'datasets/{dataset}/{dataset}-corpus.jsonl','r') as f:
        corpus = [json.loads(x) for x in f.readlines()]
        for x in corpus:
            corpus_dict[x['_id']] = x
        del corpus

    for qa in ids:
        for key, corpus in qa.items():
            if key == '_id':
                continue
            tmp_coprus = []
            for c in corpus['corpus']:
                _id =  c['_id']
                tc = corpus_dict[_id] # text corpus
                corpus_str = tc['title'] + ' ' + tc['text'] if tc.get('title') else tc['text']
                tmp_coprus.append({
                    '_id' : _id,
                    'text' : corpus_str
                })
            qa[key]['corpus'] = tmp_coprus
        yield qa # qa is a dict with keys: origin, '0', '1', '2' ,'3'
                 # each is with a query/instruction and a list of corpus

def measure(evaluator, inst: List[str], levels: List[int], corpus: List[str], dataset: str, topk: int = 20) -> float:
    messages = []
    if dataset in reason_datasets:
        for i in range(min(len(inst), 3)):
            tmp_corpus = corpus[:topk] + corpus[topk*(i+1):topk*(i+2)]
            message = [
               eval("GEvalPrompts."+dataset+str(levels[i]))(instruction=inst[i], corpus=c)
               for c in tmp_corpus
            ]
            messages.extend(message)
    else:
        messages = [ eval("GEvalPrompts."+dataset)(inst[0], c) for c in corpus ]

    formated_responses = evaluator.score(messages)
    
    return formated_responses

def evaluate(evaluator, model: str, dataset: str, topk: int = 20, domain: bool = False, sample: bool = False, hybrid: bool = False):
    # TODO: Add LLM evaluation for the model
    data = load_query(dataset)
    # Sampling logic
    sample_fraction = 5 if dataset != "fiqa" else 10
    if sample:
        data = random.sample(data, len(data) // sample_fraction)
    
    ids = sorted(x['_id'] for x in data)
    to_export = []
    l1, l2, l3 = [], [], []
    
    for x in tqdm(generate_qa_history(model, dataset, ids, topk, hybrid, domain), total=len(data)):
        query_corpus = x['origin']['corpus']
        has_levels = {1: False, 2: False, 3: False}

        # Prepare for level-based corpus construction (reason_datasets specific)
        if dataset in reason_datasets:
            level_data = [deepcopy(x) for _ in range(3)]  # [x1, x2, x3] as a list
            level_data = remove_extra_fields(level_data) # Remove extra fields in the data
            corpus = query_corpus
            inst, levels = [], []

            for level in range(1, 4):
                if x.get(str(level)):
                    level_corpus = x[str(level)]['corpus']
                    has_levels[level] = True
                    corpus += level_corpus
                    inst.append(x[str(level)]['query'])
                    levels.append(level)

            corpus = [item['text'] for item in corpus]
            response = measure(evaluator, inst=inst, levels=levels, corpus=corpus, dataset=dataset, topk=topk)

            true_idx = 0
            for level in range(1, 4):
                if not has_levels[level]:
                    continue
                true_idx += 1
                for idx, c in enumerate(x[str(level)]['corpus']):
                    c['score'] = response[idx + topk * 2 * (true_idx - 1)]['score']
                    c['reason'] = response[idx + topk * 2 * (true_idx - 1)]['reason']
                for idx, c in enumerate(x[str(level)]['corpus']):
                    c['score'] = response[idx + topk * 2 * true_idx - topk]['score']
                    c['reason'] = response[idx + topk * 2 * true_idx - topk]['reason']

            if has_levels[1]: l1.append(level_data[0])
            if has_levels[2]: l2.append(level_data[1])
            if has_levels[3]: l3.append(level_data[2])
        else:
            inst_corpus = x['0']['corpus']
            corpus = query_corpus + inst_corpus
            corpus = [item['text'] for item in corpus]
            response = measure(evaluator, inst=[x['0']['query']], levels=[0], corpus=corpus, dataset=dataset, topk=topk)

            # Update scores and reasons
            for idx, c in enumerate(x['0']['corpus']):
                c['score'] = response[idx + topk]['score']
                c['reason'] = response[idx + topk]['reason']
            for idx, c in enumerate(x['origin']['corpus']):
                c['score'] = response[idx]['score']
                c['reason'] = response[idx]['reason']

        # Store results
        to_export.append(x)

    # Export results
    if dataset not in reason_datasets:
        with open(f"{evaluation_path}/{model}_{dataset}_{topk}_evaluated.json", "w") as f:
            json.dump(to_export, f, indent=4)
    else:
        for i in range(1, 4):
            if locals().get(f"l{i}"):  # Ensure the list is not empty before saving
                with open(f"{evaluation_path}/{model}_{dataset}_{topk}_evaluated_{i}.json", "w") as f:
                    json.dump(locals()[f"l{i}"], f, indent=4)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True) # bm25, contriever, bert
    parser.add_argument("--dataset", type=str, required=True) # fiqa, nfcorpus,
    parser.add_argument("--engine", type=str, default="gpt-4o-mini") # gpt-4o, gpt-4o-mini, gpt-4o-2024-11-20
    parser.add_argument("--device", type=str, default="cuda:0") # device
    parser.add_argument("--topk", type=int, default=20) # topk
    parser.add_argument("--level", type=int, default=0) # the reasoning level(only for fiqa, pm, scifact_open and aila)
    parser.add_argument("--domain", type=bool, default=False)
    parser.add_argument("--sample", type=bool, default=False)
    parser.add_argument("--hybrid", type=bool, default=False)
    args = parser.parse_args()


    if args.engine in ['Llama-3.1-8B-Instruct', 'Llama-3.1-70B-Instruct']:
        evaluator = LLamaEvaluator(model_name=args.engine)
    if args.engine in ['gpt-4o', 'gpt-4o-mini', 'gpt-4o-2024-11-20']:
        evaluator = OpenAIEvaluator(dataset=args.dataset, model_name=args.engine)
    
    
    evaluation_path = f"{args.engine}-hybrid-evaluation" if args.hybrid else f"{args.engine}-evaluation"
    hybrid_path = "hybrid-" if args.hybrid else ""
    record_path  = f"{hybrid_path}qa/{args.model}_{args.dataset}_{args.topk}.json" 
    
    if not os.path.exists(f"{hybrid_path}qa"):
        os.makedirs(f"{hybrid_path}qa")
    
    if not os.path.exists(evaluation_path):
        os.makedirs(evaluation_path)
    
    print("-----------------------------------------------------------------------")
    print(f"Evaluating {args.model} on {args.dataset} with topk={args.topk}")
    export_qa_ids(model=args.model, dataset=args.dataset, device=args.device, topk=args.topk, domain=args.domain, hybrid=args.hybrid)
    evaluate(evaluator=evaluator, model=args.model, dataset=args.dataset, topk=args.topk, domain=args.domain, sample = args.sample, hybrid = args.hybrid)
