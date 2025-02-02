import json
import math
import argparse
import numpy as np
from typing import List
import pandas as pd
import warnings
from utils.dataset import load_query
reason_datasets = ['fiqa','aila', 'pm', 'scifact_open' ]

datasets = ['fiqa', 'scifact_open', 'nfcorpus', 'aila', 'fire', 'pm', 'cds']

evaluation_path = ""

def fmss_avg(qscores: List[int], iscores: List[int], max_score: float = 3.0):
    q_avg = np.mean(qscores)
    i_avg = np.mean(iscores)
    with warnings.catch_warnings(record=True) as w:
        # 设置警告过滤器，捕获警告
        warnings.simplefilter("always")
        if i_avg - q_avg == 0:
            return 0
        if max_score == q_avg:
            return i_avg - q_avg
        result = (i_avg - q_avg) / (max_score - q_avg)
        
    return result

# As LLM can return negative scores, we need to remove the negative scores
def get_problem_ids(pred, level: str = '0'):
    problem_ids = []
    for idx, c in enumerate(pred['origin']['corpus']):
        if c['score'] < 0 or pred[level]['corpus'][idx]['score'] < 0:
            problem_ids.append(idx)
        
        if pred.get('1'):
            if pred['1']['corpus'][idx]['score'] < 0:
                problem_ids.append(idx)
        if pred.get('2'):
            if pred['2']['corpus'][idx]['score'] < 0:
                problem_ids.append(idx)
        if pred.get('3'):
            if pred['3']['corpus'][idx]['score'] < 0:
                problem_ids.append(idx)
    return set(problem_ids)

def evaluate(model, dataset, topk: int = 20, level: int = 3):
    topk = 20
    is_reason = False
    if dataset in reason_datasets:
        is_reason = True
    
    if not is_reason:
        level = 0

    test_data = load_query(dataset)

    if is_reason:
        with open(f'{evaluation_path}/{model}_{dataset}_{topk}_evaluated_{level}.json', 'r') as f:
            predictions = json.load(f)
    else:
        with open(f'{evaluation_path}/{model}_{dataset}_{topk}_evaluated.json', 'r') as f:
            predictions = json.load(f)

    predictions = sorted(predictions, key=lambda x: x['_id'])
    pred_ids = set([x['_id'] for x in predictions])
    test_data = [x for x in test_data if x['_id'] in pred_ids]
    test_data = sorted(test_data, key=lambda x: x['_id'])

    mss_avg = []
    for i, pred in enumerate(predictions):
        qa = test_data[i]
        # First extract ids from the qa
        target_ids = []
        level_match = False
        if not is_reason:
            for cid in qa['instructions'][0]['rel']:
                target_ids.append(qa['corpus'][cid]['_id'])
        else:
            for inst in qa['instructions']:
                if inst['level'] == level:
                    level_match = True
                    for cid in inst['rel']:
                        target_ids.append(qa['corpus'][cid]['_id'])
                    break  
                    
        if not level_match and is_reason:
            continue

        q_dist, i_dist = [], []
        qlen, ilen = 0, 0
        problem_ids = get_problem_ids(pred, str(level))

        for idx, c in enumerate(pred['origin']['corpus']):
            if idx in problem_ids:
                continue
            qlen += 1
            q_dist.append(c['score'])

        for idx, c in enumerate(pred[str(level)]['corpus']):
            if idx in problem_ids:
                continue
            ilen += 1
            i_dist.append(c['score'])

        mss_avg.append(fmss_avg(q_dist, i_dist))

    return np.mean(mss_avg)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, help="Model name")
    parser.add_argument("--dataset", type=str, help="Dataset name")
    parser.add_argument("--topk", type=int, default=20, help="Topk")
    parser.add_argument("--level", type=int, default=3, help="Level")
    parser.add_argument("--evaluator", type=str, default="gpt-4o-mini", help="Evaluator")
    parser.add_argument("--evaluation_path", type=str, default="evaluation", help="Evaluation path")
    args = parser.parse_args()

    evaluation_path = f"{args.evaluator}-{args.evaluation_path}"
    result = evaluate(args.model, args.dataset, args.topk, args.level)
    print(result)
