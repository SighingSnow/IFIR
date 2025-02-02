import os
import json
import argparse
from tqdm import tqdm

from utils.dataset import load_query
from utils import metrics
from utils.evaluator import Evaluator
from utils.constant import get_es, get_hybrid_es, dense_search, hybrid_search
reason_datasets = ['fiqa', 'pm', 'scifact_open', 'aila']

def evaluation(model: str, dataset: str, topk: int = 20, domain: bool = False, device: str = "cuda:0", hybrid: bool = False):
    print('-----------------------------------------------------------------------')
    print(f'Evaluation of {model} on {dataset} with topk={topk}')
    evaluator = Evaluator() 
    ir_data = load_query(dataset)
    if hybrid:
        db, bm25_db = get_hybrid_es(model=model, dataset=dataset, domain=domain, device=device)
    else:
        db = get_es(model=model, dataset=dataset, domain=domain, device=device)
    
    inst_num = 0
    for x in tqdm(ir_data):
        query = x['text']
        if hybrid:
            test_ids = hybrid_search(dense_db=db, bm25_db=bm25_db, query=query, topk=topk)
        else:
            test_ids = dense_search(dense_db=db, query=query, topk=topk)

        corpus_ids = [c['_id'] for c in x['corpus']]
        for inst in x['instructions']:
            inst_num += 1
            level = inst['level']
            # w/o instruct
            inst_ids = [corpus_ids[i] for i in inst['rel']]
            hit, ap, mrr, dcg = metrics.to_calculate(ids= test_ids, gids=inst_ids)
            evaluator.update(hit, ap, mrr, dcg, 'origin')
            # w/ instruct
            if hybrid:
                test_ids = hybrid_search(dense_db=db, bm25_db=bm25_db, query=query+" "+inst['instruction'], topk=topk)
            else:
                test_ids = dense_search(dense_db=db, query=query+" "+inst['instruction'], topk=topk)
            hit, ap, mrr, dcg = metrics.to_calculate(ids=test_ids, gids=inst_ids)
            evaluator.update(hit, ap, mrr, dcg, str(level))
    return evaluator


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True) # bm25, contriever, bert
    parser.add_argument("--dataset", type=str, required=True) # fiqa, nfcorpus,
    parser.add_argument("--device", type=str, default="cuda:0") # device
    parser.add_argument("--topk", type=int, default=20) # topk
    parser.add_argument("--level", type=int, default=0) # the reasoning level(only for fiqa, pm, scifact_open and aila)
    parser.add_argument("--domain", type=bool, default=False)
    parser.add_argument("--hybrid", type=bool, default=False)
    args = parser.parse_args()

    hybrid = args.hybrid
    output_path = "hybrid_results" if hybrid else "results"
    
    evaluator = evaluation(model=args.model, dataset=args.dataset, device=args.device, topk=args.topk, domain=args.domain, hybrid = hybrid)
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    
    results = {
        "model": args.model,
        "dataset": args.dataset,
        'top-k': args.topk,
        'domain': args.domain,
        "no_inst" : evaluator.get_origin(),
        "inst" :  evaluator.get_inst()
    }
    if args.dataset in reason_datasets:
        results["level"] = {
            "1": evaluator.get_level(1),
            "2": evaluator.get_level(2),
            "3": evaluator.get_level(3)
        }
    file_name = f'{output_path}/domain-{args.model}-{args.dataset}-{args.topk}.json' if args.domain else f'{output_path}/{args.model}-{args.dataset}-{args.topk}.json'
    with open(file_name, 'w') as f:
        json.dump(results, f , indent = 4)
