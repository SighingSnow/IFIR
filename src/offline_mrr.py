import json
import copy
from utils.metrics import MRR, nDCG
from utils.dataset import load_query
models = [
    'bm25', 'contriever', 'colbert', 'gtr-t5-base', 'gtr-t5-large', 'gtr-t5-xl', 
    'instructor-base', 'instructor-large', 'instructor-xl', 
    'openai-small', 'openai-large' ,
    'e5', 'gritlm', "promptriever", "nv" 
]


datasets = [
    'fiqa', 'scifact_open', 'nfcorpus', 'aila', 'fire', 'pm', 'cds', 
]

reason_datasets = [
    'fiqa', 'aila', 'pm', 'scifact_open' 
]

record_dict = { 'origin': [] , '0': [], '1': [], '2': [], '3': [] }

mode = "hybrid"
if mode != "":
    models = ['gtr-t5-xl', 'instructor-xl', 'gritlm', 'promptriever']
    model_names = ["GTR-xl", "Instructor-xl", "GritLM-7B", "Promptriever-7B"]
for model in models:
    for d in datasets:
        local_record = copy.deepcopy(record_dict)
        if mode == "hybrid":
            file_name = f"{mode}-qa/{model}_{d}_20.json"
        else:
            file_name = f"qa/{model}-{d}-20.json"
        
        with open(f"{file_name}", "r") as f:
            data = json.load(f)
        
        test_data = load_query(d)
            
        for idx, td in enumerate(test_data):
            to_test = data[idx]
            corpus_ids = [c['_id'] for c in td['corpus']]
        
            for inst in td['instructions']:
                level = inst['level']
                inst_ids = [corpus_ids[i] for i in inst['rel']]
                
                origin_test_ids = [c for c in to_test['origin']['corpus']]
                origin_mrr = MRR(origin_test_ids, inst_ids )
                local_record['origin'].append(origin_mrr)
                try:
                    test_ids = [c for c in to_test[str(level)]['corpus']]
                except:
                    test_ids = []
                mrr = MRR(test_ids, inst_ids )
                local_record[str(level)].append(mrr)
        if mode == "hybrid":
            results_folder = f"{mode}_results/"
        else:
            results_folder = f"results/"
        with open(f"{results_folder}/{model}-{d}-20.json", 'r') as f:
            results = json.load(f)
        
        mrr_inst, total_inst = 0.0, 0
        for i in range(0, 4):
            mrr_inst += sum(local_record[str(i)])
            total_inst += len(local_record[str(i)])
        mrr_inst = mrr_inst / total_inst
        
        results['inst']['MRR'] = mrr_inst
        for i in range(1, 4):
            if results.get('level') is None:
                continue
            if results.get('level').get(str(i)) is None:
                continue
            results['level'][str(i)]['MRR'] = sum(local_record[str(i)])/ len(local_record[str(i)])
        
        with open(f"{results_folder}/{model}-{d}-20.json", 'w') as f:
            json.dump(results, f, indent=4)            