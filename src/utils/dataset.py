import json
from langchain_core.documents import Document

def get_dataset_len(dataset: str):
    with open(f'datasets/{dataset}/{dataset}-corpus.jsonl', 'r') as f:
        data = [json.loads(line) for line in f.readlines()]
    dl = len(data)
    del data
    return dl

def load_corpus(dataset: str, batch_size: int = 256):
    with open(f'datasets/{dataset}/{dataset}-corpus.jsonl', 'r') as f:
        batch = []
        for line in f:
            data = json.loads(line)
            if len(data['text']) == 0:
                continue
            content = ""
            if data.get('title') is not None:
                content = data['title'] + " "
            content += data['text']
            batch.append(Document(page_content=content, metadata={"_id": data['_id']}))
            if len(batch) == batch_size:
                yield batch
                batch = []
        if len(batch) > 0:
            yield batch


def load_query(dataset: str):
    with open(f'datasets/{dataset}/test_data.json', 'r') as f:
        data = json.load(f)
    return data
