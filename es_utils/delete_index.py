import argparse
from elasticsearch import Elasticsearch

es = Elasticsearch(
    "http://localhost:9200"
)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", type=str, required=True) # index name
    args = parser.parse_args()
    index = args.index

    if es.indices.exists(index=index):
        es.indices.delete(index=index)
        print(f"Index {index} deleted")
    else:
        print(f"Index {index} not found")