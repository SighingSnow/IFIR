import argparse
from elasticsearch import Elasticsearch

es = Elasticsearch(
    "http://localhost:9200"
)

indices = dict(es.indices.get_alias(index="*"))
target = list(indices.keys())

target = sorted(target)
for x in target:
    print(x)