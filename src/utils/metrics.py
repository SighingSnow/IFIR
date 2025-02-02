import math
import numpy as np
from typing import List

def to_calculate(ids: List[str], gids: List[str]):
    hit = hitrate(ids, gids)
    map = mAP(ids, gids)
    mrr = MRR(ids, gids)
    dcg = nDCG(ids, gids)
    return hit, map, mrr, dcg

def hitrate(ids: List[str], gids: List[str]):
    for x in ids:
        if x in gids:
            return 1
    return 0

# ids: predicted ids
# gid: golden ids
def mAP(ids: List[str], gids: List[str]):
    gid_set = set(gids)
    total_rel = 0
    precision = 0

    for i, id in enumerate(ids, 1):
        if id in gids:
            total_rel += 1
            p_at_i = total_rel / i
            precision += p_at_i
    
    ap = precision / len(gids) if len(gid_set) > 0 else 0
    return ap

def MRR(ids: List[str], gids: List[str]):
    def reciprocal_rank(scores):
        for rank, rel in enumerate(scores):
            if rel == 1:  # 找到第一个相关文档
                return 1 / (rank + 1)
        return 0  # 如果没有相关文档

    relevant_scores = [1 if x in gids else 0 for x in ids]
    
    return reciprocal_rank(relevant_scores)

def nDCG(ids: List[str], gids: List[str]):
    def dcg(scores):
        return sum(
            rel / math.log2(rank+2) for rank, rel in enumerate(scores)
        )

    gids_set = set(gids)
    relevant_scores = [1 if x in gids else 0 for x in ids]
    ideal_scores = sorted(relevant_scores, reverse=True)

    dcg_retrieved = dcg(relevant_scores)
    dcg_ideal = dcg(ideal_scores)

    return dcg_retrieved / dcg_ideal if dcg_ideal > 0 else 0