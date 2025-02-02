import copy

class Evaluator:
    def __init__(self):
        proto_dict = {'origin': [] , '0': [], '1': [], '2': [], '3': [] }
        self.ndcg_dict = copy.deepcopy(proto_dict)
        self.mrr_dict = copy.deepcopy(proto_dict)
        self.hit_dict = copy.deepcopy(proto_dict)
        self.ap_dict = copy.deepcopy(proto_dict)
        
    def update(self, hit, ap, mrr, ndcg, level):
        self.hit_dict[level].append(hit)
        self.ap_dict[level].append(ap)
        self.mrr_dict[level].append(mrr)
        self.ndcg_dict[level].append(ndcg)
        
    def get_origin(self):
        return {
            "hit": sum(self.hit_dict['origin'])/len(self.hit_dict['origin']),
            "mAP": sum(self.ap_dict['origin'])/len(self.ap_dict['origin']),
            "MRR": sum(self.mrr_dict['origin'])/len(self.mrr_dict['origin']),
            "nDCG": sum(self.ndcg_dict['origin'])/len(self.ndcg_dict['origin']),
        }
        
    def get_inst(self):
        hit, ap, mrr, ndcg = 0, 0, 0, 0
        num_hit, num_ap, num_mrr, num_ndcg = 0, 0, 0, 0
        for i in range(0, 4):
            hit += sum(self.hit_dict[str(i)])
            ap += sum(self.ap_dict[str(i)])
            mrr += sum(self.mrr_dict[str(i)])
            ndcg += sum(self.ndcg_dict[str(i)])
            num_hit += len(self.hit_dict[str(i)])
            num_ap += len(self.ap_dict[str(i)])
            num_mrr += len(self.mrr_dict[str(i)])
            num_ndcg += len(self.ndcg_dict[str(i)])
        return {
            "hit": hit/num_hit,
            "mAP": ap/num_ap,
            "MRR": mrr/num_mrr,
            "nDCG": ndcg/num_ndcg,
        }
    
    def get_level(self, level: int):
        return {
            "hit": sum(self.hit_dict[str(level)])/len(self.hit_dict[str(level)]),
            "mAP": sum(self.ap_dict[str(level)])/len(self.ap_dict[str(level)]),
            "MRR": sum(self.mrr_dict[str(level)])/len(self.mrr_dict[str(level)]),
            "nDCG": sum(self.ndcg_dict[str(level)])/len(self.ndcg_dict[str(level)]),
        }