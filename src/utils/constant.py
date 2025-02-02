from langchain_elasticsearch import ElasticsearchStore
from utils.embeddings import Contriever, OpenAIEmbedding, Instructor, ColBERT, GTR
from utils.inst_llms import Promptriever, NVEmbed, E5, GritLM

def remove_extra_fields(data_list):
    for idx, data in enumerate(data_list):
        level = idx + 1
        if level == 1:
            data.pop('2', None)
            data.pop('3', None)
        if level == 2:
            data.pop('3', None)
            data.pop('1', None)
        if level == 3:
            data.pop('1', None)
            data.pop('2', None)
    return data_list

def get_es(model: str, dataset: str, domain: bool, device: str):
    if model == "bm25":
        return ElasticsearchStore(
            index_name="bm25_"+dataset,
            es_url="http://localhost:9200",
            strategy=ElasticsearchStore.BM25RetrievalStrategy()
        )
    elif model == "contriever":
        embedding = Contriever(device)
    elif model.find("openai") != -1:
        embedding = OpenAIEmbedding(model)
    elif model.find('instructor') != -1:
        embedding = Instructor(model, dataset, device, domain)
    elif model == 'colbert':
        embedding = ColBERT(device)
    elif model.find('gtr') != -1:
        embedding = GTR(model, device)
    elif model == 'e5':
        embedding = E5(dataset, device, domain)
    elif model == 'gritlm':
        embedding = GritLM(dataset, device, domain)
    elif model.find('promptriever') != -1:
        embedding = Promptriever(model, device)
    elif model.find('nv') != -1:
        embedding = NVEmbed(device)
    else:
        raise ValueError("Invalid model")

    if domain and model.find('instructor') != -1:
        index_name = f"{model}_{dataset}_d"
    else:
        index_name = f"{model}_{dataset}"

    return ElasticsearchStore(
            es_url="http://localhost:9200",
            index_name=index_name,
            distance_strategy="COSINE",
            embedding=embedding
        )
    
def get_hybrid_es(model: str, dataset: str, domain: bool, device: str):
    dense_db = get_es(model, dataset, domain, device)
    bm25_db = ElasticsearchStore(
        es_url="http://localhost:9200",
        index_name="bm25_"+dataset,
        strategy=ElasticsearchStore.BM25RetrievalStrategy()
    )
    return dense_db, bm25_db

def dense_search(dense_db, query, topk):
    responses = dense_db.similarity_search_with_score(query=query, k=topk)
    results = [x[0] for x in responses]
    test_ids = [x.metadata['_id'] for x in results]
    return test_ids

def hybrid_search(dense_db, bm25_db, query, topk):
    # DPR
    dense_responses = dense_db.similarity_search_with_score(query=query, k=topk)
    dense_results = [x[0] for x in dense_responses]    
    dense_ids = [x.metadata['_id'] for x in dense_results]
    
    # BM25
    bm25_responses = bm25_db.similarity_search_with_score(query=query, k=topk)
    bm25_results = [x[0] for x in bm25_responses]    
    bm25_ids = [x.metadata['_id'] for x in bm25_results]
    
    # RRF
    k = 60 
    hybrid_dict = {}
    for idx, did in enumerate(dense_ids):
        hybrid_dict[did] = 1.0 / (k+idx+1)
    
    for idx, bid in enumerate(bm25_ids):
        if hybrid_dict.get(bid) is not None:
            hybrid_dict[bid] += 0.3 / (k+idx+1)
        else:
            hybrid_dict[bid] = 0.3 / (k+idx+1)
    
    # sort hybrid_dict according to the value of hybrid_dict
    sorted_hybrid= sorted(hybrid_dict.items(), key=lambda x: x[1], reverse=True)
    # return ids
    return [x[0] for x in sorted_hybrid][:topk]