import torch
import time
import os
import asyncio
from tqdm import tqdm
from typing import List, Union, Dict
from transformers import AutoTokenizer, AutoModel, AutoConfig, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer

from .openai_utils import aopenai_client, openai_client, generate_from_openai_embeddings_completion
import numpy as np
import tiktoken

def d2list(corpus: Dict[str, str], sep: str = " ") -> List[str]:
    return [(corpus["title"][i] + sep + corpus["text"][i]).strip() if "title" in corpus else corpus["text"][i].strip() for i in range(len(corpus['text']))]

def dl2list(corpus: List[Dict[str, str]], sep: str = " ") -> List[str]:
    return [(doc["title"] + sep + doc["text"]).strip() if "title" in doc else doc["text"].strip() for doc in corpus]

class Contriever:
    def __init__(self, device):
        super().__init__()
        if device.find("cuda") != -1:
            self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device("cpu")
        self.tokenizer = AutoTokenizer.from_pretrained("facebook/contriever")
        self.model = AutoModel.from_pretrained("facebook/contriever")
        self.slide = False
        self.model.to(self.device)
        self.model.eval()
    
    def segment_text(self, text, max_seq_length: int = 512, stride: int = 384):
        tokens = self.tokenizer.tokenize(text) # List[str]
        segments = []
        for i in range(0, len(tokens), stride):
            segment = tokens[i:i + max_seq_length]
            s = self.tokenizer.convert_tokens_to_string(segment)
            if len(s):
                segments.append(s)
        return segments

    def mean_pooling(self, token_embeddings, mask):
        token_embeddings = token_embeddings.masked_fill(~mask[..., None].bool(), 0.)
        sentence_embeddings = token_embeddings.sum(dim=1) / mask.sum(dim=1)[..., None]
        return sentence_embeddings
    
    def encode_corpus(
        self, 
        corpus: Union[List[str], Dict[str, str], List[Dict[str, str]]],
        batch_size: int = 32,
        show_progress_bar: bool = True,
        convert_to_tensor: bool = True, # For BEIR compatibility
        normalize_embeddings: bool = False
    ) -> Union[torch.tensor, np.ndarray]:
        sentence_to_segments = []
        all_segments = []
        final_embeddings = []

        if type(corpus) is dict:
            corpus = d2list(corpus)
        elif type(corpus) is list and type(corpus[0]) is dict:
            corpus = dl2list(corpus)

        for text in corpus:
            segments = self.segment_text(text)
            sentence_to_segments.append((text, len(segments)))
            all_segments.extend(segments)

        iter_range = tqdm(range(0, len(all_segments), batch_size)) if show_progress_bar else range(0, len(all_segments), batch_size)

        with torch.no_grad():
            embeddings_list = None
            for i in iter_range:
                inputs = self.tokenizer(all_segments[i:i+batch_size], max_length=512, padding=True, truncation=True, return_tensors='pt').to(self.device)
                # Compute token embeddings
                outputs = self.model(**inputs)
                embeddings = self.mean_pooling(outputs[0], inputs['attention_mask']) # tensor [batch_size, embedding_dim]
        
                if embeddings_list is None:
                    embeddings_list = embeddings
                else:
                    embeddings_list = torch.vstack((embeddings_list, embeddings))
            sidx = 0
            
            for (t, l) in sentence_to_segments:
                final_embeddings.append(torch.mean(embeddings_list[sidx:sidx+l], axis = 0).cpu().tolist())
                sidx += l

        return torch.tensor(final_embeddings)

    def encode_queries(
        self, 
        texts: List[str],
        batch_size: int = 32,
        show_progress_bar: bool = True,
        convert_to_tensor: bool = True,
        normalize_embeddings: bool = False
    ) -> Union[torch.tensor, np.ndarray]:
        return self.encode_corpus(
            corpus=texts,
            batch_size=batch_size,
            show_progress_bar=show_progress_bar,
            convert_to_tensor=convert_to_tensor
        )
    
class OpenAIEmbedding:
    def __init__(self, model: str = "openai-large"):
        super().__init__()
        # you can choose text-embedding-3-large, text-embedding-3-small
        if model == 'openai-large':
            model = 'text-embedding-3-large'
        elif model == 'openai-small':
            model = 'text-embedding-3-small'
        elif model == 'openai-ada':
            model = 'text-embedding-ada-002'
        else:
            raise ValueError('Not supported openai embedding model')
        self.model = model
        self.tokenizer = tiktoken.encoding_for_model(self.model)
        self.client = aopenai_client()

    def segment_text(self, text, max_seq_length: int = 512, stride: int = 384):
        tokens = self.tokenizer.encode(text) # List[str]
        segments = []

        for i in range(0, len(tokens), stride):
            segment = tokens[i:i + max_seq_length]
            s = self.tokenizer.decode(segment)
            if len(s):
                segments.append(s)
        return segments

    def encode_corpus(
        self, 
        corpus: Union[List[str], Dict[str, str], List[Dict[str, str]]],
        batch_size: int = 32,
        show_progress_bar: bool = True,
        convert_to_tensor: bool = True, # For BEIR compatibility
        normalize_embeddings: bool = False
    ) -> Union[torch.tensor, np.ndarray]:
        sentence_to_segments = []
        all_segments = []
        final_embeddings = []
        
        if type(corpus) is dict:
            corpus = d2list(corpus)
        elif type(corpus) is list and type(corpus[0]) is dict:
            corpus = dl2list(corpus)
        

        for text in corpus:
            segments = self.segment_text(text)
            sentence_to_segments.append((text, len(segments)))
            all_segments.extend(segments)

        embeddings = None
        iter_range = tqdm(range(0, len(all_segments), batch_size)) if show_progress_bar else range(0, len(all_segments), batch_size)
        for i in iter_range:
            embedding = asyncio.run(
                generate_from_openai_embeddings_completion(
                    client = self.client,
                    messages = all_segments[i:i+batch_size],
                    engine_name = self.model
                )
            )
            embedding = np.array(embedding)
            if embeddings is None:
                embeddings = embedding
            else:
                embeddings = np.vstack((embeddings, embedding))
            time.sleep(1) # sleep
        sidx = 0

        for (t, l) in sentence_to_segments:   
            final_embeddings.append(np.mean(embeddings[sidx:sidx+l], axis = 0).tolist())
            sidx += l

        return torch.tensor(final_embeddings)
    
    def encode_queries(
        self, 
        texts: List[str],
        batch_size: int = 32,
        show_progress_bar: bool = True,
        convert_to_tensor: bool = True,
        normalize_embeddings: bool = False
    ) -> Union[torch.tensor, np.ndarray]:
        return self.encode_corpus(
            corpus=texts,
            batch_size=batch_size,
            show_progress_bar=show_progress_bar,
            convert_to_tensor=convert_to_tensor
        )

class ColBERT:
    def __init__(self, device):
        super().__init__()
        if device.find("cuda") != -1:
            self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device("cpu")
        self.tokenizer = AutoTokenizer.from_pretrained("colbert-ir/colbertv2.0")
        self.model = AutoModel.from_pretrained("colbert-ir/colbertv2.0")
        self.slide = False
        self.model.to(self.device)
        self.model.eval()
    
    def segment_text(self, text, max_seq_length: int = 512, stride: int = 384):
        tokens = self.tokenizer.tokenize(text) # List[str]
        segments = []
        for i in range(0, len(tokens), stride):
            segment = tokens[i:i + max_seq_length]
            s = self.tokenizer.convert_tokens_to_string(segment)
            if len(s):
                segments.append(s)
        return segments

    def mean_pooling(self, token_embeddings, mask):
        token_embeddings = token_embeddings.masked_fill(~mask[..., None].bool(), 0.)
        sentence_embeddings = token_embeddings.sum(dim=1) / mask.sum(dim=1)[..., None]
        return sentence_embeddings
    
    def encode_corpus(
        self, 
        corpus: Union[List[str], Dict[str, str], List[Dict[str, str]]],
        batch_size: int = 32,
        show_progress_bar: bool = True,
        convert_to_tensor: bool = True, # For BEIR compatibility
        normalize_embeddings: bool = False
    ) -> Union[torch.tensor, np.ndarray]:
        sentence_to_segments = []
        all_segments = []
        final_embeddings = []

        if type(corpus) is dict:
            corpus = d2list(corpus)
        elif type(corpus) is list and type(corpus[0]) is dict:
            corpus = dl2list(corpus)
        
        for text in corpus:
            segments = self.segment_text(text)
            sentence_to_segments.append((text, len(segments)))
            all_segments.extend(segments)

        iter_range = tqdm(range(0, len(all_segments), batch_size)) if show_progress_bar else range(0, len(all_segments), batch_size)

        with torch.no_grad():
            embeddings_list = None
            for i in iter_range:
                inputs = self.tokenizer(all_segments[i:i+batch_size], max_length=512, padding=True, truncation=True, return_tensors='pt').to(self.device)
                # Compute token embeddings
                outputs = self.model(**inputs)
                embeddings = self.mean_pooling(outputs[0], inputs['attention_mask']) # tensor [batch_size, embedding_dim]
                if embeddings_list is None:
                    embeddings_list = embeddings
                else:
                    embeddings_list = torch.vstack((embeddings_list, embeddings))
            sidx = 0
            
            for (t, l) in sentence_to_segments:
                final_embeddings.append(torch.mean(embeddings_list[sidx:sidx+l], axis = 0).cpu().tolist())
                sidx += l

        return torch.tensor(final_embeddings)

    def encode_queries(
        self, 
        texts: List[str],
        batch_size: int = 32,
        show_progress_bar: bool = True,
        convert_to_tensor: bool = True,
        normalize_embeddings: bool = False
    ) -> Union[torch.tensor, np.ndarray]:
        return self.encode_corpus(
            corpus=texts,
            batch_size=batch_size,
            show_progress_bar=show_progress_bar,
            convert_to_tensor=convert_to_tensor
        )
    
class Instructor:
    def __init__(self, name: str, dataset:str, device: str, domain: bool):
        model_path = 'hkunlp/' + name
        self.model = SentenceTransformer(model_path)
        self.inst = 'Represent the '
        self.domain = domain
        self.sep = ' '
        if dataset == 'fiqa':
            self.dinst = 'Represent the financial'
        elif dataset == 'nfcorpus' or dataset == 'scifact_open':
            self.dinst = 'Represent the science'
        elif dataset == 'cds' or dataset == 'pm':
            self.dinst = 'Represent the biomedical'
        elif dataset == 'aila' or dataset == 'fire':
            self.dinst = 'Represent the legal'
        else:
            raise ValueError('not support dataset')

        if device.find("cuda") != -1:
            self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device("cpu")
    
    def segment_text(self, inst, text, max_seq_length: int = 512, stride: int = 384):
        inst_tokens = self.model.tokenizer.tokenize(inst)
        tokens = self.model.tokenizer.tokenize(text)
        max_seq_length -= len(inst_tokens)
        segments = []
        for i in range(0, len(tokens), stride):
            segment = tokens[i:i + max_seq_length]
            s = self.model.tokenizer.convert_tokens_to_string(segment)
            if len(s):
                segments.append(s)
        return segments
    
    def encode_corpus(self, 
        corpus: Union[List[str],Dict[str, str], List[Dict[str, str]]], 
        batch_size: int = 32, 
        is_query: bool = False, 
        show_progress_bar: bool = True, 
        convert_to_tensor: bool = False,
        normalize_embeddings: bool = False
    ) -> Union[np.ndarray, torch.Tensor]:
        sentence_to_segments = []
        all_segments = []
        final_embeddings = []

        if type(corpus) is dict:
            corpus = d2list(corpus)
        elif type(corpus) is list and type(corpus[0]) is dict:
            corpus = dl2list(corpus)

        if self.domain:
            inst = self.dinst
        else:
            inst = self.inst

        if is_query:
            inst += ' query for retrieve relevant paragraphs: '
        else:
            inst += ' paragraph for retrieval: '

        for text in corpus:
            segments = self.segment_text(inst, text)
            sentence_to_segments.append((text, len(segments)))
            all_segments.extend(segments)


        final_embeddings = []
        sidx = 0
        with torch.no_grad():
            embeddings = self.model.encode(
                all_segments, 
                batch_size=batch_size, 
                prompt=inst, 
                device=self.device,
                show_progress_bar=show_progress_bar,
                convert_to_tensor=False
            )
            for (t,l) in sentence_to_segments:
                final_embeddings.append(np.mean(embeddings[sidx:sidx+l], axis=0).tolist())
                sidx += l
        
        return torch.tensor(final_embeddings)

    def encode_queries(
        self, 
        queries: List[str], 
        batch_size: int = 32, 
        show_progress_bar: bool = True, 
        convert_to_tensor: bool = False,
        normalize_embeddings: bool = False
    ) -> Union[np.ndarray, torch.tensor]:
        return self.encode_corpus(
            corpus=queries, 
            batch_size=batch_size,
            is_query=True,
            show_progress_bar=show_progress_bar,
            convert_to_tensor=False
        )

class GTR:
    def __init__(self, name: str, device: str):
        model_path = 'sentence-transformers/' + name
        self.model = SentenceTransformer(model_path)
    
        if device.find("cuda") != -1:
            self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device("cpu")
    
    def segment_text(self, text, max_seq_length: int = 512, stride: int = 384):
        tokens = self.model.tokenizer.tokenize(text)
        max_seq_length = max_seq_length
        segments = []
        for i in range(0, len(tokens), stride):
            segment = tokens[i:i + max_seq_length]
            s = self.model.tokenizer.convert_tokens_to_string(segment)
            if len(s):
                segments.append(s)
        return segments

    def encode_corpus(
        self, 
        corpus: Union[List[str], Dict[str, str], List[Dict[str, str]]],
        batch_size: int = 32,
        show_progress_bar: bool = True,
        convert_to_tensor: bool = True, # For BEIR compatibility
        normalize_embeddings: bool = False
    ) -> Union[torch.tensor, np.ndarray]:
        sentence_to_segments = []
        all_segments = []
        final_embeddings = []

        if type(corpus) is dict:
            corpus = d2list(corpus)
        elif type(corpus) is list and type(corpus[0]) is dict:
            corpus = dl2list(corpus)

        for text in corpus:
            segments = self.segment_text(text)
            sentence_to_segments.append((text, len(segments)))
            all_segments.extend(segments)


        final_embeddings = []
        sidx = 0
        with torch.no_grad():
            embeddings = self.model.encode(all_segments, device=self.device, batch_size=batch_size, show_progress_bar=show_progress_bar, convert_to_tensor=False)
            for (t,l) in sentence_to_segments:
                final_embeddings.append(np.mean(embeddings[sidx:sidx+l], axis=0).tolist())
                sidx += l
            
        return np.ndarray(final_embeddings)

    def encode_queries(
        self, 
        texts: List[str],
        batch_size: int = 32,
        show_progress_bar: bool = True,
        convert_to_tensor: bool = True,
        normalize_embeddings: bool = False
    ) -> Union[torch.tensor, np.ndarray]:
        return self.encode_corpus(
            corpus=texts,
            batch_size=batch_size,
            show_progress_bar=show_progress_bar,
            convert_to_tensor=convert_to_tensor
        )

