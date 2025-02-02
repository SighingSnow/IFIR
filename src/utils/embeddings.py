import torch
import time
import os
import asyncio
from typing import List
from langchain_core.embeddings import Embeddings
from transformers import AutoTokenizer, AutoModel, AutoConfig, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer
from .openai_utils import aopenai_client, openai_client, generate_from_openai_embeddings_completion
from gritlm import GritLM as GGritLM
import numpy as np
import tiktoken
# import logging
# logging.basicConfig(level=logging.DEBUG)

class DensePassageretriever(Embeddings):
    def __init__(self):
        pass

class Contriever(Embeddings):
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
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        sentence_to_segments = []
        all_segments = []
        final_embeddings = []
        
        for text in texts:
            segments = self.segment_text(text)
            sentence_to_segments.append((text, len(segments)))
            all_segments.extend(segments)
        
        with torch.no_grad():
            inputs = self.tokenizer(all_segments, padding=True, truncation=True, return_tensors='pt').to(self.device)
            # Compute token embeddings
            outputs = self.model(**inputs)
            embeddings = self.mean_pooling(outputs[0], inputs['attention_mask'])
            sidx = 0
            
            for (t, l) in sentence_to_segments:
                
                final_embeddings.append(torch.mean(embeddings[sidx:sidx+l], axis = 0).cpu().tolist())
                sidx += l

        
        return final_embeddings

    def embed_query(self, text: str) -> List[float]:
        return self.embed_documents([text])[0]
    
class OpenAIEmbedding(Embeddings):
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

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        sentence_to_segments = []
        all_segments = []
        final_embeddings = []
        
        for text in texts:
            segments = self.segment_text(text)
            sentence_to_segments.append((text, len(segments)))
            all_segments.extend(segments)
        
        embeddings = None
        for i in range(0, len(all_segments), 256):
            embedding = asyncio.run(
                generate_from_openai_embeddings_completion(
                    client = self.client,
                    messages = all_segments[i:i+256],
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

        return final_embeddings
    
    def embed_query(self, text: str) -> List[float]:
        return self.embed_documents([text])[0]

class ColBERT(Embeddings):
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
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        sentence_to_segments = []
        all_segments = []
        final_embeddings = []
        
        for text in texts:
            segments = self.segment_text(text)
            sentence_to_segments.append((text, len(segments)))
            all_segments.extend(segments)
        
        with torch.no_grad():
            inputs = self.tokenizer(all_segments, padding=True, truncation=True, return_tensors='pt').to(self.device)
            # Compute token embeddings
            outputs = self.model(**inputs)
            embeddings = self.mean_pooling(outputs[0], inputs['attention_mask'])
            sidx = 0
            
            for (t, l) in sentence_to_segments:
                
                final_embeddings.append(torch.mean(embeddings[sidx:sidx+l], axis = 0).cpu().tolist())
                sidx += l

        
        return final_embeddings

    def embed_query(self, text: str) -> List[float]:
        return self.embed_documents([text])[0]
    
class Instructor(Embeddings):
    def __init__(self, name: str, dataset:str, device: str, domain: bool, batch_size: int = 8):
        model_path = 'hkunlp/' + name
        self.model = SentenceTransformer(model_path)
        self.inst = 'Represent the '
        self.domain = domain
        self.batch_size = batch_size
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
    
    def embed_documents(self, texts: List[str], is_query: bool = False) -> List[List[float]]:
        sentence_to_segments = []
        all_segments = []
        final_embeddings = []
        
        if self.domain:
            inst = self.dinst
        else:
            inst = self.inst

        if is_query:
            inst += ' query for retrieve relevant paragraphs: '
        else:
            inst += ' paragraph for retrieval: '
        
        for text in texts:
            segments = self.segment_text(inst, text)
            sentence_to_segments.append((text, len(segments)))
            all_segments.extend(segments)

        
        final_embeddings = []
        sidx = 0
        with torch.no_grad():
            
            embeddings = self.model.encode(all_segments, prompt=inst, device=self.device, batch_size=self.batch_size)
            for (t,l) in sentence_to_segments:
                final_embeddings.append(np.mean(embeddings[sidx:sidx+l], axis=0).tolist())
                sidx += l
            
        return final_embeddings

    def embed_query(self, text: str) -> List[float]:
        return self.embed_documents([text], True)[0]

class GTR(Embeddings):
    def __init__(self, name: str, device: str, batch_size: int = 8):
        model_path = 'sentence-transformers/' + name
        self.model = SentenceTransformer(model_path)
        self.inst = 'Represent the paragraph for retrieval'
        self.batch_size = batch_size
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

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        sentence_to_segments = []
        all_segments = []
        final_embeddings = []

        for text in texts:
            segments = self.segment_text(text)
            sentence_to_segments.append((text, len(segments)))
            all_segments.extend(segments)

        
        final_embeddings = []
        sidx = 0
        with torch.no_grad():
            
            embeddings = self.model.encode(all_segments, device=self.device, batch_size=self.batch_size)
            for (t,l) in sentence_to_segments:
                final_embeddings.append(np.mean(embeddings[sidx:sidx+l], axis=0).tolist())
                sidx += l
            
        return final_embeddings

    def embed_query(self, text: str) -> List[float]:
        return self.embed_documents([text])[0]
