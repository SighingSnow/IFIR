import torch
import torch.nn.functional as F

from gritlm import GritLM as GGritLM
from tqdm import tqdm
from typing import List, Union, Dict
import numpy as np
from peft import PeftModel, PeftConfig
from transformers import AutoModel, AutoTokenizer
from sentence_transformers import SentenceTransformer

def d2list(corpus: Dict[str, str], sep: str = " ") -> List[str]:
    return [(corpus["title"][i] + sep + corpus["text"][i]).strip() if "title" in corpus else corpus["text"][i].strip() for i in range(len(corpus['text']))]

def dl2list(corpus: List[Dict[str, str]], sep: str = " ") -> List[str]:
    return [(doc["title"] + sep + doc["text"]).strip() if "title" in doc else doc["text"].strip() for doc in corpus]

# E5-mistral-7b-instruct
class E5:
    def __init__(self, dataset, device, domain: bool = False):
        self.domain = domain
    
        if dataset == 'fiqa':
            self.domain_name = "finance"
        elif dataset == 'aila' or dataset == 'fire':
            self.domain_name = 'legal'
        elif dataset == 'scifact_open' or dataset == 'nfcorpus':
            self.domain_name = 'science'
        elif dataset == 'pm' or dataset == 'cds':
            self.domain_name = 'biomedial' 
        
        self.dinst = f"Given a {self.domain_name} query, retrieval relevant passages."
        self.inst = "Give a query, retrieval relevant passages"
        # device
        if device.find("cuda") != -1:
            self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device("cpu")
        
        self.model = SentenceTransformer(
            "intfloat/e5-mistral-7b-instruct",
            model_kwargs={'torch_dtype': torch.float16},
            device=device
        )
        self.max_seq_length = 4096
        self.model.to(device)
        self.model.eval()
        
    def encode_corpus(self, 
        corpus: Union[List[str], Dict[str, str], List[Dict[str, str]]], 
        batch_size: int = 32, 
        show_progress_bar: bool = True, 
        convert_to_tensor: bool = True,
        normalize_embeddings: bool = True
    ) -> Union[np.ndarray, torch.Tensor]:
        if type(corpus) is dict:
            corpus = d2list(corpus)
        elif type(corpus) is list and type(corpus[0]) is dict:
            corpus = dl2list(corpus)
        
        with torch.no_grad():
            embeddings = self.model.encode(
                sentences=corpus, 
                batch_size=batch_size,
                show_progress_bar=show_progress_bar,
                normalize_embeddings=normalize_embeddings,
                convert_to_tensor=convert_to_tensor
            ).cpu()
        
        return embeddings
    
    def detailed_query(self, text: str):
        if self.domain:
            return f'Instruct{self.dinst}\nQuery{text}'
        else:
            return f'Instruct{self.inst}\nQuery{text}'

    def encode_queries(
        self, 
        queries: List[str], 
        batch_size: int = 32, 
        show_progress_bar: bool = True, 
        convert_to_tensor: bool = True,
        normalize_embeddings: bool = True
    ) -> Union[np.ndarray, torch.tensor]:
        # add instruction for queries
        return self.encode_corpus(
            corpus = [self.detailed_query(text) for text in queries],
            batch_size = batch_size,
            show_progress_bar = show_progress_bar,
            convert_to_tensor = convert_to_tensor,
            normalize_embeddings=normalize_embeddings
        )

class GritLM:
    def __init__(self, dataset, device, domain: bool = False):
        self.domain = domain
    
        if dataset == 'fiqa':
            self.domain_name = "finance"
        elif dataset == 'aila' or dataset == 'fire':
            self.domain_name = 'legal'
        elif dataset == 'scifact_open' or dataset == 'nfcorpus':
            self.domain_name = 'science'
        elif dataset == 'pm' or dataset == 'cds':
            self.domain_name = 'biomedial' 
        
        self.dinst = f"Given a {self.domain_name} query, retrieval relevant passages."
        self.inst = "Give a query, retrieval relevant passages"
        
        self.max_seq_length = 4096
        if device.find("cuda") != -1:
            self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device("cpu")
        self.model = GGritLM('GritLM/GritLM-7B', pooling_method="mean", mode="embedding", torch_dtype=torch.float16, device=device)
        # self.model.to(device)
        self.model.eval()

    def encode_corpus(self, 
        corpus: Union[List[str], Dict[str, str], List[Dict[str, str]]], 
        batch_size: int = 32, 
        show_progress_bar: bool = True, 
        convert_to_tensor: bool = False,
        normalize_embeddings: bool = False,
        is_query: bool = False, 
    ) -> Union[np.ndarray, torch.Tensor]:

        if type(corpus) is dict:
            corpus = d2list(corpus)
        elif type(corpus) is list and type(corpus[0]) is dict:
            corpus = dl2list(corpus)
    
        inst = self.dinst if self.domain else self.inst
        
        with torch.no_grad():
                
            if self.domain:
                embeddings = self.model.encode(
                    corpus, 
                    max_length=self.max_seq_length, 
                    instruction=self.gritlm_instruction(inst), 
                    batch_size=batch_size,
                    show_progress_bar=show_progress_bar,
                    convert_to_tensor=convert_to_tensor
                )
            else:
                embeddings = self.model.encode(
                    corpus, 
                    max_length=self.max_seq_length, 
                    instruction=self.gritlm_instruction(""), 
                    batch_size=batch_size,
                    show_progress_bar=show_progress_bar,
                    convert_to_tensor=convert_to_tensor
                )    
        
        return embeddings
    
    def gritlm_instruction(self, text: str):
        return "<|user|>\n" + text + "\n<|embed|>\n" if text else "<|embed|>\n"

    def encode_queries(
        self, 
        queries: List[str], 
        batch_size: int = 32, 
        show_progress_bar: bool = True, 
        convert_to_tensor: bool = True,
        normalize_embeddings: bool = True
    ) -> Union[np.ndarray, torch.tensor]:
        return self.encode_corpus(
            corpus=queries, 
            batch_size=batch_size,
            is_query=True,
            show_progress_bar=show_progress_bar,
            convert_to_tensor=True
        )
        
class Promptriever:
    def __init__(self, device):
        model_name_or_path = 'samaya-ai/promptriever-llama2-7b-v1'
        self.model, self.tokenizer = self.get_model(model_name_or_path)
        if device.find("cuda") != -1:
            self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device("cpu")
        # half precision
        self.model.half()
        self.model.eval().to(self.device)
    
    def get_model(self, peft_model_name):
        # Load the PEFT configuration to get the base model name
        peft_config = PeftConfig.from_pretrained(peft_model_name)
        base_model_name = peft_config.base_model_name_or_path

        # Load the base model and tokenizer
        base_model = AutoModel.from_pretrained(base_model_name)
        tokenizer = AutoTokenizer.from_pretrained(base_model_name)
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
        tokenizer.padding_side = "right"

        # Load and merge the PEFT model
        model = PeftModel.from_pretrained(base_model, peft_model_name)
        model = model.merge_and_unload()
        
        # can be much longer, but for the example 512 is enough
        model.config.max_length = 1024
        tokenizer.model_max_length = 1024

        return model, tokenizer
    
    def create_batch_dict(self, tokenizer, input_texts):
        max_length = self.model.config.max_length
        batch_dict = tokenizer(
            input_texts,
            max_length=max_length - 1,
            return_token_type_ids=False,
            return_attention_mask=False,
            padding=False,
            truncation=True,
        )
        batch_dict["input_ids"] = [
            input_ids + [tokenizer.eos_token_id]
            for input_ids in batch_dict["input_ids"]
        ]
        return tokenizer.pad(
            batch_dict,
            padding=True,
            pad_to_multiple_of=8,
            return_attention_mask=True,
            return_tensors="pt",
        )
    
    def encode_corpus(self, 
        corpus: Union[List[str], Dict[str, str], List[Dict[str, str]]], 
        batch_size: int = 32, 
        show_progress_bar: bool = True, 
        convert_to_tensor: bool = True,
        normalize_embeddings: bool = True
    ) -> Union[np.ndarray, torch.Tensor]:
        if type(corpus) is dict:
            corpus = d2list(corpus)
        elif type(corpus) is list and type(corpus[0]) is dict:
            corpus = dl2list(corpus)

        all_embeddings = []
        
        for i in range(0, len(corpus), batch_size):
            batch_texts = corpus[i : i + batch_size]

            batch_dict = self.create_batch_dict(self.tokenizer, batch_texts)
            batch_dict = {
                key: value.to(self.model.device) for key, value in batch_dict.items()
            }

            with torch.cuda.amp.autocast():
                with torch.no_grad():
                    outputs = self.model(**batch_dict)
                    last_hidden_state = outputs.last_hidden_state.to(torch.float32)
                    
                    if torch.isnan(last_hidden_state).any() or torch.isinf(last_hidden_state).any():
                        raise ValueError("Model output contains NaN or Inf values")

                    sequence_lengths = batch_dict["attention_mask"].sum(dim=1) - 1
                    batch_size = last_hidden_state.shape[0]
                    reps = last_hidden_state[
                        torch.arange(batch_size, device=last_hidden_state.device),
                        sequence_lengths,
                    ]

                    reps = reps.to(torch.float32)
                    if torch.isnan(reps).any() or torch.isinf(reps).any():
                        raise ValueError("Reps contain NaN or Inf values")
                    if (reps.abs().sum(dim=-1) == 0).any():
                        raise ValueError("Reps contain all-zero vectors")

                    embeddings = F.normalize(reps, p=2, dim=-1)
                    all_embeddings.append(embeddings.cpu())
        
        return torch.cat(all_embeddings, dim=0)

    def encode_queries(
        self, 
        queries: List[str], 
        batch_size: int = 32, 
        show_progress_bar: bool = True, 
        convert_to_tensor: bool = False,
        normalize_embeddings: bool = False
    ) -> Union[np.ndarray, torch.tensor]:
        # add instruction for queries
        return self.encode_corpus(
            corpus = [text for text in queries],
            batch_size = batch_size,
            show_progress_bar = show_progress_bar,
            convert_to_tensor = convert_to_tensor
        )
        
class NVEmbed:
    def __init__(self, device):
        self.model = SentenceTransformer(
            'nvidia/NV-Embed-v2', 
            trust_remote_code=True,
            model_kwargs={'torch_dtype': torch.float16},
            device=device
        )
        self.model.max_seq_length = 1024
        self.model.tokenizer.padding_side="right"
        if device.find("cuda") != -1:
            self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device("cpu")
        self.inst = "Instruct: Retrieve passages satisfying the query. \nQuery: "
    
    def add_eos(self, corpus):
        corpus = [c for c in corpus if len(c) > 0]
        corpus = [c + self.model.tokenizer.eos_token for c in corpus]
        return corpus
    
    def encode_corpus(self, 
        corpus: Union[List[str], Dict[str, str], List[Dict[str, str]]], 
        batch_size: int = 32, 
        show_progress_bar: bool = True, 
        convert_to_tensor: bool = True,
        normalize_embeddings: bool = True,
        is_query: bool = False,
    ) -> Union[np.ndarray, torch.Tensor]:
        if type(corpus) is dict:
            corpus = d2list(corpus)
        elif type(corpus) is list and type(corpus[0]) is dict:
            corpus = dl2list(corpus)
        corpus = self.add_eos(corpus)
        inst = self.inst if is_query else ""
        with torch.no_grad():
            embeddings = self.model.encode(
                corpus, 
                prompt=inst,
                batch_size=batch_size, 
                device=self.device, 
                show_progress_bar=show_progress_bar,
                normalize_embeddings=normalize_embeddings,
                convert_to_tensor=convert_to_tensor
            ).cpu()
        
        
        return embeddings
    
    def encode_queries(
        self, 
        queries: List[str], 
        batch_size: int = 32, 
        show_progress_bar: bool = True, 
        convert_to_tensor: bool = True,
        normalize_embeddings: bool = True
    ) -> Union[np.ndarray, torch.tensor]:
        # add instruction for quersies
        return self.encode_corpus(
            corpus = queries,
            batch_size = batch_size,
            show_progress_bar = show_progress_bar,
            convert_to_tensor = convert_to_tensor,
            normalize_embeddings=normalize_embeddings,
            is_query=True
        )