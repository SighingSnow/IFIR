from typing import List
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
from langchain_core.embeddings import Embeddings
from peft import PeftModel, PeftConfig
import numpy as np
from sentence_transformers import SentenceTransformer
from gritlm import GritLM as GGritLM

class Promptriever(Embeddings):
    def __init__(self, model_name_or_path, device, batch_size: int = 8):
        model_name_or_path = 'samaya-ai/promptriever-llama2-7b-v1'
        self.model, self.tokenizer = self.get_model(model_name_or_path)
        if device.find("cuda") != -1:
            self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device("cpu")
        self.batch_size = batch_size
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

    def encode(self, sentences) -> List[List[float]]:
        all_embeddings = []
        
        for i in range(0, len(sentences), self.batch_size):
            batch_texts = sentences[i : i + self.batch_size]

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
                    all_embeddings.append(embeddings.cpu().numpy())

        return np.concatenate(all_embeddings, axis=0).tolist()

    def embed_documents(self, texts: List[str], is_query: bool = False) -> List[List[float]]:
        if is_query:
            texts = ["query:  " + text for text in texts]
        else:
            texts = ["passage:  " + text for text in texts]
        embeddings = self.encode(texts)
        return embeddings

    def embed_query(self, query) -> List[float]:
        return self.embed_documents([query])[0]

class NVEmbed(Embeddings):
    def __init__(self, device, batch_size: int = 8):
        self.model = SentenceTransformer(
            'nvidia/NV-Embed-v2', 
            trust_remote_code=True,
            model_kwargs={'torch_dtype': torch.float16},
            device=device
        )
        self.model.max_seq_length = 1024
        self.model.tokenizer.padding_side="right"
        self.batch_size = batch_size
        if device.find("cuda") != -1:
            self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device("cpu")
        self.inst = "Instruct: Retrieve passages satisfying the query. \nQuery: "
        # half precision
        # self.model.eval().to(self.device)

    def add_eos(self, corpus):
        corpus = [c for c in corpus if len(c) > 0]
        corpus = [c + self.model.tokenizer.eos_token for c in corpus]
        return corpus

    def embed_documents(self, texts: List[str], is_query: bool = False) -> List[List[float]]:

        inst = self.inst if is_query else ""
        # final_embeddings = []
        with torch.no_grad():
            embeddings = self.model.encode(self.add_eos(texts), prompt=inst, batch_size=self.batch_size, device=self.device, normalize_embeddings=True)
            embeddings = embeddings.tolist()
        
        return embeddings

    def embed_query(self, text: str) -> List[float]:
        return self.embed_documents([text], True)[0]

class E5(Embeddings):
    def __init__(self, dataset, device, domain: bool = False, batch_size: int = 8):
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
        self.batch_size = batch_size
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
        self.model.max_seq_length = 4096
        self.model.to(device)
        self.model.eval()

    def embed_documents(self, texts: List[str], is_query: bool = False) -> List[List[float]]:
        with torch.no_grad():
            embeddings = self.model.encode(sentences=texts, prompt=self.detailed_query(), batch_size=self.batch_size, normalize_embeddings=True)
           
            embeddings = embeddings.tolist()
            return embeddings
    
    def detailed_query(self):
        if self.domain:
            return f'Instruct: {self.dinst}\nQuery: '
        else:
            return f'Instruct: {self.inst}\nQuery: '

    def embed_query(self, text: str) -> List[float]:
        # add instruction for queries
        return self.embed_documents([text], True)[0]

class GritLM(Embeddings):

    def __init__(self, dataset, device, domain: bool = False, batch_size: int = 8):
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
        
        self.max_seq_length = 2048
        self.batch_size = batch_size
        # self.stride = 384
        if device.find("cuda") != -1:
            self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device("cpu")
        self.model = GGritLM('GritLM/GritLM-7B', pooling_method="mean", mode="embedding", torch_dtype=torch.float16, device=device)
        # self.model.to(device)
        self.model.eval()

    def segment_text(self, text):
        tokens = self.model.tokenizer.tokenize(text) # List[str]
        stride = self.stride
        max_seq_length = self.max_seq_length
        segments = []
        for i in range(0, len(tokens), stride):
            segment = tokens[i:i + max_seq_length]
            s = self.model.tokenizer.convert_tokens_to_string(segment)
            if len(s):
                segments.append(s)
        return segments

    def embed_documents(self, texts: List[str], is_query: bool = False) -> List[List[float]]:
    
        inst = self.dinst if self.domain else self.inst
        with torch.no_grad():
            # embeddings = None
            if is_query:
                embeddings = self.model.encode(texts, instruction=self.gritlm_instruction(inst), batch_size=self.batch_size, max_length=2048)
            else:
                embeddings = self.model.encode(texts, max_length=self.max_seq_length, instruction=self.gritlm_instruction(""), batch_size=self.batch_size)    

        return embeddings.tolist()
    
    def gritlm_instruction(self, text: str):
        return "<|user|>\n" + text + "\n<|embed|>\n" if text else "<|embed|>\n"

    def embed_query(self, text: str) -> List[float]:
        # add instruction for queries
        return self.embed_documents([text], is_query=True)[0]