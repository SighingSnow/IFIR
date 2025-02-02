# Load model directly
from typing import List, Dict
import asyncio
from utils.openai_utils import generate_from_openai_chat_completion_prob, aopenai_client

# from vllm import LLM, SamplingParams

from utils.llm_metrics.GEval.prompts import GEvalPrompts
from utils.llm_metrics.GEval.geval import content2json, openai_weighted_sum_score, llama_weighted_sum_score

class OpenAIEvaluator:
    
    def __init__(self, dataset: str, model_name: str = "gpt-4o"):
        """
            dataset: dataset is used to decide rpm
        """
        self.engine = model_name
        self.client = aopenai_client()
        
        self.rpm = 60
    
    def format_prompt(self, message: str):
        messages = [
            {'role' : 'user', 'content' : message}
        ]
        return messages
            
    def score(self, messages: List[str]):
        prompts = [self.format_prompt(m) for m in messages]
        responses = asyncio.run(
            generate_from_openai_chat_completion_prob(
                self.client,
                messages=prompts,
                engine_name=self.engine,
                temperature=0.0,
                max_tokens=512,
                top_p=0.7,
                logprobs=True,
                top_logprobs=10,
                requests_per_minute=self.rpm,
            )
        )
        
        d = {'score': -1.0, 'reason': 'Invalid Message'}
        
        formated_responses = [content2json(x.choices[0].message.content) if type(x) != str else d for x in responses]
        logprobs = [x.choices[0].logprobs.content for x in responses]
        weighted_scores = [openai_weighted_sum_score(x, y) for x, y in zip(formated_responses, logprobs)]
        for idx, x in enumerate(weighted_scores):
            formated_responses[idx]['score'] = x
            if formated_responses[idx].get('reason') is None:
                formated_responses[idx]['reason'] = "No reason provided."
        return formated_responses
    
        
class LLamaEvaluator:
    def __init__(self, model_name: str = "Llama-3.1-8B-Instruct"):
        model_name = "meta-llama/" + model_name
    
        self.sampling_params = SamplingParams(
            temperature=0.0,  
            top_p=0.9,      
            max_tokens=128,    
            logprobs=20,
        )
        self.llm = LLM(
            model=model_name,  
            dtype="bfloat16",
            max_model_len=2048,
            gpu_memory_utilization=0.95,
            tensor_parallel_size=2
        )

    def format_prompt(self, prompt, mode: str = "user"):
        return f"""<|begin_of_text|>\n <|start_header_id|>{mode}<|end_header_id|>\n{prompt}\n<|eot_id|>"""

    def score(self, messages: List[str]):
        prompts = [self.format_prompt(m) for m in messages]
        outputs = self.llm.generate(
            prompts, 
            self.sampling_params,
        )

        formated_responses = [content2json(o.outputs[0].text) for o in outputs]
        
        # logprobs is a List
        logprobs = [o.outputs[0].logprobs for o in outputs] # [len(outputs), 10]
        target_logprobs = []
        for logprob in logprobs:
            tmp_logprobs = []
            for l in logprob:
                tmp_logprobs.append(list(l.values()))
            target_logprobs.append(tmp_logprobs)
        # logprobs = [[l.values()] for logprob in logprobs for l in logprob]

        weighted_scores = [llama_weighted_sum_score(x, y) for x, y in zip(formated_responses, target_logprobs)]
        for idx, x in enumerate(weighted_scores):
            formated_responses[idx]['score'] = x
            if formated_responses[idx].get('reason') is None:
                formated_responses[idx]['reason'] = "No reason provided."
        return formated_responses
        
