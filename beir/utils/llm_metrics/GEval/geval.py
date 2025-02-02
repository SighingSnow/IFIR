import json
import numpy as np
# From response string to json
def content2json(input_string):
    start = input_string.find("{")
    end = input_string.rfind("}") + 1

    if end == 0 and start != -1:
        input_string = input_string + "}"
        end = len(input_string)

    jsonStr = input_string[start:end] if start != -1 and end != 0 else ""

    try:
        return json.loads(jsonStr)
    except json.JSONDecodeError:
        error_str = "Evaluation LLM outputted an invalid JSON. Please use a better evaluation model."
        raise ValueError(error_str)
    except Exception as e:
        raise Exception(f"An unexpected error occurred: {str(e)}")

def weighted_sum_score(score: int, res) -> float:
    try:
        logprobs = res.choices[0].logprobs.content
        
        for token_logprobs in logprobs:
            if token_logprobs.token == str(score):
                score_logprobs = token_logprobs
                break
        
        min_logprob = np.log(0.01)
        token_prob_dict = {}
        for token_logprob in score_logprobs.top_logprobs:
            logprob = token_logprob.logprob

            if logprob < min_logprob:
                continue

            if not token_logprob.token.isdecimal():
                continue

            linear_prob = np.exp(logprob)
            score = int(token_logprob.token)
            if score > 5 or score < 0:
                continue
            token_prob_dict[score] = linear_prob if score not in token_prob_dict else token_prob_dict[score] + linear_prob
        
        sum_of_weighted_score = 0.0
        for score, prob in token_prob_dict.items():
            sum_of_weighted_score += prob * score 
        
        return sum_of_weighted_score/np.sum(list(token_prob_dict.values()))
    except Exception as e:
        return 0.0
