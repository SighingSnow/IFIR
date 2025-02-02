class GEvalPrompts:
    @staticmethod
    def nfcorpus(instruction, corpus):
        return f"""Given an instruction: {instruction},
and a corpus: {corpus},
Please evaluate the corpus according to the instruction and Evaluation Criteria and return a JSON object with the score and reason.

There is 3 relevant levels to evaluate the corpus regarding the instruction:
1. Research fields and research topics;
2. Research objectives;
3. Totally match the instruction. 

Evaluation Criteria:
1. If the corpus only meets the instruction in first level, the score is 1;
2. If the corpus meets the instruction in first and second levels, the score is 2;
3. If the corpus meets instruction with all three levels, the score is 3.
4. If the corpus does not meet any of the levels, the score is 0.

Please give a score between 0 and 3.

**
IMPORTANT: Please make sure to only return in JSON format, with the "score" and "reason" key. No words or explanation is needed.
Please think step by step of the reason and give the score according to the Evaluation Criteria.

Example JSON:
{{
    "score": 1,
    "reason": "The corpus only match the instruction with research field and research topics."
}}
**

JSON:
"""

    @staticmethod
    def cds(instruction, corpus):
        return f"""Given an instruction: {instruction},
and a corpus: {corpus},
Please evaluate the corpus according to the instruction and Evaluation Criteria and return a JSON object with the score and reason.

There is 3 relevant levels of the corpus:
1. Disease information.
2. Patient's demographics, especially age and gender;
3. Totally match the instruction.

Evaluation Criteria:
1. If the corpus only meets the instructions in first level, the score is 1;
2. If the corpus meets the instruction in first and second levels, the score is 2;
3. If the corpus meets all three levels, the score is 3.
4. If the corpus does not meet any of the levels, the score is 0.

Please give a score between 0 and 3.

**
IMPORTANT: Please make sure to only return in JSON format, with the "score" and "reason" key. No words or explanation is needed.
Please think step by step of the reason and give the score according to the Evaluation Criteria.

Example JSON:
{{
    "score": 1,
    "reason": "The corpus only match the instruction with disease information."
}}
**

JSON:
"""

    @staticmethod
    def fire(instruction, corpus):
        return f"""Given a prior case combined with a query: {instruction},  
and a corpus: {corpus}, 
please evaluate the corpus according to the query and Evaluation Criteria, and return a JSON object with the score and reason.

There are 3 relevant levels of the corpus:

1. Similar to the prior case in the query.
2. Partially helps with the query.
3. Totally matches the query.

Evaluation Criteria:

1. If the corpus only meets the query at the first level, the score is 1.
2. If the corpus meets the query at the first and second levels, the score is 2.
3. If the corpus meets all three levels, the score is 3.
4. If the corpus does not meet any of the levels, the score is 0.
Please give a score between 0 and 3.

**
IMPORTANT: Please make sure to only return the result in JSON format, with the "score" and "reason" keys. No additional words or explanations are needed.
Please think step by step of the reason and give the score according to the Evaluation Criteria.

Example JSON:
{{
"score": 1,
"reason": "The corpus only matches the instruction with disease information."
}}
**

JSON:
"""

    @staticmethod
    def pm3(instruction, corpus):
        return f"""Given an instruction: {instruction},
and a corpus: {corpus},
Please evaluate the corpus according to the instruction and Evaluation Criteria and return a JSON object with the score and reason.

There is 3 relevant levels of the corpus:
1. Disease information.
2. Patient's demographics, especially age and gender;
3. Totally match the instruction, including the family medical history and treatment history of the patient and other information.

Evaluation Criteria:
1. If the corpus only meets the instructions in first level, the score is 1;
2. If the corpus meets the instruction in first and second levels, the score is 2;
3. If the corpus meets all three levels, the score is 3.
4. If the corpus does not meet any of the levels, the score is 0.

Please give a score between 0 and 3.

**
IMPORTANT: Please make sure to only return in JSON format, with the "score" and "reason" key. No words or explanation is needed.
Please think step by step of the reason and give the score according to the Evaluation Criteria.

Example JSON:
{{
    "score": 1,
    "reason": "The corpus only match the instruction with disease information."
}}
**

JSON:
"""

    @staticmethod
    def pm2(instruction, corpus):
        return f"""Given an instruction: {instruction},
and a corpus: {corpus},
Please evaluate the corpus according to the instruction and Evaluation Criteria and return a JSON object with the score and reason.

There is 2 relevant levels of the corpus:
1. Disease information.
2. Totally match the instruction, including patient's demographics, especially age and gender;

Evaluation Criteria:
1. If the corpus only meets the instructions in first level, the score is 1;
2. If the corpus meets the instruction in first and second levels, the score is 2;
3. If the corpus does not meet any of the levels, the score is 0.

Please give a score between 0 and 2.

**
IMPORTANT: Please make sure to only return in JSON format, with the "score" and "reason" key. No words or explanation is needed.
Please think step by step of the reason and give the score according to the Evaluation Criteria.

Example JSON:
{{
    "score": 1,
    "reason": "The corpus only match the instruction with disease information."
}}
**

JSON:
"""
    
    @staticmethod
    def pm1(instruction, corpus):
        return f"""Given an instruction: {instruction},
and a corpus: {corpus},
Please evaluate the corpus according to the instruction and Evaluation Criteria and return a JSON object with the score and reason.

There is 1 relevant levels of the corpus:
1. Matching the instruction, including disease information.

Evaluation Criteria:
1. If the corpus only meets the instructions in first level, the score is 1;
3. If the corpus does not meet any of the levels, the score is 0.

Please give a score between 0 and 1.

**
IMPORTANT: Please make sure to only return in JSON format, with the "score" and "reason" key. No words or explanation is needed.
Please think step by step of the reason and give the score according to the Evaluation Criteria.

Example JSON:
{{
    "score": 1,
    "reason": "The corpus only match the instruction with disease information."
}}
**

JSON:
"""

    @staticmethod
    def scifact_open3(instruction, corpus):
        return f"""Given an instruction: {instruction},
and a corpus: {corpus},
Please evaluate the corpus according to the instruction and Evaluation Criteria and return a JSON object with the score and reason.

There is 3 relevant levels to evaluate the corpus regarding the instruction:
1. Theme relevance, for example, the research field and research topics;
2. Relationship, for example, support or refute the claim;
3. Totally match the instruction. 

Evaluation Criteria:
1. If the corpus only meets the instruction in first level, the score is 1;
2. If the corpus meets the instruction in first and second levels, the score is 2;
3. If the corpus meets instruction with all three levels, the score is 3.
4. If the corpus does not meet any of the levels, the score is 0.

Please give a score between 0 and 3.

**
IMPORTANT: Please make sure to only return in JSON format, with the "score" and "reason" key. No words or explanation is needed.
Please think step by step of the reason and give the score according to the Evaluation Criteria.

Example JSON:
{{
    "score": 1,
    "reason": "The corpus only match the instruction with research field and research topics."
}}
**

JSON:
"""

    @staticmethod
    def scifact_open2(instruction, corpus):
        return f"""Given an instruction: {instruction},
and a corpus: {corpus},
Please evaluate the corpus according to the instruction and Evaluation Criteria and return a JSON object with the score and reason.

There is 2 relevant levels to evaluate the corpus regarding the instruction:
1. Theme relevance, for example, the research field and research topics;
2. Totally match the instruction. Including relationship, for example, support or refute the claim;

Evaluation Criteria:
1. If the corpus only meets the instruction in first level, the score is 1;
2. If the corpus meets the instruction in first and second levels, the score is 2;
3. If the corpus does not meet any of the levels, the score is 0.

Please give a score between 0 and 2.

**
IMPORTANT: Please make sure to only return in JSON format, with the "score" and "reason" key. No words or explanation is needed.
Please think step by step of the reason and give the score according to the Evaluation Criteria.

Example JSON:
{{
    "score": 1,
    "reason": "The corpus only match the instruction with research field and research topics."
}}
**

JSON:
"""

    @staticmethod
    def scifact_open1(instruction, corpus):
        return f"""Given an instruction: {instruction},
and a corpus: {corpus},
Please evaluate the corpus according to the instruction and Evaluation Criteria and return a JSON object with the score and reason.

There is 2 relevant levels to evaluate the corpus regarding the instruction:
1. Totally match the instruction. Including theme relevance, for example, the research field and research topics;

Evaluation Criteria:
1. If the corpus meets the instruction in first level, the score is 1;
2. If the corpus does not meet any of the levels, the score is 0.

Please give a score between 0 and 1.

**
IMPORTANT: Please make sure to only return in JSON format, with the "score" and "reason" key. No words or explanation is needed.
Please think step by step of the reason and give the score according to the Evaluation Criteria.

Example JSON:
{{
    "score": 1,
    "reason": "The corpus only match the instruction with research field and research topics."
}}
**

JSON:
"""

    @staticmethod
    def fiqa3(instruction, corpus):
        return f"""Given an instruction: {instruction},
and an answer to the query: {corpus},
Please evaluate the corpus according to the instruction and Evaluation Criteria and return a JSON object with the score and reason.

There is 3 relevant levels to evaluate the answer regarding the instruction:
1. Relevant to the instruction.
2. Aligned with the personal financial status in the instruction.
3. Totally match the instruction, including the personal financial goals and financial status.

Evaluation Criteria:
1. If the answer only meets the instruction in first level, the score is 1;
2. If the answer meets the instruction in first and second levels, the score is 2;
3. If the answer meets the instruction with all three levels, the score is 3.
4. If the corpus does not meet any of the levels, the score is 0.

Please give a score between 0 and 3.

**
IMPORTANT: Please make sure to only return in JSON format, with the "score" and "reason" key. No words or explanation is needed.
Please think step by step of the reason and give the score according to the Evaluation Criteria.

Example JSON:
{{
    "score": 1,
    "reason": "The corpus only match the instruction with research field and research topics."
}}
**

JSON:
"""

    @staticmethod
    def fiqa2(instruction, corpus):
        return f"""Given an instruction: {instruction},
and an answer to the query: {corpus},
Please evaluate the corpus according to the instruction and Evaluation Criteria and return a JSON object with the score and reason.

There is 2 relevant levels to evaluate the answer regarding the instruction:
1. Relevant to the instruction.
2. Totally match the instruction, aligned with the personal financial status in the instruction.

Evaluation Criteria:
1. If the answer only meets the instruction in first level, the score is 1;
2. If the answer meets the instruction in first and second levels, the score is 2;
4. If the corpus does not meet any of the levels, the score is 0.

Please give a score between 0 and 2.

**
IMPORTANT: Please make sure to only return in JSON format, with the "score" and "reason" key. No words or explanation is needed.
Please think step by step of the reason and give the score according to the Evaluation Criteria.

Example JSON:
{{
    "score": 1,
    "reason": "The corpus only match the instruction with research field and research topics."
}}
**

JSON:
"""

    @staticmethod
    def fiqa1(instruction, corpus):
        return f"""Given an instruction: {instruction},
and an answer to the query: {corpus},
Please evaluate the corpus according to the instruction and Evaluation Criteria and return a JSON object with the score and reason.

There is 1 relevant levels to evaluate the answer regarding the instruction:
1. Relevant to the instruction.

Evaluation Criteria:
1. If the answer meets the instruction in first level, the score is 1;
4. If the corpus does not meet any of the levels, the score is 0.

Please give a score between 0 and 1.

**
IMPORTANT: Please make sure to only return in JSON format, with the "score" and "reason" key. No words or explanation is needed.
Please think step by step of the reason and give the score according to the Evaluation Criteria.

Example JSON:
{{
    "score": 1,
    "reason": "The corpus only match the instruction with research field and research topics."
}}
**

JSON:
"""

    @staticmethod
    def aila3(instruction, corpus):
        return f"""Given an instruction: {{instruction}},
and a prior case: {{corpus}},
please evaluate the prior case according to the instruction and Evaluation Criteria and return a JSON object with the score and reason.

There are 3 relevant levels to evaluate the case regarding the instruction:
1. The prior case is similar to the one in the instruction.
2. The prior case satisfies the instruction at the 'plaintiff' or 'defendant' beneficial level.
3. The prior case totally matches the instruction, including the detailed requirements in the instruction.

Evaluation Criteria:
1. If the prior case only meets the instruction at the first level, the score is 1.
2. If the prior case meets the instruction at the first and second levels, the score is 2.
3. If the prior case meets the instruction at all three levels, the score is 3.
4. If the prior case does not meet any of the levels, the score is 0.

Please give a score between 0 and 3.

**
IMPORTANT: Please make sure to only return in JSON format, with the "score" and "reason" key. No additional words or explanations are needed.
Please think step by step about the reason and give the score according to the Evaluation Criteria.

Example JSON:
{{
    "score": 1,
    "reason": "The corpus only matches the instruction in terms of research field and research topics."
}}
**

JSON:
"""

    @staticmethod
    def aila2(instruction, corpus):
        return f"""Given an instruction: {instruction},
and a prior case: {{corpus}},
please evaluate the prior case according to the instruction and Evaluation Criteria and return a JSON object with the score and reason.

There are 2 relevant levels to evaluate the case regarding the instruction:
1. The prior case is similar to the one in the instruction.
2. The prior case totally matches the instruction, satisfing the instruction at the 'plaintiff' or 'defendant' beneficial level.

Evaluation Criteria:
1. If the prior case only meets the instruction at the first level, the score is 1.
2. If the prior case meets the instruction at the first and second levels, the score is 2.
3. If the prior case does not meet any of the levels, the score is 0.

Please give a score between 0 and 2.

**
IMPORTANT: Please make sure to only return in JSON format, with the "score" and "reason" key. No additional words or explanations are needed.
Please think step by step about the reason and give the score according to the Evaluation Criteria.

Example JSON:
{{
    "score": 1,
    "reason": "The corpus only matches the instruction in terms of research field and research topics."
}}
**

JSON:
"""

    @staticmethod
    def aila1(instruction, corpus):
        return f"""Given an instruction: {instruction},
and a prior case: {corpus},
please evaluate the prior case according to the instruction and Evaluation Criteria and return a JSON object with the score and reason.

There are 1 relevant levels to evaluate the case regarding the instruction:
1. The prior case is similar to the one in the instruction.

Evaluation Criteria:
1. If the prior case meets the instruction at the first level, the score is 1.
2. If the prior case does not meet any of the levels, the score is 0.

Please give a score between 0 and 1.

**
IMPORTANT: Please make sure to only return in JSON format, with the "score" and "reason" key. No additional words or explanations are needed.
Please think step by step about the reason and give the score according to the Evaluation Criteria.

Example JSON:
{{
    "score": 1,
    "reason": "The corpus only matches the instruction in terms of research field and research topics."
}}
**

JSON:
"""
