PROMPT_TEMPLATE = ("""
<|user|>
Given the context, generate an SQL query for the following question. Please generate only 'SELECT' sql query and don't provide any other extra information.
context:{context}
question:{question}</s>
<|assistant|>
""")