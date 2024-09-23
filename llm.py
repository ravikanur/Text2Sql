from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableSequence
from langchain_core.output_parsers import StrOutputParser
from langchain_huggingface.llms import HuggingFacePipeline
from dotenv import load_dotenv

import os

load_dotenv()
HUGGINGFACE_API_KEY = os.getenv('HUGGINGFACE_API_KEY')
os.environ['HUGGINGFACE_API_KEY'] = HUGGINGFACE_API_KEY

class Text2Sql_llm:
    def __init__(self) :
        self.model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

    def create_prompt_template(self) -> PromptTemplate :
        prompt_template = PromptTemplate.from_template(template = """\
                                                        <|user|>
                                                        Given the context, generate an SQL query for the following question. Please generate only 'SELECT' sql query and don't provide any other extra information.
                                                        context:{context}
                                                        question:{question}</s>
                                                        <|assistant|>
                                                        """)
        
        return prompt_template
    
    def create_chain(self, prompt: PromptTemplate, llm ) -> RunnableSequence:
        hf_chain = (prompt | llm | StrOutputParser())

        return hf_chain