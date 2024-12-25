from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableSequence
from langchain_core.output_parsers import StrOutputParser
from langchain_huggingface.llms import HuggingFacePipeline
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from src.helper import PROMPT_TEMPLATE
from src.logger1 import logger
from dotenv import load_dotenv

import os

load_dotenv()
HUGGINGFACE_API_KEY = os.environ['HUGGINGFACE_API_KEY']
#os.environ['HUGGINGFACE_API_KEY'] = HUGGINGFACE_API_KEY

class Text2Sql_llm:
    def __init__(self) :
        self.model_id = "RaviKanur/Phi-3.5-mini-4k-instruct-text2sql"
        logger.info("Initializing")

    def create_prompt_template(self) -> PromptTemplate :
        logger.info("Entered create_prompt_template method")
        prompt_template = PromptTemplate.from_template(template = PROMPT_TEMPLATE)
        
        return prompt_template
    
    def get_llm(self) -> HuggingFacePipeline:
        logger.info("Entered get_llm method")
        model = AutoModelForCausalLM.from_pretrained(self.model_id, token=HUGGINGFACE_API_KEY)
        tokenizer = AutoTokenizer.from_pretrained(self.model_id, token=HUGGINGFACE_API_KEY)
        pipe = pipeline(task="text-generation", model=model, tokenizer=tokenizer, model_kwargs={'max_length':512})
        hf_llm = HuggingFacePipeline(name="", pipeline=pipe)
        
        return hf_llm
    
    def create_chain(self, prompt: PromptTemplate, llm ) -> RunnableSequence:
        logger.info("Entered create_chain method")
        #retriever = {'context':"CREATE TABLE endowment (school_id VARCHAR, amount INTEGER); CREATE TABLE budget (school_id VARCHAR, budgeted INTEGER); CREATE TABLE school (school_name VARCHAR, school_id VARCHAR"}
        hf_chain = (prompt | 
                    llm | 
                    StrOutputParser())

        return hf_chain