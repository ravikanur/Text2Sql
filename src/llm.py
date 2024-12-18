from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableSequence
from langchain_core.output_parsers import StrOutputParser
from langchain_huggingface.llms import HuggingFacePipeline
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
        hf_llm = HuggingFacePipeline.from_model_id(model_id=self.model_id, task="text-generation", 
                                           model_kwargs={'max_length':512})
        
        return hf_llm
    
    def create_chain(self, prompt: PromptTemplate, llm ) -> RunnableSequence:
        logger.info("Entered create_chain method")
        #retriever = {'context':"CREATE TABLE endowment (school_id VARCHAR, amount INTEGER); CREATE TABLE budget (school_id VARCHAR, budgeted INTEGER); CREATE TABLE school (school_name VARCHAR, school_id VARCHAR"}
        hf_chain = (prompt | 
                    llm | 
                    StrOutputParser())

        return hf_chain