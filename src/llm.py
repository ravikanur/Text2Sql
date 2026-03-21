import sys

from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableSequence
from langchain_core.output_parsers import StrOutputParser
from langchain_huggingface.llms import HuggingFacePipeline
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from peft import PeftModel
from src.helper import PROMPT_TEMPLATE
from src.logger1 import logger
from dotenv import load_dotenv
from src.exception import CustomException

import os

load_dotenv()
HUGGINGFACE_API_KEY = os.environ['HUGGINGFACE_API_KEY']
#os.environ['HUGGINGFACE_API_KEY'] = HUGGINGFACE_API_KEY

class Text2Sql_llm:
    def __init__(self) :
        #self.base_model_id = "microsoft/Phi-3-mini-4k-instruct"
        #self.model_id = "RaviKanur/Phi-3.5-mini-4k-instruct-text2sql"
        self.base_model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
        self.model_id = "RaviKanur/TinyLlama1"
        logger.info("Initializing Text2sql class")

    def create_prompt_template(self) -> PromptTemplate :
        try:
            logger.info("Entered create_prompt_template method")
            prompt_template = PromptTemplate.from_template(template = PROMPT_TEMPLATE)
        except Exception as e:
            logger.error(f"An error occurred while creating the prompt template: {e}")
            raise CustomException(e, sys)
        
        return prompt_template
    
    def get_llm(self) -> HuggingFacePipeline:
        try:
            logger.info("Entered get_llm method")
            model = AutoModelForCausalLM.from_pretrained(self.base_model_id, token=HUGGINGFACE_API_KEY)
            logger.info("Downloaded the model")
            model_1 = PeftModel.from_pretrained(model, self.model_id) 
            logger.info("Downloaded the adapter")
            tokenizer = AutoTokenizer.from_pretrained(self.base_model_id, token=HUGGINGFACE_API_KEY)
            logger.info("Downloaded the tokenizer")
            pipe = pipeline(task="text-generation", model=model, tokenizer=tokenizer, max_length=512, model_kwargs={'max_length':512})
            hf_llm = HuggingFacePipeline(name="", pipeline=pipe)
        except Exception as e:
            logger.error(f"An error occurred while initializing the LLM: {e}")
            raise CustomException(e, sys)
        return hf_llm
    
    def create_chain(self, prompt: PromptTemplate, llm ) -> RunnableSequence:
        try:
            logger.info("Entered create_chain method")
            #retriever = {'context':"CREATE TABLE endowment (school_id VARCHAR, amount INTEGER); CREATE TABLE budget (school_id VARCHAR, budgeted INTEGER); CREATE TABLE school (school_name VARCHAR, school_id VARCHAR"}
            hf_chain = (prompt | 
                        llm | 
                        StrOutputParser())
        except Exception as e:
            logger.error(f"An error occurred while creating the chain: {e}")
            raise CustomException(e, sys)

        return hf_chain