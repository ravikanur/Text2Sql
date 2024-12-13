from src.logger1 import logger
import chainlit as cl
from src.logger1 import logger
from langchain_core.runnables import RunnableConfig, RunnableSequence
from src.llm import Text2Sql_llm
from src.vector_db import retrieve_doc


#logger = Logger()


@cl.on_chat_start
async def factory():
    obj = Text2Sql_llm()
    logger.info("Initializing the factory...")

    #logging.info("Initializing the factory")

    prompt_template = obj.create_prompt_template()

    llm = obj.get_llm()

    chain = obj.create_chain(prompt_template, llm)

    cl.user_session.set("llm_chain", chain)

@cl.on_message
async def main(message: cl.Message):
    #logger.update_id()
    llm_chain: RunnableSequence = cl.user_session.get("llm_chain")

    msg: cl.Message = cl.Message(content="")

    logger.info(message.content)

    question = message.content

    context = retrieve_doc(query=question, alpha=0.5)

    logger.info(context)

    resp = llm_chain.invoke({'question': question, 'context': context})

    logger.info(resp)

    msg.content = resp.split('<|assistant|>')[1].strip()

    await msg.send()