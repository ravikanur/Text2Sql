import chainlit as cl
from langchain_core.runnables import RunnableConfig, RunnableSequence
from src.llm import Text2Sql_llm

@cl.on_chat_start
async def factory():
    obj = Text2Sql_llm()

    prompt_template = obj.create_prompt_template()

    llm = obj.get_llm()

    chain = obj.create_chain(prompt_template, llm)

    cl.user_session.set("llm_chain", chain)

@cl.on_message
async def main(message: cl.Message):
    llm_chain: RunnableSequence = cl.user_session.get("llm_chain")

    msg: cl.Message = cl.Message(content="")

    print(message.content)

    question = message.content

    context = "CREATE TABLE endowment (school_id VARCHAR, amount INTEGER); CREATE TABLE budget (school_id VARCHAR, budgeted INTEGER); CREATE TABLE school (school_name VARCHAR, school_id VARCHAR"

    resp = llm_chain.invoke({'question': question, 'context': context})

    msg.content = resp.split('<|assistant|>')[1].strip()

    await msg.send()