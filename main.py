import os

from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_openai.chat_models.base import BaseChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from vector import LawVectordb

if __name__ == '__main__':
    print("欢迎体验 LLM + RAG 在法律领域应用")

    # 重要参数，可自行修改
    # DeepSeek LLM
    TEMPERATURE = 0.7
    MAX_TOKENS = 1024

    # Vectordb
    DATA_DIR = "./law_data"
    PERSIST_DIR = "./chroma"
    CHUNK_SIZE = 20
    OVERLAP = 5

    llm = BaseChatOpenAI(
        model='deepseek-chat',
        openai_api_key=os.environ["DEEPSEEK_API_KEY"],
        openai_api_base='https://api.deepseek.com',
        temperature=TEMPERATURE,
        max_tokens=MAX_TOKENS
    )

    lawVectordb = LawVectordb(data_directory=DATA_DIR, persist_directory=PERSIST_DIR,
                        chunk_size=CHUNK_SIZE, overlap=OVERLAP)
    db = lawVectordb.makeVectordb()
    retriever = db.as_retriever()

    output_parser = StrOutputParser()

    template = """你是一名法官，可以帮助我审判法律案件。你得到法律库:
     {context}
     问题：{question}
    """
    prompt = ChatPromptTemplate.from_template(template)

    setup_and_retrieval = RunnableParallel(
        {"context": retriever, "question": RunnablePassthrough()}
    )

    rag_chain = setup_and_retrieval | prompt | llm | output_parser

    questions = ["根据未成年人保护法，在学校幼儿园和其他未成年人集中活动的公共场所吸烟饮酒应该受到什么处罚？",
                 "2024年，某市某体育公司举办大型体育活动，有上千名群众参加，因为天气恶劣，主办方没有做好安全工作导致超过20名群众死亡，根据刑法，其主管人员犯了什么罪，将会受到何种刑事处罚，刑期如何？"]

    for q in questions:
        print("*" * 50)
        print(f"问题：{q}")
        print(f"LLM 回答：{llm.invoke(q).content}")
        print("-" * 50)
        print(f"LLM with RAG ：{rag_chain.invoke(q)}")
