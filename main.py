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
    CHUNK_SIZE = 50
    OVERLAP = 10

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

    questions = ["赵某为某市小学教师。在执教期间，曾有学生家长于2019年1月以赵某非礼其女儿为由向学校举报，赵某因此写下书面检讨，保证不再发生此类事件。然而，2019年1月至2022年6月以来，赵某利用教学之便，在课间活动及补课期间，多次对5名女学生进行猥亵。你作为法官审理该案件，该如何审判？"]

    for q in questions:
        print("*" * 50)
        print(f"问题：{q}")
        print(f"LLM 回答：{llm.invoke(q).content}")
        print("-" * 50)
        print(f"LLM with RAG ：{rag_chain.invoke(q)}")
