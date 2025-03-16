import os

from langchain_community.document_loaders import PyMuPDFLoader
from langchain_community.document_loaders import UnstructuredMarkdownLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from zhipuai_embedding import ZhipuAIEmbeddings
from langchain_chroma import Chroma

class LawVectordb:
    def __init__(self, data_directory="./law_data", persist_directory="./chroma", chunk_size=20, overlap=5):
        self.data_directory = data_directory
        self.persist_directory = persist_directory
        self.chunk_size = chunk_size
        self.overlap = overlap

    def makeVectordb(self):
        embedding = ZhipuAIEmbeddings()

        if os.listdir(self.persist_directory):
            # 持久库是非空的，直接加载
            vectordb = Chroma(
                embedding_function=embedding,
                persist_directory=self.persist_directory
            )
        else:
            loaders = []
            paths = os.listdir(self.data_directory)
            for path in paths:
                relative_path = os.path.join(self.data_directory, path)
                file_type = path.split(".")[-1]
                if file_type == "pdf":
                    loaders.append(PyMuPDFLoader(relative_path))
                elif file_type == "md":
                    loaders.append(UnstructuredMarkdownLoader(relative_path))
            texts = []
            for loader in loaders:
                pdf_documents = loader.load()
                for page in pdf_documents:
                    page.page_content = page.page_content.replace("\n", "")
                print(pdf_documents)
                texts.extend(pdf_documents)

            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.chunk_size,
                chunk_overlap=self.overlap,
            )

            split_docs = text_splitter.split_documents(texts)

            # 加载文档到chrome向量库中，并持久化，vectordb = persist + docs
            vectordb = Chroma.from_documents(
                documents=split_docs,
                embedding=embedding,
                persist_directory=self.persist_directory
            )
        return vectordb

    def reloadVectordb(self, chunk_size=50, overlap=10):
        files = os.listdir(self.data_directory)
        if len(files) != 0:
            for file in files:
                os.remove(os.path.join(self.data_directory, file))

        self.chunk_size = chunk_size
        self.overlap = overlap
        return self.makeVectordb()
