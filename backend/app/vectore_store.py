from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma

embeddings = OpenAIEmbeddings()

def create_vector_store(chunks, persist_directory="vector_db"):
    vectordb = Chroma.from_texts(
        texts=chunks,
        embedding=embeddings,
        persist_directory=persist_directory
    )
    vectordb.persist()
    return vectordb

def load_vector_store(persist_directory="vector_db"):
    return Chroma(
        persist_directory=persist_directory,
        embedding_function=embeddings
    )