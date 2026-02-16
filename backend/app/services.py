import os
import fitz  # PyMuPDF
import shutil
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

load_dotenv(override=True)

# 1. Setup FREE Local Embeddings
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# 2. Setup FREE Groq LLM
llm = ChatGroq(
    temperature=0,
    groq_api_key=os.getenv("GROQ_API_KEY"),
    model_name="llama-3.3-70b-versatile"
)

# Use a very specific absolute path for the database
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHROMA_PATH = os.path.join(BASE_DIR, "chroma_db")

def process_pdf(file_path: str):
    doc = None
    try:
        # 1. Complete Reset: Wipe the old database folder entirely
        if os.path.exists(CHROMA_PATH):
            shutil.rmtree(CHROMA_PATH)
        
        # 2. Extract Text with 'sort=True' to handle certificate layouts
        doc = fitz.open(file_path)
        text = ""
        for page in doc:
            text += page.get_text("text", sort=True) + "\n"
        doc.close()

        if not text.strip():
            print("CRITICAL: No text found in PDF!")
            return 0

        # 3. Use smaller chunks for high-density certificates
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
        chunks = text_splitter.split_text(text)
        
        # 4. Create new database
        vector_db = Chroma.from_texts(
            chunks, 
            embeddings, 
            persist_directory=CHROMA_PATH
        )
        print(f"SUCCESS: Created {len(chunks)} chunks in {CHROMA_PATH}")
        return len(chunks)
    except Exception as e:
        if doc: doc.close()
        print(f"ERROR during process_pdf: {e}")
        raise e

def query_vector_db(user_query: str):
    if not os.path.exists(CHROMA_PATH):
        return "Database folder missing. Please upload the PDF again."

    # Load the database
    vector_db = Chroma(
        persist_directory=CHROMA_PATH, 
        embedding_function=embeddings
    )

    # TEST: See if the database actually has data
    collection_count = vector_db._collection.count()
    if collection_count == 0:
        return "The database is empty. Please re-upload the PDF."

    # The Prompt: We tell the AI to be helpful but strict
    template = """You are a helpful assistant. Use the provided context to answer the question.
    
    Context:
    {context}

    Question: {question}
    
    Answer:"""
    
    prompt = ChatPromptTemplate.from_template(template)

    # Build the Chain
    chain = (
        {"context": vector_db.as_retriever(search_kwargs={"k": 5}), "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    return chain.invoke(user_query)