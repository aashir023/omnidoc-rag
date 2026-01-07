import os
from dotenv import load_dotenv

# Load environment variables (for local testing)
load_dotenv()

from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader

# --- FIX IS HERE: Updated import path ---
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_huggingface import HuggingFaceInferenceAPIEmbeddings
from langchain_groq import ChatGroq
from langchain_pinecone import PineconeVectorStore
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

# --- CONFIGURATION ---
INDEX_NAME = "omnidoc-rag"

# 1. Setup Embeddings
# Uses Hugging Face Inference API (Free, Cloud-based)
embeddings = HuggingFaceInferenceAPIEmbeddings(
    api_key=os.environ.get("HUGGINGFACEHUB_API_TOKEN"),
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# 2. Setup LLM (Groq Llama-3)
llm = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
)

def process_documents(files):
    """
    Takes a list of file paths, reads them, chunks them, and uploads to Pinecone.
    """
    if not files:
        return "No files provided."
        
    documents = []
    
    for file in files:
        if file.endswith('.pdf'):
            loader = PyPDFLoader(file)
        elif file.endswith('.docx'):
            loader = Docx2txtLoader(file)
        elif file.endswith('.txt'):
            loader = TextLoader(file)
        else:
            continue 
            
        documents.extend(loader.load())

    if not documents:
        return "No valid documents found."

    # Chunking: Split text into manageble pieces
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(documents)

    # Upload to Pinecone
    # This automatically converts text -> vectors -> stores in cloud
    vectorstore = PineconeVectorStore.from_documents(
        documents=splits,
        embedding=embeddings,
        index_name=INDEX_NAME
    )
    
    return f"Successfully processed {len(splits)} chunks! Uploaded to Pinecone."

def get_answer(query):
    """
    Searches Pinecone for context and asks the LLM.
    """
    # Connect to the existing Pinecone index
    vectorstore = PineconeVectorStore(index_name=INDEX_NAME, embedding=embeddings)
    
    # Create the Retriever
    retriever = vectorstore.as_retriever()

    # Define the "Personality" of the AI
    system_prompt = (
        "You are an assistant for question-answering tasks. "
        "Use the following pieces of retrieved context to answer "
        "the question. If you don't know the answer, say that you "
        "don't know. Use three sentences maximum and keep the "
        "answer concise."
        "\n\n"
        "{context}"
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", "{input}"),
        ]
    )

    # Create the Chain
    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)

    # Run the chain
    response = rag_chain.invoke({"input": query})
    
    answer = response["answer"]
    # Extract sources safely
    sources = list(set([doc.metadata.get("source", "Unknown") for doc in response["context"]]))
    
    return answer, sources