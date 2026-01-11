import os
from dotenv import load_dotenv

load_dotenv()

# Loaders & Splitters
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Embeddings
from langchain_huggingface import HuggingFaceEmbeddings

# LLM & Vector Store
from langchain_groq import ChatGroq
from langchain_pinecone import PineconeVectorStore

# Chains    
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

# --- CONFIGURATION ---
INDEX_NAME = "omnidoc-rag"

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

llm = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0.0,
    max_tokens=1024
)

def process_documents(file_paths):
    """
    Reads files from string paths and indexes them.
    Returns the list of filenames processed.
    """
    if not file_paths:
        return []
        
    documents = []
    processed_names = []
    
    for path in file_paths:
        path = str(path)
        file_name = os.path.basename(path)
        
        try:
            if path.endswith('.pdf'):
                loader = PyPDFLoader(path)
            elif path.endswith('.docx'):
                loader = Docx2txtLoader(path)
            elif path.endswith('.txt'):
                loader = TextLoader(path)
            else:
                continue 
            
            loaded_docs = loader.load()
            for doc in loaded_docs:
                doc.metadata["source_name"] = file_name
                
            documents.extend(loaded_docs)
            processed_names.append(file_name)
        except Exception as e:
            print(f"Error loading {file_name}: {e}")

    if not documents:
        return []

    # Improved chunking for better context
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(documents)

    try:
        PineconeVectorStore.from_documents(
            documents=splits,
            embedding=embeddings,
            index_name=INDEX_NAME
        )
        return processed_names
    except Exception as e:
        raise Exception(f"Pinecone Error: {e}")

def get_context_and_answer(query, active_filenames):
    vectorstore = PineconeVectorStore(index_name=INDEX_NAME, embedding=embeddings)
    
    # Define metadata filter to only search active documents
    search_filter = {"source_name": {"$in": active_filenames}} if active_filenames else None
    
    # 1. Search for candidates with filtering
    docs_and_scores = vectorstore.similarity_search_with_score(query, k=6, filter=search_filter)
    
    # 2. Extract documents (Sorting by similarity) - Removed aggressive score filtering
    filtered_docs = []
    if docs_and_scores:
        # Sort by score (higher is typically better for cosine similarity)
        docs_and_scores.sort(key=lambda x: x[1], reverse=True)
        # No score filtering applied, all retrieved documents are passed
        filtered_docs = [doc for doc, score in docs_and_scores]

    # 3. Chain
    system_prompt = ( # Refined system prompt
        "You are a helpful assistant for question-answering tasks. "
        "Use the following pieces of retrieved context to answer the question. "
        "If you don't know the answer based on the context, just say that you "
        "cannot find the answer in the provided documents. "
        "Keep the answer concise, accurate, and directly relevant to the question. "
        "Do not make up information or elaborate beyond the given context.\n\n"
        "Context:\n{context}"
    )
    prompt = ChatPromptTemplate.from_messages(
        [("system", system_prompt), ("human", "{input}")]
    )
    chain = create_stuff_documents_chain(llm, prompt)
    
    # 4. Generator
    response_generator = chain.stream({"context": filtered_docs, "input": query})
    
    return filtered_docs, response_generator