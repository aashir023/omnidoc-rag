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

    # Reduced chunk size slightly to isolate names/titles better
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=100)
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

def get_context_and_answer(query):
    vectorstore = PineconeVectorStore(index_name=INDEX_NAME, embedding=embeddings)
    
    # 1. Search WIDER (Get top 10 candidates to ensure Title Page is caught)
    docs_and_scores = vectorstore.similarity_search_with_score(query, k=10)
    
    # 2. Relaxed Filtering Logic
    filtered_docs = []
    if docs_and_scores:
        docs_and_scores.sort(key=lambda x: x[1], reverse=True)
        highest_score = docs_and_scores[0][1]
        
        for doc, score in docs_and_scores:
            # RELAXED RULES:
            # 1. Keep anything above 0.20 (captures sparse title pages)
            # 2. OR keep it if it's very close to the best score
            if score > 0.20:
                filtered_docs.append(doc)
    
    # Fallback: If filtering failed, force keep the top 3
    if not filtered_docs and docs_and_scores:
        filtered_docs = [x[0] for x in docs_and_scores[:3]]
    
    # Cap at 6 docs max to prevent LLM confusion
    filtered_docs = filtered_docs[:6]

    # 3. Chain
    system_prompt = (
        "You are a strict research assistant. Use ONLY the provided context to answer. "
        "If the answer is not in the context, say 'I cannot find the answer in the documents.' "
        "Do not hallucinate.\n\n"
        "Context:\n{context}"
    )
    prompt = ChatPromptTemplate.from_messages(
        [("system", system_prompt), ("human", "{input}")]
    )
    chain = create_stuff_documents_chain(llm, prompt)
    
    # 4. Generator
    response_generator = chain.stream({"context": filtered_docs, "input": query})
    
    return filtered_docs, response_generator