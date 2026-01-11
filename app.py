import streamlit as st
import os
import tempfile
from rag_backend import process_documents, get_context_and_answer

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="OmniDoc RAG",
    page_icon="🧠",
    layout="wide"
)

# --- STATE MANAGEMENT ---
if "messages" not in st.session_state:
    st.session_state.messages = []

if "processed_files" not in st.session_state:
    st.session_state.processed_files = set()

# --- SIDEBAR: AUTO-UPLOAD ---
with st.sidebar:
    st.title("🧠 OmniDoc RAG")
    
    uploaded_files = st.file_uploader(
        "Upload Knowledge Base", 
        type=["pdf", "docx", "txt"], 
        accept_multiple_files=True
    )
    
    # --- AUTO-PROCESSING LOGIC ---
    if uploaded_files:
        # Get list of current filenames
        current_names = [f.name for f in uploaded_files]
        current_set = set(current_names)
        
        # Check if these are new files compared to what we have stored
        if current_set != st.session_state.processed_files:
            
            with st.status("🔄 Processing Documents...", expanded=True) as status:
                temp_paths = []
                try:
                    st.write("📂 Reading files...")
                    
                    # 1. Create Temp Files (Windows Safe Way)
                    for file in uploaded_files:
                        file_ext = f".{file.name.split('.')[-1]}"
                        
                        # Create, Write, Close (Crucial for Windows)
                        with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as tmp:
                            tmp.write(file.getbuffer())
                            tmp_path = tmp.name # Get the string path
                        
                        # Now the file is closed, we can append the path
                        temp_paths.append(tmp_path)

                    # 2. Process
                    st.write("🧠 Embedding content...")
                    processed_names = process_documents(temp_paths)
                    
                    # 3. Update State
                    st.session_state.processed_files = current_set
                    
                    status.update(label="✅ Ready!", state="complete", expanded=False)
                    st.success(f"Indexed: {', '.join(processed_names)}")

                except Exception as e:
                    st.error(f"Processing Error: {e}")
                finally:
                    # 4. Cleanup Temp Files
                    for p in temp_paths:
                        if os.path.exists(p):
                            try:
                                os.unlink(p)
                            except:
                                pass
    
    if st.session_state.processed_files:
        st.info(f"📚 Active Docs: {len(st.session_state.processed_files)}")
        
    st.markdown("---")
    if st.button("🗑️ Clear Chat History"):
        st.session_state.messages = []
        st.rerun()

# --- MAIN CHAT AREA ---
st.subheader("Chat with your Documents")

# Display History
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        
        if "sources" in message:
            count = len(message["sources"])
            with st.expander(f"📚 Evidence Used ({count} Chunks)"):
                for i, text in enumerate(message["sources"]):
                    st.markdown(f"**Chunk {i+1}**")
                    st.info(text)

# Input
if prompt := st.chat_input("Ask a question..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        try:
            # Retrieve & Generate
            active_files = list(st.session_state.processed_files)
            retrieved_docs, response_stream = get_context_and_answer(prompt, active_files)
            
            # Stream Text
            full_response = st.write_stream(response_stream)
            
            # Show Sources (Dynamic Count)
            count = len(retrieved_docs)
            source_texts = [doc.page_content for doc in retrieved_docs]
            
            with st.expander(f"📚 Evidence Used ({count} Chunks)"):
                for i, text in enumerate(source_texts):
                    st.markdown(f"**Chunk {i+1}**")
                    st.info(text)
            
            # Save
            st.session_state.messages.append({
                "role": "assistant", 
                "content": full_response,
                "sources": source_texts
            })
            
        except Exception as e:
            st.error(f"An error occurred: {e}")