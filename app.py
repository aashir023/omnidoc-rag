import gradio as gr
import os
from rag_backend import process_documents, get_answer

def upload_files(files):
    if not files:
        return "No files uploaded."
    # Gradio passes file objects, we need their paths
    file_paths = [file.name for file in files]
    status = process_documents(file_paths)
    return status

def chat_function(message, history):
    answer, sources = get_answer(message)
    
    # Format sources for display
    formatted_sources = "\n".join([f"- {os.path.basename(s)}" for s in sources])
    
    final_response = f"{answer}\n\n**Sources:**\n{formatted_sources}"
    return final_response

# Design the UI
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🧠 Omni-Doc RAG System")
    gr.Markdown("Upload PDFs, DOCX, or TXT files and chat with them using Llama-3 & Pinecone.")
    
    with gr.Row():
        with gr.Column(scale=1):
            file_uploader = gr.File(
                label="Upload Documents", 
                file_count="multiple",
                file_types=[".pdf", ".docx", ".txt"]
            )
            upload_btn = gr.Button("Process Documents")
            upload_status = gr.Textbox(label="Status", interactive=False)
            
            upload_btn.click(fn=upload_files, inputs=file_uploader, outputs=upload_status)
            
        with gr.Column(scale=2):
            chatbot = gr.ChatInterface(
                fn=chat_function, 
                title="Chat with your Docs"
            )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)