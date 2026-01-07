
# 🧠 Omni-Doc RAG System

**Omni-Doc RAG** is an enterprise-grade Retrieval Augmented Generation system designed to "chat" with multiple document formats (PDF, DOCX, TXT). It retrieves precise information from your uploaded files and provides answers with source citations using the Llama-3 LLM.

---

## 🏗️ Architecture & Tech Stack

This project is built using a modern, cloud-native AI stack:

| Component | Technology | Description |
| :--- | :--- | :--- |
| **LLM Inference** | **Groq** | Runs `Llama-3.1-8b-instant` at lightning speed. |
| **Vector Database** | **Pinecone** | Serverless, persistent cloud storage for vector embeddings. |
| **Embeddings** | **Hugging Face** | Uses `all-MiniLM-L6-v2` to convert text into vectors. |
| **Orchestration** | **LangChain** | Connects the data pipeline (Loaders -> Splitters -> LLM). |
| **User Interface** | **Gradio** | Provides the web-based chat and upload interface. |
| **Deployment** | **Docker** | Containerizes the application for consistent runtime. |
| **CI/CD** | **GitHub Actions** | Automates deployment to Hugging Face Spaces on every push. |

---

## 🚀 How It Works

1.  **Ingestion:** The user uploads documents via the Gradio UI.
2.  **Processing:**
    *   Files are read using `PyPDFLoader`, `Docx2txtLoader`, or `TextLoader`.
    *   Text is split into chunks of 1000 characters (with 200 overlap) to preserve context.
3.  **Embedding:** Each chunk is converted into a vector (a list of numbers) using the Hugging Face Inference API.
4.  **Storage:** Vectors are uploaded to the **Pinecone** cloud index.
5.  **Retrieval & Generation:**
    *   When a user asks a question, the system searches Pinecone for the most relevant text chunks.
    *   It sends the **Question + Relevant Chunks** to Llama-3 via Groq.
    *   Llama-3 answers the question based *only* on the provided context and cites the source.

---

## 🛠️ Local Installation

If you want to run this on your local machine:

### 1. Clone the Repository
```bash
git clone https://github.com/YOUR_USERNAME/omnidoc-rag.git
cd omnidoc-rag
```

### 2. Set up Environment
```bash
conda create -n omnidoc python=3.10 -y
conda activate omnidoc
pip install -r requirements.txt
```

### 3. Configure Secrets
Create a `.env` file in the root directory and add your keys:
```env
GROQ_API_KEY=your_groq_key_here
HUGGINGFACEHUB_API_TOKEN=your_hf_token_here
PINECONE_API_KEY=your_pinecone_key_here
```

### 4. Run the App
```bash
python app.py
```
Access the app at `http://localhost:7860`.

---

## 🐳 Docker Deployment

To build and run the container locally:

```bash
# Build
docker build -t omnidoc-rag .

# Run (Passing environment variables)
docker run -it -p 7860:7860 --env-file .env omnidoc-rag
```

---

## 🔄 CI/CD Pipeline (DevOps)

This repository includes a GitHub Actions workflow (`.github/workflows/sync_to_hub.yml`). 

**Automation Flow:**
1.  You push code to the `main` branch on GitHub.
2.  GitHub Actions triggers a runner.
3.  The runner logs into Hugging Face using the `HF_TOKEN` secret.
4.  It force-pushes the code to the Hugging Face Space.
5.  Hugging Face detects the `Dockerfile`, builds the image, and restarts the server.

---

## 🔑 Environment Variables Required

To deploy this on Hugging Face Spaces, add these secrets in the **Settings -> Variables and Secrets** tab:

*   `GROQ_API_KEY`: API key from Groq Cloud.
*   `HUGGINGFACEHUB_API_TOKEN`: Write-access token from Hugging Face.
*   `PINECONE_API_KEY`: API key from Pinecone Console.

---

Built by Aashir Ali

