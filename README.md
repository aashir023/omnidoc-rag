---
title: OmniDoc RAG
emoji: 🧠
colorFrom: indigo
colorTo: blue
sdk: docker
app_file: app.py
pinned: false
---

# 🧠 Omni-Doc RAG System

**Omni-Doc RAG** is an enterprise-grade Retrieval Augmented Generation system designed to "chat" with multiple document formats (PDF, DOCX, TXT). It retrieves precise information from your uploaded files and provides answers with source citations using the Llama-3 LLM.

This version utilizes **Streamlit** for a modern UI and **Local Embeddings** for robustness and cost-efficiency.

---

## 🏗️ Architecture & Tech Stack

This project is built using a modern, cloud-native AI stack:

| Component | Technology | Description |
| :--- | :--- | :--- |
| **LLM Inference** | **Groq** | Runs `Llama-3.1-8b-instant` at lightning speed. |
| **Vector Database** | **Pinecone** | Serverless, persistent cloud storage for vector embeddings. |
| **Embeddings** | **Hugging Face (Local)** | Uses `all-MiniLM-L6-v2` running locally on CPU (Dimensions: 384). |
| **Orchestration** | **LangChain** | Connects the data pipeline (Loaders -> Splitters -> LLM). |
| **User Interface** | **Streamlit** | Provides the interactive chat, sidebar, and source visualization. |
| **Deployment** | **Docker** | Containerizes the application for Hugging Face Spaces. |
| **CI/CD** | **GitHub Actions** | Automates deployment to Hugging Face Spaces on every push. |

---

## 🚀 How It Works

1.  **Ingestion:** The user drops documents into the sidebar. The system automatically detects new files.
2.  **Processing:**
    *   Files are read using `PyPDFLoader`, `Docx2txtLoader`, or `TextLoader`.
    *   Text is split into chunks of 800 characters (with 150 overlap).
3.  **Embedding:** Text is converted into vectors using `sentence-transformers` locally (no API calls required for embeddings).
4.  **Storage:** Vectors are upserted to the **Pinecone** index (`omnidoc-rag`).
5.  **Retrieval & Generation:**
    *   When a user asks a question, the system performs a **Similarity Search** with dynamic score filtering.
    *   It retrieves only the most relevant chunks (ignoring low-relevance noise).
    *   Llama-3 answers based *strictly* on the provided context.
    *   **Exact Source Evidence** is displayed in expandable dropdowns.

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
PINECONE_API_KEY=your_pinecone_key_here
# Note: HUGGINGFACEHUB_API_TOKEN is not strictly needed for local embeddings, 
# but good to have for future expansions.
```

### 4. Pinecone Setup (Crucial!)
Since we use the `all-MiniLM-L6-v2` model, you must configure your Pinecone Index as follows:
*   **Dimensions:** `384`
*   **Metric:** `Cosine`

### 5. Run the App
```bash
streamlit run app.py
```
Access the app at `http://localhost:8501`.

---

## 🐳 Docker Deployment

The project includes a custom `Dockerfile` optimized for Hugging Face Spaces.

```bash
# Build
docker build -t omnidoc-rag .

# Run (Mapped to port 7860 for HF compatibility)
docker run -it -p 7860:7860 --env-file .env omnidoc-rag
```

---

## 🔄 CI/CD Pipeline (DevOps)

This repository includes a GitHub Actions workflow (`.github/workflows/sync_to_hub.yml`). 

**Automation Flow:**
1.  You push code to the `main` branch on GitHub.
2.  GitHub Actions triggers a runner.
3.  The runner logs into Hugging Face using the `HF_TOKEN` secret.
4.  It force-pushes the code to your Hugging Face Space.
5.  Hugging Face detects the `Dockerfile`, builds the image, and restarts the server on port 7860.

---

## 🔑 Environment Variables Required

To deploy this on Hugging Face Spaces, add these secrets in the **Settings -> Variables and Secrets** tab:

*   `GROQ_API_KEY`: API key from Groq Cloud.
*   `PINECONE_API_KEY`: API key from Pinecone Console.


Built by **Aashir Ali**
