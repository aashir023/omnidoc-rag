# 1. Base Image
FROM python:3.10-slim

# 2. Set working directory
WORKDIR /app

# 3. Create a non-root user (Required for Hugging Face Spaces)
RUN useradd -m -u 1000 user

# 4. Set environment variables to fix permission issues with Transformers
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    TRANSFORMERS_CACHE=/home/user/.cache/huggingface \
    HF_HOME=/home/user/.cache/huggingface

# 5. Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 6. Copy the application code (and give ownership to 'user')
COPY --chown=user . .

# 7. Switch to the new user
USER user

# 8. Create cache directory explicitly (to avoid permission issues)
RUN mkdir -p /home/user/.cache/huggingface

# 9. Expose the port (Hugging Face expects 7860)
EXPOSE 7860

# 10. Run the app with CORS and XSRF disabled to fix the 403 Error
CMD ["streamlit", "run", "app.py", \
    "--server.port=7860", \
    "--server.address=0.0.0.0", \
    "--server.enableCORS=false", \
    "--server.enableXsrfProtection=false"]