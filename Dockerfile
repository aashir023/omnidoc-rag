# 1. Base Image
FROM python:3.10-slim

# 2. Set working directory
WORKDIR /app

# 3. Create a non-root user (Security Best Practice for Hugging Face)
RUN useradd -m -u 1000 user

# 4. Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 5. Copy the application code (and give ownership to 'user')
COPY --chown=user . .

# 6. Switch to the new user
USER user

# 7. Expose the port
EXPOSE 7860

# 8. Set Environment Variables
ENV GRADIO_SERVER_NAME="0.0.0.0"
ENV GRADIO_SERVER_PORT=7860
ENV PYTHONUNBUFFERED=1

# 9. Run the app
CMD ["python", "app.py"]