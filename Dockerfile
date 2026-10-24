FROM python:3.11-slim

# OCR local des PDF scannés (français + anglais)
RUN apt-get update && apt-get install -y --no-install-recommends tesseract-ocr tesseract-ocr-fra tesseract-ocr-eng \
    && rm -rf /var/lib/apt/lists/*

# Hugging Face Spaces exécute le conteneur avec l'utilisateur 1000
RUN useradd -m -u 1000 akori && mkdir -p /data && chown akori /data
USER akori
ENV HOME=/home/akori PATH=/home/akori/.local/bin:$PATH \
    HF_HOME=/home/akori/.cache/huggingface \
    AKORI_PUBLIC=1 AKORI_DATA_DIR=/data PORT=7860 PYTHONUNBUFFERED=1
WORKDIR /home/akori/app

COPY --chown=akori requirements.txt .
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r requirements.txt

# Le modèle d'embeddings est téléchargé une fois pour toutes pendant la construction (démarrage rapide ensuite)
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('BAAI/bge-m3')"

COPY --chown=akori akori.py .
EXPOSE 7860
CMD ["python", "akori.py"]
