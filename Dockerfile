FROM python:3.12-slim
WORKDIR /srv
ENV PYTHONUNBUFFERED=1
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY scraper/download_models.py ./scraper/download_models.py
ENV MODEL_CACHE=/srv/models
RUN python scraper/download_models.py
COPY app ./app
COPY data ./data
COPY scraper ./scraper
ENV PORT=7860
EXPOSE 7860
CMD ["sh","-c","uvicorn app.main:app --host 0.0.0.0 --port ${PORT}"]
