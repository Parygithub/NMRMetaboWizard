FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN python -m pip install --no-cache-dir --upgrade pip \
    && python -m pip install --no-cache-dir -r requirements.txt

COPY app.py clinical_analysis.py nmr_pipeline.py ./

EXPOSE 8000

CMD ["python", "-m", "shiny", "run", "--host", "0.0.0.0", "--port", "8000", "app.py"]
