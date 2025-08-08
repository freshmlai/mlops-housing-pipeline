FROM python:3.10-slim

WORKDIR /app

COPY src/requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ .

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080"]
