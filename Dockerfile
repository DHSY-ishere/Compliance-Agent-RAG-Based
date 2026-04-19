FROM python:3.11-slim
WORKDIR /app
ENV PIP_DEFAULT_TIMEOUT=300
COPY requirements.txt .
RUN python -m pip install --isolated --no-cache-dir --retries 10 -r requirements.txt
COPY . .
EXPOSE 8000
CMD ["uvicorn", "main:api", "--host", "0.0.0.0", "--port", "8000"]
