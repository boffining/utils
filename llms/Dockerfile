FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .

RUN apt-get update && apt-get install -y --no-install-recommends git ca-certificates curl iputils-ping && \
    pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir --upgrade certifi

COPY utils.py .
COPY metrics.py .
COPY metric_tracker.py .
COPY test.py .
COPY main.py .

# # Set Hugging Face Hub environment variables (for authentication)
# #ENV HF_TOKEN=$HF_TOKEN
# ENV REQUESTS_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt
# ENV SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt

# # Run the unit tests using pytest.
# CMD ["pytest", "main.py"]
CMD ["python", "main.py"]