FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
COPY wheelhouse /tmp/wheelhouse

RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates git && \
    pip install --no-cache-dir --no-index --find-links=/tmp/wheelhouse -r requirements.txt

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