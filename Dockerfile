# Dockerfile — reines ML-Projekt für E-Mail-Spam-Klassifikation

FROM python:3.12-slim

WORKDIR /app

# System-Abhängigkeiten für Downloads und Python-Pakete
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Python-Abhängigkeiten
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Projektcode
COPY . .

# Standard: E-Mail-Daten ingestieren, Features bauen und ein schnelles Modell trainieren.
# Für vollständige Cross-Validation: docker run <image> python email_classifier.py
CMD ["python", "email_classifier.py", "--quick"]
