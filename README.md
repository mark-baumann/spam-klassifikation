# 📧 Spam-Klassifikation — ML-Pipeline für E-Mail-Ingestion

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3%2B-f7931e.svg)](https://scikit-learn.org/)
[![Status](https://img.shields.io/badge/Status-ML--Pipeline-brightgreen.svg)]()

Dieses Repository ist ein **reines Machine-Learning-Projekt ohne UI**. Es enthält eine reproduzierbare Pipeline, die rohe E-Mails ingestiert, RFC-822/MIME-Inhalte parst, Leakage-gefährliche Header entfernt, Text- und Metafeatures erstellt und Spam/Ham-Modelle mit scikit-learn trainiert.

## ✨ Features

- **📥 E-Mail-Ingestion** — Download und Cache des SpamAssassin Public Corpus
- **✉️ RFC-822/MIME-Parsing** — Betreff, Body, HTML-Text, Anhänge und Empfängerstruktur extrahieren
- **🛡️ Leakage-Schutz** — Received-, X-Spam-, List- und weitere sammlungsspezifische Header werden ignoriert
- **🔤 Feature-Pipeline** — TF-IDF für Subject/Body, optionale Zeichen-n-Gramme und skalierte Metafeatures
- **🤖 Modellvergleich** — MultinomialNB, ComplementNB, Logistic Regression und LinearSVC
- **🎚️ Schwellenkalibrierung** — Betriebspunkt mit hoher Precision für produktionsnähere Spamfilter
- **💾 Modell-Artefakte** — trainierte Pipeline wird als `models/email_spam_model.joblib` gespeichert
- **📊 W&B-Integration** — optionales Experiment-Tracking mit Weights & Biases
- **✅ Tests** — Unit-Tests für Datenparser, Feature Engineering, Training und Inferenz

## 🚀 Installation

```bash
# Repository klonen
git clone https://github.com/mark-baumann/spam-klassifikation.git
cd spam-klassifikation

# Virtuelle Umgebung erstellen
uv venv
source .venv/bin/activate  # Linux/macOS
# .venv\Scripts\activate   # Windows

# Abhängigkeiten installieren
uv pip install -e ".[dev]"
```

Alternativ mit pip:

```bash
pip install -r requirements.txt
```

## 🔁 Pipeline ausführen

```bash
# Vollständige E-Mail-Pipeline: Ingestion, Feature Engineering, Modellvergleich, Training, Evaluation, Speichern
python email_classifier.py

# Schneller Lauf ohne Cross-Validation-Modellvergleich
python email_classifier.py --quick

# Ohne Zeichen-n-Gramme trainieren (schneller, potenziell weniger robust gegen obfuskierten Spam)
python email_classifier.py --no-char-ngrams

# Metriken optional zu Weights & Biases loggen
python email_classifier.py --wandb
```

Die Pipeline lädt den SpamAssassin-Korpus bei Bedarf nach `.data_cache/spamassassin/`, parst die E-Mails und speichert das trainierte Modell unter `models/email_spam_model.joblib`.

## 🔮 Inferenz

```bash
# Rohe .eml-Datei klassifizieren
python email_classifier.py --predict path/to/mail.eml

# Freien Text als E-Mail-Body klassifizieren
python email_classifier.py --subject "Sonderangebot" --text "Click here to win money now"
```

## 🐳 Docker

```bash
# Image bauen
docker build -t spam-klassifikation .

# Standard: schneller Pipeline-Lauf
docker run --rm spam-klassifikation

# Vollständigen Modellvergleich starten
docker run --rm spam-klassifikation python email_classifier.py
```

## 🧪 Tests ausführen

```bash
pytest tests/ -v
```

## 🛠️ Tech-Stack

| Technologie | Einsatz |
|-------------|---------|
| **scikit-learn** | TF-IDF, Pipelines, Modelltraining, Cross-Validation, Metriken |
| **Pandas** | Daten-Management und Feature Engineering |
| **NumPy** | Numerische Operationen |
| **SciPy** | Sparse-Matrix-Operationen |
| **joblib** | Modellpersistenz |
| **Weights & Biases** | Optionales Experiment-Tracking |
| **Pytest** | Test-Framework |

## 📁 Projektstruktur

```text
spam-klassifikation/
├── Dockerfile                  # Container für CLI-/Pipeline-Ausführung
├── pyproject.toml              # Projekt-Konfiguration
├── requirements.txt            # Runtime-Abhängigkeiten ohne UI-Pakete
├── email_data.py               # E-Mail-Download, Cache, Parsing und Ingestion
├── email_classifier.py         # ML-Pipeline, Training, Evaluation und Inferenz
├── spam_classifier.py          # SMS-Baseline für klassische Textklassifikation
├── wandb_utils.py              # W&B-Integration
└── tests/
    ├── test_email_classifier.py
    └── test_spam_classifier.py
```

## 📖 Pipeline-Überblick

1. **Ingestion** — `email_data.download_corpus()` lädt die SpamAssassin-Archive und legt sie im lokalen Cache ab.
2. **Parsing** — `email_data.parse_email()` extrahiert robuste, produktionsnahe Felder aus rohen E-Mails.
3. **Feature Engineering** — `email_classifier.add_meta_features()` ergänzt Struktur- und Textmetafeatures; `build_feature_union()` erstellt TF-IDF-Features.
4. **Training & Evaluation** — `email_classifier.train()` vergleicht Modelle, kalibriert die Spam-Schwelle und evaluiert auf einem Holdout-Testset.
5. **Persistenz & Inferenz** — `load_model()`, `classify()` und `predict_file()` laden das Artefakt und klassifizieren neue E-Mails.

## 👤 Autor

**Mark Baumann** — [GitHub](https://github.com/mark-baumann)

---

*Spam-Erkennung ist eine klassische NLP-Anwendung — dieses Projekt fokussiert jetzt bewusst auf Daten-Ingestion, Feature Engineering und reproduzierbare ML-Pipelines statt auf eine Web-UI.*
