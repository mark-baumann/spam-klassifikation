# 📧 Spam-Klassifikation — SMS & E-Mail

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3%2B-f7931e.svg)](https://scikit-learn.org/)
[![Status](https://img.shields.io/badge/Status-Aktiv-brightgreen.svg)]()

Automatische **Spam-Erkennung** für SMS und E-Mails mit TF-IDF-Vektorisierung und mehreren Klassifikationsmodellen. Vergleiche Naive Bayes, Logistic Regression und LinearSVC — trainiere Modelle und teste sie programmatisch.

## ✨ Features

- **📊 Daten-Exploration** — SMS-Spam-Dataset analysieren: Klassenverteilung, Wortlängen, häufige Begriffe
- **🔤 TF-IDF-Vektorisierung** — Text in numerische Feature-Vektoren umwandeln, Top-Wörter visualisieren
- **🤖 Modellvergleich** — MultinomialNB, Logistic Regression und LinearSVC mit Metriken vergleichen
- **📈 Metriken** — Accuracy, Precision, Recall, F1-Score und Confusion Matrix
- **📊 W&B-Integration** — Experiment-Tracking mit Weights & Biases
- **✅ Vollständig getestet** — Unit-Tests für Klassifikatoren und Feature-Engineering

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

## 🎯 Nutzung

```bash
# Spam-Klassifikation trainieren und evaluieren
python spam_classifier.py

# Tests ausführen
pytest tests/ -v
```

## 🧪 Tests ausführen

```bash
pytest tests/ -v
```

## 🛠️ Tech-Stack

| Technologie | Einsatz |
|-------------|---------|
| **scikit-learn** | TF-IDF, Naive Bayes, Logistic Regression, LinearSVC, Metriken |
| **NumPy** | Numerische Operationen |
| **Pandas** | Daten-Management und -Analyse |
| **SciPy** | Sparse-Matrix-Operationen |
| **Matplotlib** | Visualisierung von Metriken und Features |
| **Seaborn** | Confusion-Matrix-Heatmaps |
| **Weights & Biases** | Experiment-Tracking |
| **Pytest** | Test-Framework |

## 📁 Projektstruktur

```
spam-klassifikation/
├── spam_classifier.py          # TF-IDF, Training, Evaluation
├── email_classifier.py         # E-Mail-spezifische Klassifikation
├── email_data.py               # E-Mail-Datengenerierung
├── wandb_utils.py              # W&B-Integration
├── pyproject.toml              # Projekt-Konfiguration
└── tests/
    ├── test_spam_classifier.py
    └── test_email_classifier.py
```

## 📖 Wie funktioniert Spam-Erkennung?

1. **Textvorverarbeitung** — Tokenisierung, Stopwort-Entfernung
2. **TF-IDF** — Wörter in numerische Gewichte umwandeln („wie wichtig ist ein Wort in diesem Dokument?")
3. **Klassifikation** — Modell sagt „Spam" oder „Ham" (kein Spam)
4. **Evaluation** — Precision/Recall zeigen, wie gut das Modell generalisiert

## 👤 Autor

**Mark Baumann** — [GitHub](https://github.com/mark-baumann)

---

*Spam-Erkennung ist eine der klassischsten NLP-Anwendungen — ideal, um Textklassifikation von Grund auf zu verstehen.*
