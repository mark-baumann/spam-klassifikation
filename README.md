# 📧 Spam-Klassifikation — SMS

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3%2B-f7931e.svg)](https://scikit-learn.org/)
[![Status](https://img.shields.io/badge/Status-Aktiv-brightgreen.svg)]()

Automatische **Spam-Erkennung** für SMS mit TF-IDF-Vektorisierung und mehreren Klassifikationsmodellen. Vergleiche Naive Bayes, Logistic Regression und ein Embedding-basiertes neuronales Netz — trainiere Modelle und teste sie interaktiv im Notebook.

## ✨ Features

- **📊 Daten-Exploration** — SMS-Spam-Dataset analysieren: Klassenverteilung, Wortlängen, häufige Begriffe
- **🔤 TF-IDF-Vektorisierung** — Text in numerische Feature-Vektoren umwandeln, Top-Wörter visualisieren
- **🤖 Modellvergleich** — MultinomialNB, Logistic Regression und Embeddings mit neuronalem Netz vergleichen
- **📈 Metriken** — Accuracy, Precision, Recall, F1-Score und Confusion Matrix
- **📊 W&B-Integration** — Experiment-Tracking mit Weights & Biases
- **✅ Getestet** — Unit-Tests für Klassifikator und Feature-Engineering
- **▶️ Google Colab** — Notebook direkt im Browser öffnen und ausführen, kein lokales Setup nötig

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

### ▶️ Google Colab

Das Notebook lässt sich ohne lokales Setup direkt in Google Colab öffnen und ausführen — Klick auf den "Open in Colab"-Badge oben in `spam_klassifikation.ipynb`. Eine Setup-Zelle lädt die benötigten Projektdateien (`spam_classifier.py`, `wandb_utils.py`) automatisch vom GitHub-Repo, alle übrigen Abhängigkeiten (NumPy, Pandas, scikit-learn, TensorFlow) sind in Colab bereits vorinstalliert.

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
├── spam_klassifikation.ipynb   # Interaktives Notebook (Colab-fähig)
├── spam_classifier.py          # TF-IDF, Training, Evaluation
├── wandb_utils.py              # W&B-Integration
├── pyproject.toml              # Projekt-Konfiguration
└── tests/
    └── test_spam_classifier.py
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
