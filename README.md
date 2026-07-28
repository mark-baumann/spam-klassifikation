# Spam-Klassifikation 📧

**Textklassifikation von Grund auf: TF-IDF, Naive Bayes, Logistic Regression**

Lernprojekt zur binären Textklassifikation — der Grundlage für Embeddings, RAG und moderne NLP-Systeme.

## 📦 Features

- **TF-IDF Vectorizer**: Text → numerische Vektoren (Unigrams + Bigrams)
- **Metafeatures**: Nachrichtenlänge, Großbuchstaben, URLs, Telefonnummern
- **Modellvergleich**: Naive Bayes vs. Logistic Regression
- **Fehleranalyse**: False Positives/Negatives mit Beispielen
- **Wichtigste Wörter**: Welche Terms treiben die Spam-Erkennung?

## 🚀 Quickstart

```bash
uv pip install numpy pandas scikit-learn scipy
python spam_classifier.py
```

## 📊 Erwartete Ergebnisse

| Modell | Accuracy | F1-Score |
|---|---|---|
| Naive Bayes | ~97% | ~0.92 |
| Logistic Regression | ~98% | ~0.94 |

## 🧠 Lernziele

1. **Bag-of-Words**: Wie wird Text zu Zahlen?
2. **TF-IDF**: Warum ist "the" weniger wichtig als "free"?
3. **Naive Bayes**: Warum "naive"? (Unabhängigkeitsannahme!)
4. **Precision vs. Recall**: Warum ist Recall bei Spam wichtiger?
5. **Embeddings**: Der Sprung von TF-IDF zu dichten Vektoren
