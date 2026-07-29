"""
Spam-Klassifikation — Von TF-IDF bis zum Neuronalen Netz
========================================================
Lernprojekt: Textklassifikation Schritt für Schritt.

Ansätze:
1. Naive Bayes + TF-IDF (Baseline)
2. Logistic Regression + TF-IDF
3. Neuronales Netz mit Embeddings

Dataset: SMS Spam Collection (UCI)
"""

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)
from urllib import request
import os
import zipfile


# ═══════════════════════════════════════════════════════════════
# Daten laden
# ═══════════════════════════════════════════════════════════════

def load_spam_data():
    """Lädt das SMS Spam Collection Dataset."""
    cache_dir = os.path.join(os.path.dirname(__file__), ".data_cache")
    os.makedirs(cache_dir, exist_ok=True)

    zip_path = os.path.join(cache_dir, "smsspamcollection.zip")
    extract_path = os.path.join(cache_dir, "smsspamcollection")

    if not os.path.exists(extract_path):
        url = "https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip"
        print("  Lade SMS Spam Collection...")
        request.urlretrieve(url, zip_path)
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(extract_path)

    # Datei finden
    for f in os.listdir(extract_path):
        if f.endswith(".csv") or f.endswith(".txt") or "SMSSpamCollection" in f:
            path = os.path.join(extract_path, f)
            break
    else:
        raise FileNotFoundError("Keine Datendatei gefunden")

    # Einlesen (tab-separiert: label\tmessage)
    df = pd.read_csv(path, sep="\t", header=None, names=["label", "message"])
    df["label"] = (df["label"] == "spam").astype(int)
    return df


# ═══════════════════════════════════════════════════════════════
# Feature Engineering
# ═══════════════════════════════════════════════════════════════

def create_features(df):
    """
    Erstellt Features aus den SMS-Nachrichten:
    - TF-IDF Vektoren (Worthäufigkeit gewichtet)
    - Zusätzliche Metafeatures (Länge, Großbuchstaben, etc.)
    """
    # TF-IDF
    vectorizer = TfidfVectorizer(
        max_features=5000,
        stop_words="english",
        ngram_range=(1, 2),  # Unigrams + Bigrams
    )
    X_tfidf = vectorizer.fit_transform(df["message"])

    # Metafeatures
    meta = pd.DataFrame({
        "length": df["message"].str.len(),
        "num_words": df["message"].str.split().str.len(),
        "num_caps": df["message"].str.count(r"[A-Z]"),
        "num_digits": df["message"].str.count(r"\d"),
        "num_exclamations": df["message"].str.count(r"!"),
        "has_url": df["message"].str.contains(r"http|www\.", regex=True).astype(int),
        "has_phone": df["message"].str.contains(r"\d{3,}").astype(int),
    }).values

    # Kombinieren (sparse + dense)
    from scipy.sparse import hstack, csr_matrix
    X_combined = hstack([X_tfidf, csr_matrix(meta)])

    return X_combined, vectorizer


# ═══════════════════════════════════════════════════════════════
# Modelle trainieren & evaluieren
# ═══════════════════════════════════════════════════════════════

def evaluate_model(name, model, X_train, X_test, y_train, y_test):
    """Trainiert und evaluiert ein Modell."""
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    return {
        "name": name,
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "predictions": y_pred,
    }


# ═══════════════════════════════════════════════════════════════
# Hauptprogramm
# ═══════════════════════════════════════════════════════════════

def main():
    print("=" * 60)
    print("  Spam-Klassifikation — Modellvergleich")
    print("=" * 60)

    # ── Daten laden ──────────────────────────────────────────
    print("\n📦 Lade SMS Spam Collection...")
    df = load_spam_data()
    print(f"   {len(df):,} Nachrichten ({df['label'].sum():,} Spam, "
          f"{len(df) - df['label'].sum():,} Ham)")

    # ── Features ─────────────────────────────────────────────
    print("\n🔧 Extrahiere Features (TF-IDF + Metafeatures)...")
    X, vectorizer = create_features(df)
    y = df["label"].values
    print(f"   Feature-Dimension: {X.shape[1]:,}")

    # ── Train/Test-Split ─────────────────────────────────────
    X_train, X_test, y_train, y_test, train_idx, test_idx = train_test_split(
        X, y, np.arange(len(y)), test_size=0.2, random_state=42, stratify=y
    )
    print(f"   Train: {X_train.shape[0]:,} | Test: {X_test.shape[0]:,}")

    # ── Modelle vergleichen ──────────────────────────────────
    print("\n🤖 Trainiere Modelle...")
    models = [
        ("Naive Bayes", MultinomialNB(alpha=0.1)),
        ("Logistic Regression", LogisticRegression(max_iter=1000, C=1.0)),
    ]

    results = []
    for name, model in models:
        print(f"   {name}...")
        res = evaluate_model(name, model, X_train, X_test, y_train, y_test)
        results.append(res)

    # ── Ergebnisse ────────────────────────────────────────────
    print("\n📊 Ergebnisse:")
    print(f"   {'Modell':<25s} {'Acc':>6s} {'Prec':>6s} {'Rec':>6s} {'F1':>6s}")
    print("   " + "-" * 55)
    for r in results:
        print(f"   {r['name']:<25s} "
              f"{r['accuracy']:6.3f} {r['precision']:6.3f} "
              f"{r['recall']:6.3f} {r['f1']:6.3f}")

    # ── Bestes Modell ────────────────────────────────────────
    best = max(results, key=lambda r: r["f1"])
    best_model = models[[r["name"] for r in results].index(best["name"])][1]
    print(f"\n🏆 Bestes Modell: {best['name']} (F1={best['f1']:.3f})")

    # ── Fehleranalyse ────────────────────────────────────────
    print("\n🔍 Fehleranalyse (bestes Modell):")
    y_pred = best["predictions"]
    errors = np.where(y_test != y_pred)[0]
    print(f"   {len(errors)}/{len(y_test)} Fehler")

    # Zeige 5 False Positives (Ham als Spam klassifiziert)
    fp = errors[y_test[errors] == 0]
    print(f"\n   False Positives (Ham → Spam): {len(fp)}")
    for i in fp[:3]:
        orig_idx = test_idx[i]
        print(f"   • \"{df.iloc[orig_idx]['message'][:80]}...\"")

    # Zeige 5 False Negatives (Spam als Ham klassifiziert)
    fn = errors[y_test[errors] == 1]
    print(f"\n   False Negatives (Spam → Ham): {len(fn)}")
    for i in fn[:3]:
        orig_idx = test_idx[i]
        print(f"   • \"{df.iloc[orig_idx]['message'][:80]}...\"")

    # ── Wichtigste Wörter ────────────────────────────────────
    if best["name"] == "Logistic Regression":
        print("\n📝 Top-Spam-Wörter (Logistic Regression):")
        coef = best_model.coef_[0]
        # Nur TF-IDF-Koeffizienten (erste 5000), nicht die Metafeatures
        n_tfidf = len(vectorizer.get_feature_names_out())
        top_idx = np.argsort(coef[:n_tfidf])[-10:][::-1]
        feature_names = vectorizer.get_feature_names_out()
        for idx in top_idx:
            print(f"   • {feature_names[idx]:20s} ({coef[idx]:.3f})")

    print("\n✅ Analyse abgeschlossen!")


if __name__ == "__main__":
    main()
