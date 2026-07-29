"""
Streamlit-App: Spam-Klassifikation
===================================
Text eingeben → Spam/Ham vorhersagen, TF-IDF visualisieren, Modellvergleich.
"""

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))
from spam_classifier import load_spam_data, create_features, evaluate_model
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

st.set_page_config(page_title="Spam-Klassifikation", page_icon="📧", layout="wide")
st.title("📧 Spam-Klassifikation")
st.markdown("Text eingeben → Spam/Ham vorhersagen | TF-IDF visualisieren | Modelle vergleichen")

page = st.sidebar.radio(
    "Bereich wählen",
    ["Spam-Prüfung", "Modellvergleich", "TF-IDF Visualisierung", "Daten-Exploration"]
)

# ═══════════════════════════════════════════════════════════════════════════
# DATEN LADEN & MODELL TRAINIEREN (Cache)
# ═══════════════════════════════════════════════════════════════════════════

@st.cache_data
def load_data():
    return load_spam_data()

@st.cache_resource
def train_models():
    df = load_data()
    X, vectorizer = create_features(df)
    y = df["label"].values
    X_train, X_test, y_train, y_test, _, _ = train_test_split(
        X, y, np.arange(len(y)), test_size=0.2, random_state=42, stratify=y
    )

    models = {
        "Naive Bayes": MultinomialNB(alpha=0.1),
        "Logistic Regression": LogisticRegression(max_iter=1000, C=1.0),
    }

    results = {}
    for name, model in models.items():
        res = evaluate_model(name, model, X_train, X_test, y_train, y_test)
        results[name] = {
            "model": model,
            "accuracy": res["accuracy"],
            "precision": res["precision"],
            "recall": res["recall"],
            "f1": res["f1"],
        }

    return results, vectorizer, df, X_test, y_test

# ═══════════════════════════════════════════════════════════════════════════
# SPAM-PRÜFUNG
# ═══════════════════════════════════════════════════════════════════════════
if page == "Spam-Prüfung":
    st.header("🔍 Spam oder Ham?")

    with st.spinner("Lade Modell..."):
        results, vectorizer, df, _, _ = train_models()

    model_choice = st.selectbox("Modell wählen", list(results.keys()))
    model = results[model_choice]["model"]

    st.subheader("Nachricht eingeben")
    user_text = st.text_area(
        "Text",
        "Congratulations! You've won a FREE iPhone! Click here to claim your prize now!!!",
        height=150
    )

    if st.button("Prüfen", type="primary"):
        # Features extrahieren
        X_tfidf = vectorizer.transform([user_text])

        # Metafeatures
        meta = np.array([[
            len(user_text),
            len(user_text.split()),
            sum(1 for c in user_text if c.isupper()),
            sum(1 for c in user_text if c.isdigit()),
            user_text.count("!"),
            int(any(w in user_text.lower() for w in ["http", "www."])),
            int(any(c.isdigit() for c in user_text) and sum(1 for c in user_text if c.isdigit()) >= 3),
        ]])

        from scipy.sparse import hstack, csr_matrix
        X_combined = hstack([X_tfidf, csr_matrix(meta)])

        prediction = model.predict(X_combined)[0]
        proba = model.predict_proba(X_combined)[0] if hasattr(model, "predict_proba") else None

        col1, col2 = st.columns(2)
        with col1:
            if prediction == 1:
                st.error("🚫 **SPAM**")
            else:
                st.success("✅ **HAM** (Kein Spam)")

        with col2:
            if proba is not None:
                st.metric("Spam-Wahrscheinlichkeit", f"{proba[1]:.2%}")
                st.metric("Ham-Wahrscheinlichkeit", f"{proba[0]:.2%}")

        st.subheader("Text-Analyse")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Länge", f"{len(user_text)} Zeichen")
        with col2:
            st.metric("Wörter", len(user_text.split()))
        with col3:
            st.metric("Großbuchstaben", sum(1 for c in user_text if c.isupper()))
        with col4:
            st.metric("Ausrufezeichen", user_text.count("!"))

# ═══════════════════════════════════════════════════════════════════════════
# MODELLVERGLEICH
# ═══════════════════════════════════════════════════════════════════════════
elif page == "Modellvergleich":
    st.header("📊 Modellvergleich")

    with st.spinner("Trainiere Modelle..."):
        results, vectorizer, df, _, _ = train_models()

    st.subheader("Ergebnisse")

    # Tabelle
    data = []
    for name, res in results.items():
        data.append({
            "Modell": name,
            "Accuracy": f"{res['accuracy']:.3f}",
            "Precision": f"{res['precision']:.3f}",
            "Recall": f"{res['recall']:.3f}",
            "F1-Score": f"{res['f1']:.3f}",
        })
    st.dataframe(pd.DataFrame(data), use_container_width=True)

    # Bar-Chart
    st.subheader("Metriken im Vergleich")
    fig, ax = plt.subplots(figsize=(10, 5))
    metrics = ["accuracy", "precision", "recall", "f1"]
    x = np.arange(len(metrics))
    width = 0.35

    for i, (name, res) in enumerate(results.items()):
        values = [res[m] for m in metrics]
        ax.bar(x + i * width, values, width, label=name, alpha=0.8)

    ax.set_ylabel("Score")
    ax.set_title("Modellvergleich: Naive Bayes vs. Logistic Regression")
    ax.set_xticks(x + width / 2)
    ax.set_xticklabels(["Accuracy", "Precision", "Recall", "F1"])
    ax.legend()
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)
    st.pyplot(fig)

    st.subheader("Modell-Details")
    model_choice = st.selectbox("Modell", list(results.keys()), key="detail_model")
    res = results[model_choice]

    st.markdown(f"""
    - **Accuracy:** {res['accuracy']:.3f} — Anteil korrekter Vorhersagen
    - **Precision:** {res['precision']:.3f} — Wie viele als Spam markierte Nachrichten sind wirklich Spam?
    - **Recall:** {res['recall']:.3f} — Wie viele echte Spam-Nachrichten wurden erkannt?
    - **F1-Score:** {res['f1']:.3f} — Harmonisches Mittel aus Precision und Recall
    """)

# ═══════════════════════════════════════════════════════════════════════════
# TF-IDF VISUALISIERUNG
# ═══════════════════════════════════════════════════════════════════════════
elif page == "TF-IDF Visualisierung":
    st.header("📝 TF-IDF Visualisierung")

    with st.spinner("Lade Daten..."):
        results, vectorizer, df, _, _ = train_models()

    st.markdown("""
    **TF-IDF** (Term Frequency — Inverse Document Frequency) gewichtet Wörter danach,
    wie wichtig sie für ein Dokument sind:
    
    - **TF:** Wie oft kommt das Wort im Dokument vor?
    - **IDF:** In wie vielen Dokumenten kommt das Wort vor? (seltene Wörter → höheres Gewicht)
    """)

    st.subheader("Top-Spam-Wörter (Logistic Regression)")
    lr_model = results["Logistic Regression"]["model"]
    coef = lr_model.coef_[0]
    n_tfidf = len(vectorizer.get_feature_names_out())
    feature_names = vectorizer.get_feature_names_out()

    top_n = st.slider("Top N Wörter", 5, 30, 15)
    top_idx = np.argsort(coef[:n_tfidf])[-top_n:][::-1]

    fig, ax = plt.subplots(figsize=(10, 6))
    words = [feature_names[i] for i in top_idx]
    weights = [coef[i] for i in top_idx]
    colors = ["#FF6B6B" if w > 0 else "#4ECDC4" for w in weights]

    ax.barh(range(len(words)), weights, color=colors, edgecolor="white")
    ax.set_yticks(range(len(words)))
    ax.set_yticklabels(words)
    ax.set_xlabel("Koeffizient (positiv = Spam-Indikator)")
    ax.set_title("Top-Spam-Wörter nach Logistic-Regression-Koeffizienten")
    ax.invert_yaxis()
    ax.grid(True, alpha=0.3, axis="x")
    st.pyplot(fig)

    st.subheader("Beispiel: TF-IDF für eine Nachricht")
    example_text = st.text_area(
        "Nachricht",
        "URGENT! You have won a free vacation to Bahamas! Call now to claim your prize!",
        height=100,
        key="tfidf_example"
    )

    if st.button("TF-IDF berechnen"):
        X_example = vectorizer.transform([example_text])
        # Finde nicht-null Features
        nonzero = X_example.nonzero()[1]
        if len(nonzero) > 0:
            values = X_example[0, nonzero].toarray()[0]
            words = [feature_names[i] for i in nonzero]

            # Sortiere nach TF-IDF-Wert
            sorted_idx = np.argsort(values)[::-1][:20]

            fig, ax = plt.subplots(figsize=(10, 5))
            ax.barh(range(len(sorted_idx)), values[sorted_idx], color="#4ECDC4", edgecolor="white")
            ax.set_yticks(range(len(sorted_idx)))
            ax.set_yticklabels([words[i] for i in sorted_idx])
            ax.set_xlabel("TF-IDF-Wert")
            ax.set_title("TF-IDF-Werte für die eingegebene Nachricht")
            ax.invert_yaxis()
            st.pyplot(fig)
        else:
            st.warning("Keine Features gefunden (Wörter nicht im Vokabular)")

# ═══════════════════════════════════════════════════════════════════════════
# DATEN-EXPLORATION
# ═══════════════════════════════════════════════════════════════════════════
elif page == "Daten-Exploration":
    st.header("📦 Daten-Exploration")

    with st.spinner("Lade Daten..."):
        df = load_data()

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Gesamt-Nachrichten", f"{len(df):,}")
    with col2:
        spam_count = df["label"].sum()
        st.metric("Spam", f"{spam_count:,} ({spam_count/len(df):.1%})")
    with col3:
        ham_count = len(df) - spam_count
        st.metric("Ham", f"{ham_count:,} ({ham_count/len(df):.1%})")

    st.subheader("Klassenverteilung")
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.pie([ham_count, spam_count], labels=["Ham", "Spam"], autopct="%1.1f%%",
           colors=["#4ECDC4", "#FF6B6B"], startangle=90, explode=(0, 0.05))
    ax.set_title("Spam vs. Ham Verteilung")
    st.pyplot(fig)

    st.subheader("Beispiel-Nachrichten")
    category = st.radio("Kategorie", ["Spam", "Ham", "Beide"])

    if category == "Spam":
        samples = df[df["label"] == 1]["message"].head(10)
    elif category == "Ham":
        samples = df[df["label"] == 0]["message"].head(10)
    else:
        samples = df["message"].head(10)

    for i, msg in enumerate(samples):
        label = "🚫 SPAM" if df.iloc[i]["label"] == 1 else "✅ HAM"
        st.markdown(f"**{label}:** {msg[:200]}{'...' if len(msg) > 200 else ''}")

st.sidebar.markdown("---")
st.sidebar.markdown("📚 **Spam-Klassifikation**")
st.sidebar.markdown("[GitHub Repository](https://github.com/mark-baumann/spam-klassifikation)")
