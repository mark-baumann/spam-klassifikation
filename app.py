"""
Streamlit-App: Spam-Klassifikation
==================================
Text eingeben → Spam/Ham vorhersagen, TF-IDF visualisieren, Modellvergleich.
"""

import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

sys.path.insert(0, os.path.dirname(__file__))
from spam_classifier import create_features, load_spam_data

st.set_page_config(page_title="Spam-Klassifikation", layout="wide")
st.title("📧 Spam-Klassifikation")
st.markdown("### Text eingeben → Spam oder Ham? Mit TF-IDF & Modellvergleich")

tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Daten & TF-IDF", "🤖 Modellvergleich", "🔮 Live-Vorhersage", "📝 Top-Wörter"
])

# ═══════════════════════════════════════════════════════════════
# Session State
# ═══════════════════════════════════════════════════════════════
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False
if 'models_trained' not in st.session_state:
    st.session_state.models_trained = False

# ═══════════════════════════════════════════════════════════════
# Tab 1: Daten & TF-IDF
# ═══════════════════════════════════════════════════════════════
with tab1:
    st.header("📊 Daten & TF-IDF")

    if st.button("📦 SMS Spam Collection laden", type="primary"):
        with st.spinner("Lade Datensatz..."):
            df = load_spam_data()
            st.session_state.df = df
            st.session_state.data_loaded = True
        st.success(f"✅ {len(df):,} Nachrichten geladen!")

    if st.session_state.data_loaded:
        df = st.session_state.df

        col_a, col_b, col_c = st.columns(3)
        col_a.metric("Gesamt-Nachrichten", f"{len(df):,}")
        col_b.metric("Spam", f"{df['label'].sum():,} ({df['label'].mean():.1%})")
        col_c.metric("Ham", f"{(df['label'] == 0).sum():,} ({(1-df['label'].mean()):.1%})")

        st.subheader("Beispiel-Nachrichten")
        col_s1, col_s2 = st.columns(2)
        with col_s1:
            st.markdown("**📛 Spam-Beispiele:**")
            spam_samples = df[df['label'] == 1]['message'].sample(min(5, df['label'].sum())).tolist()
            for s in spam_samples:
                st.markdown(f"> 🔴 {s[:150]}")
        with col_s2:
            st.markdown("**✅ Ham-Beispiele:**")
            ham_samples = df[df['label'] == 0]['message'].sample(5).tolist()
            for h in ham_samples:
                st.markdown(f"> 🟢 {h[:150]}")

        st.subheader("Nachrichtenlänge: Spam vs. Ham")
        fig_len, ax_len = plt.subplots(figsize=(10, 4))
        df_spam_len = df[df['label'] == 1]['message'].str.len()
        df_ham_len = df[df['label'] == 0]['message'].str.len()
        ax_len.hist(df_ham_len, bins=50, alpha=0.6, label='Ham', color='green')
        ax_len.hist(df_spam_len, bins=50, alpha=0.6, label='Spam', color='red')
        ax_len.set_xlabel('Nachrichtenlänge (Zeichen)')
        ax_len.set_ylabel('Anzahl')
        ax_len.set_title('Verteilung der Nachrichtenlänge', fontweight='bold')
        ax_len.legend()
        ax_len.set_xlim(0, 300)
        st.pyplot(fig_len)

        st.subheader("TF-IDF verstehen")
        st.markdown(r"""
        **TF-IDF** (Term Frequency — Inverse Document Frequency) gewichtet Wörter danach,
        wie wichtig sie für ein Dokument sind:

        - **TF (Term Frequency):** Wie oft kommt ein Wort in dieser Nachricht vor?
        - **IDF (Inverse Document Frequency):** In wie vielen Nachrichten kommt das Wort vor?
          → Seltene Wörter bekommen höheres Gewicht!

        **Formel:** $\text{TF-IDF}(t, d) = \text{TF}(t, d) \times \log\frac{N}{\text{DF}(t)}$
        """)

        if st.button("🔬 TF-IDF für Beispiel-Nachricht berechnen"):
            sample_msg = df['message'].iloc[0]
            st.code(f"Nachricht: \"{sample_msg}\"")

            vectorizer = TfidfVectorizer(max_features=20, stop_words='english')
            tfidf_matrix = vectorizer.fit_transform([sample_msg])
            feature_names = vectorizer.get_feature_names_out()
            tfidf_scores = tfidf_matrix.toarray()[0]

            # Nur Wörter mit Score > 0
            nonzero = tfidf_scores > 0
            if np.any(nonzero):
                fig_tfidf, ax_tfidf = plt.subplots(figsize=(10, 4))
                indices = np.argsort(tfidf_scores[nonzero])[::-1]
                words = feature_names[nonzero][indices]
                scores = tfidf_scores[nonzero][indices]
                ax_tfidf.barh(range(len(words)), scores)
                ax_tfidf.set_yticks(range(len(words)))
                ax_tfidf.set_yticklabels(words)
                ax_tfidf.set_xlabel('TF-IDF Score')
                ax_tfidf.set_title('TF-IDF Scores für Beispiel-Nachricht', fontweight='bold')
                ax_tfidf.invert_yaxis()
                st.pyplot(fig_tfidf)
            else:
                st.info("Keine Wörter mit TF-IDF > 0 gefunden (alle Stopwords?)")

# ═══════════════════════════════════════════════════════════════
# Tab 2: Modellvergleich
# ═══════════════════════════════════════════════════════════════
with tab2:
    st.header("🤖 Modellvergleich")

    if not st.session_state.data_loaded:
        st.warning("⚠️ Bitte zuerst die Daten laden (Tab 1)!")
    else:
        if st.button("🚀 Modelle trainieren & vergleichen", type="primary"):
            df = st.session_state.df

            with st.spinner("Extrahiere Features (TF-IDF + Metafeatures)..."):
                X, vectorizer = create_features(df)
                y = df['label'].values

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )

            st.info(f"Feature-Dimension: **{X.shape[1]:,}** | Train: {X_train.shape[0]:,} | Test: {X_test.shape[0]:,}")

            # Modelle trainieren
            models = {
                "Naive Bayes": MultinomialNB(alpha=0.1),
                "Logistic Regression": LogisticRegression(max_iter=1000, C=1.0),
            }

            results = []
            progress_bar = st.progress(0)

            for i, (name, model) in enumerate(models.items()):
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                results.append({
                    "name": name,
                    "model": model,
                    "accuracy": accuracy_score(y_test, y_pred),
                    "precision": precision_score(y_test, y_pred),
                    "recall": recall_score(y_test, y_pred),
                    "f1": f1_score(y_test, y_pred),
                    "y_pred": y_pred,
                })
                progress_bar.progress((i + 1) / len(models))

            st.session_state.results = results
            st.session_state.y_test = y_test
            st.session_state.vectorizer = vectorizer
            st.session_state.X_test = X_test
            st.session_state.models_trained = True

            # Ergebnisse-Tabelle
            st.subheader("📊 Ergebnisse")
            result_df = pd.DataFrame([
                {"Modell": r['name'], "Accuracy": f"{r['accuracy']:.3f}",
                 "Precision": f"{r['precision']:.3f}", "Recall": f"{r['recall']:.3f}",
                 "F1-Score": f"{r['f1']:.3f}"}
                for r in results
            ])
            st.dataframe(result_df.set_index('Modell'), use_container_width=True)

            # Visualisierung
            fig_comp, ax_comp = plt.subplots(figsize=(8, 5))
            metrics = ['accuracy', 'precision', 'recall', 'f1']
            x = np.arange(len(metrics))
            width = 0.35
            for i, r in enumerate(results):
                values = [r[m] for m in metrics]
                ax_comp.bar(x + i * width, values, width, label=r['name'])
            ax_comp.set_ylabel('Score')
            ax_comp.set_title('Modellvergleich', fontweight='bold')
            ax_comp.set_xticks(x + width / 2)
            ax_comp.set_xticklabels(['Accuracy', 'Precision', 'Recall', 'F1'])
            ax_comp.legend()
            ax_comp.set_ylim(0.8, 1.0)
            st.pyplot(fig_comp)

            # Confusion Matrices
            st.subheader("Confusion Matrices")
            fig_cms, axes_cms = plt.subplots(1, 2, figsize=(12, 5))
            for ax, r in zip(axes_cms, results):
                cm = confusion_matrix(y_test, r['y_pred'])
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                            xticklabels=['Ham', 'Spam'], yticklabels=['Ham', 'Spam'])
                ax.set_title(r['name'], fontweight='bold')
                ax.set_xlabel('Vorhergesagt')
                ax.set_ylabel('Tatsächlich')
            plt.tight_layout()
            st.pyplot(fig_cms)

            # Bestes Modell speichern
            best = max(results, key=lambda r: r['f1'])
            st.session_state.best_model = best['model']
            st.session_state.best_name = best['name']
            st.success(f"🏆 Bestes Modell: **{best['name']}** (F1={best['f1']:.3f})")

# ═══════════════════════════════════════════════════════════════
# Tab 3: Live-Vorhersage
# ═══════════════════════════════════════════════════════════════
with tab3:
    st.header("🔮 Live-Vorhersage")

    if not st.session_state.models_trained:
        st.warning("⚠️ Bitte zuerst die Modelle trainieren (Tab 2)!")
    else:
        st.markdown("Gib einen beliebigen Text ein und lass ihn klassifizieren:")

        user_text = st.text_area(
            "Nachricht eingeben:",
            placeholder="z.B. 'Congratulations! You have won a free iPhone. Click here to claim your prize!'",
            height=100,
        )

        col_ex1, col_ex2 = st.columns(2)
        with col_ex1:
            if st.button("📝 Spam-Beispiel laden"):
                spam_examples = [
                    "URGENT! You have won a £2,000 prize! Call 09050000000 NOW to claim. Valid 24 hours only!",
                    "FREE entry in a weekly competition to win an iPad. Just text WIN to 80085 now!",
                    "Congratulations! You've been selected for a exclusive VIP offer. Click http://spam.com/win",
                ]
                rng = np.random.default_rng(seed=42)
                st.session_state.user_text = rng.choice(spam_examples)
                st.rerun()
        with col_ex2:
            if st.button("📝 Ham-Beispiel laden"):
                ham_examples = [
                    "Hey, are we still meeting for lunch tomorrow at 12?",
                    "Don't forget to pick up milk on your way home.",
                    "The report is ready for review. Please check your email.",
                ]
                rng = np.random.default_rng(seed=42)
                st.session_state.user_text = rng.choice(ham_examples)
                st.rerun()

        if user_text:
            st.subheader("Vorhersage-Ergebnis")

            # Features für den eingegebenen Text
            vectorizer = st.session_state.vectorizer
            best_model = st.session_state.best_model
            best_name = st.session_state.best_name

            # TF-IDF transformieren
            text_tfidf = vectorizer.transform([user_text])

            # Metafeatures
            meta = np.array([[
                len(user_text),
                len(user_text.split()),
                sum(1 for c in user_text if c.isupper()),
                sum(1 for c in user_text if c.isdigit()),
                user_text.count('!'),
                1 if any(w in user_text.lower() for w in ['http', 'www.']) else 0,
                1 if any(c.isdigit() for c in user_text) and len([c for c in user_text if c.isdigit()]) >= 3 else 0,
            ]])

            from scipy.sparse import csr_matrix, hstack
            X_input = hstack([text_tfidf, csr_matrix(meta)])

            # Vorhersage
            proba = best_model.predict_proba(X_input)[0]
            pred = best_model.predict(X_input)[0]

            col_r1, col_r2 = st.columns(2)
            with col_r1:
                if pred == 1:
                    st.error("### 🔴 SPAM")
                else:
                    st.success("### 🟢 HAM (Kein Spam)")
                st.caption(f"Modell: {best_name}")

            with col_r2:
                fig_proba, ax_proba = plt.subplots(figsize=(4, 3))
                labels = ['Ham', 'Spam']
                colors = ['green', 'red']
                ax_proba.bar(labels, proba, color=colors)
                ax_proba.set_ylabel('Wahrscheinlichkeit')
                ax_proba.set_title('Klassen-Wahrscheinlichkeiten', fontweight='bold')
                ax_proba.set_ylim(0, 1)
                for i, p in enumerate(proba):
                    ax_proba.text(i, p + 0.02, f'{p:.1%}', ha='center', fontweight='bold')
                st.pyplot(fig_proba)

            # TF-IDF Top-Wörter im eingegebenen Text
            st.subheader("Wichtigste Wörter im Text (TF-IDF)")
            tfidf_dense = text_tfidf.toarray()[0]
            feature_names = vectorizer.get_feature_names_out()
            nonzero = tfidf_dense > 0
            if np.any(nonzero):
                indices = np.argsort(tfidf_dense[nonzero])[::-1][:15]
                words = feature_names[nonzero][indices]
                scores = tfidf_dense[nonzero][indices]

                fig_words, ax_words = plt.subplots(figsize=(8, 4))
                ax_words.barh(range(len(words)), scores, color='#4A90D9')
                ax_words.set_yticks(range(len(words)))
                ax_words.set_yticklabels(words)
                ax_words.set_xlabel('TF-IDF Score')
                ax_words.invert_yaxis()
                st.pyplot(fig_words)

# ═══════════════════════════════════════════════════════════════
# Tab 4: Top-Wörter
# ═══════════════════════════════════════════════════════════════
with tab4:
    st.header("📝 Top Spam- & Ham-Wörter")

    if not st.session_state.models_trained:
        st.warning("⚠️ Bitte zuerst die Modelle trainieren (Tab 2)!")
    else:
        # Logistic Regression Koeffizienten
        results = st.session_state.results
        lr_result = next((r for r in results if 'Logistic' in r['name']), None)

        if lr_result:
            vectorizer = st.session_state.vectorizer
            model = lr_result['model']
            coef = model.coef_[0]
            feature_names = vectorizer.get_feature_names_out()
            n_tfidf = len(feature_names)

            # Top Spam-Wörter (höchste positive Koeffizienten)
            top_spam_idx = np.argsort(coef[:n_tfidf])[-20:][::-1]

            st.subheader("🔴 Top-20 Spam-Wörter")
            fig_spam, ax_spam = plt.subplots(figsize=(8, 6))
            spam_words = [feature_names[i] for i in top_spam_idx]
            spam_scores = [coef[i] for i in top_spam_idx]
            ax_spam.barh(range(len(spam_words)), spam_scores, color='red', alpha=0.7)
            ax_spam.set_yticks(range(len(spam_words)))
            ax_spam.set_yticklabels(spam_words)
            ax_spam.set_xlabel('Koeffizient (Logistic Regression)')
            ax_spam.set_title('Top Spam-Wörter', fontweight='bold')
            ax_spam.invert_yaxis()
            st.pyplot(fig_spam)

            # Top Ham-Wörter (niedrigste/negativste Koeffizienten)
            top_ham_idx = np.argsort(coef[:n_tfidf])[:20]

            st.subheader("🟢 Top-20 Ham-Wörter")
            fig_ham, ax_ham = plt.subplots(figsize=(8, 6))
            ham_words = [feature_names[i] for i in top_ham_idx]
            ham_scores = [coef[i] for i in top_ham_idx]
            ax_ham.barh(range(len(ham_words)), ham_scores, color='green', alpha=0.7)
            ax_ham.set_yticks(range(len(ham_words)))
            ax_ham.set_yticklabels(ham_words)
            ax_ham.set_xlabel('Koeffizient (Logistic Regression)')
            ax_ham.set_title('Top Ham-Wörter', fontweight='bold')
            ax_ham.invert_yaxis()
            st.pyplot(fig_ham)

            st.info("""
            **Interpretation:**
            - **Positive Koeffizienten** → Wort erhöht Spam-Wahrscheinlichkeit
            - **Negative Koeffizienten** → Wort spricht für Ham
            - Je größer der Betrag, desto stärker der Einfluss
            """)

st.sidebar.markdown("""
### 📚 Über diese App

Interaktive **Spam-Klassifikation** mit TF-IDF und
verschiedenen ML-Modellen.

**Funktionen:**
- 📊 SMS Spam Collection erkunden
- 🔬 TF-IDF verstehen & visualisieren
- 🤖 Naive Bayes vs. Logistic Regression
- 🔮 Eigene Texte live klassifizieren
- 📝 Wichtigste Spam/Ham-Wörter

**Code:** `spam_classifier.py` im Repo
""")
