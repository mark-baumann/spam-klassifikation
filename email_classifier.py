"""
E-Mail-Klassifikation — Supervised Learning mit scikit-learn
============================================================
Binäre Klassifikation echter E-Mails (SpamAssassin Public Corpus, siehe
``email_data.py``) in Spam (1) und Ham (0).

Warum scikit-learn und nicht PyTorch?
-------------------------------------
Bei ~3.200 Trainingsbeispielen und hochdimensionalen, spärlichen TF-IDF-Features
sind lineare Modelle die richtige Wahl:

* **Datenmenge**: Ein neuronales Netz mit eigenen Embeddings braucht
  Zehntausende Beispiele, um TF-IDF + LinearSVC zu schlagen. Bei 3k Mails
  overfittet es zuverlässig.
* **Datenstruktur**: Spam-Signal steckt in einzelnen Tokens ("viagra", "click
  here", "$$$"). Genau dafür sind spärliche lineare Modelle gebaut — sie nutzen
  50.000 Features ohne Dichte-Kompression.
* **Kosten/Nutzen**: Training dauert Sekunden auf der CPU, keine GPU, kein
  Batching, kein Early Stopping. PyTorch würde hier Komplexität ohne
  Metrik-Gewinn hinzufügen.
* **Interpretierbarkeit**: Koeffizienten sind direkt lesbar (siehe Top-Features
  in der Ausgabe) — bei einem Spamfilter, der echte Mails wegwerfen kann, ist
  das kein Luxus.

PyTorch wäre die richtige Wahl bei sehr viel mehr Daten oder wenn ein
vortrainierter Transformer (BERT & Co.) feingetunt werden soll — dann liegt der
Gewinn aber im Pretraining, nicht in der Architektur.

Verwendung:
    python email_classifier.py                       # trainieren + evaluieren
    python email_classifier.py --quick               # ohne Cross-Validation
    python email_classifier.py --predict mail.eml    # rohe .eml klassifizieren
    python email_classifier.py --text "Betreff ..."  # freien Text klassifizieren
"""

from __future__ import annotations

import argparse
import os
import re
import time

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import (
    StratifiedKFold,
    cross_val_predict,
    cross_validate,
    train_test_split,
)
from sklearn.naive_bayes import ComplementNB, MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import LinearSVC

from email_data import load_email_data, parse_email

try:
    from wandb_utils import WandBTracker
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "models", "email_spam_model.joblib")

RANDOM_STATE = 42

# Ziel-Precision für den Betriebspunkt: ein Spamfilter darf keine echte Mail
# verlieren, deshalb wird die Schwelle auf hohe Precision statt auf max. F1
# kalibriert (siehe tune_threshold).
TARGET_PRECISION = 0.99

URL_RE = re.compile(r"https?://|www\.", re.IGNORECASE)
IP_URL_RE = re.compile(r"https?://\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}")
REPLY_RE = re.compile(r"^\s*(re|aw|fwd?|wg)\s*:", re.IGNORECASE)

# Struktur-Features, die email_data.parse_email() direkt liefert
STRUCT_COLS = [
    "is_html", "is_multipart", "num_attachments", "num_recipients",
    "num_headers", "has_reply_to",
]

# Abgeleitete Features aus Betreff und Body
DERIVED_COLS = [
    "body_len", "body_num_words", "body_avg_word_len", "body_caps_ratio",
    "body_digit_ratio", "body_num_exclam", "body_num_currency",
    "body_num_urls", "body_num_ip_urls", "body_num_shouted_words",
    "body_nonascii_ratio", "subject_len", "subject_num_words",
    "subject_caps_ratio", "subject_num_exclam", "subject_is_reply",
]

META_COLS = STRUCT_COLS + DERIVED_COLS


# ═══════════════════════════════════════════════════════════════
# Feature Engineering
# ═══════════════════════════════════════════════════════════════

def _ratio(counts: pd.Series, totals: pd.Series) -> pd.Series:
    """Elementweises Verhältnis mit 0 statt Division durch 0."""
    return (counts / totals.replace(0, np.nan)).fillna(0.0)


def add_meta_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ergänzt handgebaute Metafeatures (Kopie, das Original bleibt unberührt).

    Diese Features fangen Spam-Merkmale, die TF-IDF nicht sieht: SCHREIEN in
    Großbuchstaben, Link-Dichte, Zahlen-/Sonderzeichen-Anteil, HTML-Struktur.
    """
    df = df.copy()
    subject = df["subject"].fillna("").astype(str)
    body = df["body"].fillna("").astype(str)

    df["body_len"] = body.str.len()
    df["body_num_words"] = body.str.split().str.len().fillna(0)
    df["body_avg_word_len"] = _ratio(df["body_len"], df["body_num_words"])
    df["body_caps_ratio"] = _ratio(body.str.count(r"[A-Z]"), df["body_len"])
    df["body_digit_ratio"] = _ratio(body.str.count(r"\d"), df["body_len"])
    df["body_num_exclam"] = body.str.count(r"!")
    df["body_num_currency"] = body.str.count(r"[$€£]")
    df["body_num_urls"] = body.str.count(URL_RE)
    df["body_num_ip_urls"] = body.str.count(IP_URL_RE)
    df["body_num_shouted_words"] = body.str.count(r"\b[A-Z]{4,}\b")
    df["body_nonascii_ratio"] = _ratio(body.str.count(r"[^\x00-\x7F]"),
                                       df["body_len"])

    df["subject_len"] = subject.str.len()
    df["subject_num_words"] = subject.str.split().str.len().fillna(0)
    df["subject_caps_ratio"] = _ratio(subject.str.count(r"[A-Z]"),
                                      df["subject_len"])
    df["subject_num_exclam"] = subject.str.count(r"!")
    df["subject_is_reply"] = subject.str.match(REPLY_RE).astype(int)

    # Fehlende Struktur-Features (z. B. bei --text) mit 0 auffüllen
    for col in STRUCT_COLS:
        if col not in df.columns:
            df[col] = 0

    return df


def build_feature_union(use_char_ngrams: bool = True) -> ColumnTransformer:
    """
    Baut die Feature-Extraktion: drei Textblöcke + skalierte Metafeatures.

    * **subject (Wörter)**: eigener Vektorisierer, weil der Betreff kurz und
      extrem aussagekräftig ist — er darf nicht im Body untergehen.
    * **body (Wörter, 1–2-Gramme)**: das Hauptsignal.
    * **body (Zeichen-3–5-Gramme)**: fängt Verschleierung wie ``V1AGRA`` oder
      ``F R E E``, die auf Wortebene unsichtbar bleibt.
    * **Metafeatures**: auf [0, 1] skaliert, damit sie neben TF-IDF-Werten
      bestehen und Naive Bayes nicht-negative Eingaben bekommt.
    """
    transformers = [
        ("subject_word", TfidfVectorizer(
            sublinear_tf=True, ngram_range=(1, 2), min_df=2,
            max_features=20_000, strip_accents="unicode", lowercase=True,
        ), "subject"),
        ("body_word", TfidfVectorizer(
            sublinear_tf=True, ngram_range=(1, 2), min_df=3,
            max_features=60_000, strip_accents="unicode", lowercase=True,
            stop_words="english",
        ), "body"),
    ]
    if use_char_ngrams:
        transformers.append(
            ("body_char", TfidfVectorizer(
                analyzer="char_wb", ngram_range=(3, 5), min_df=5,
                max_features=30_000, sublinear_tf=True, lowercase=True,
            ), "body")
        )
    transformers.append(("meta", MinMaxScaler(), META_COLS))

    return ColumnTransformer(transformers, sparse_threshold=1.0)


def build_pipeline(model, use_char_ngrams: bool = True) -> Pipeline:
    """Feature-Extraktion + Klassifikator als ein sklearn-Estimator."""
    return Pipeline([
        ("features", build_feature_union(use_char_ngrams)),
        ("clf", model),
    ])


def candidate_models() -> list[tuple[str, object]]:
    """Die Modellkandidaten des Vergleichs."""
    return [
        ("MultinomialNB", MultinomialNB(alpha=0.1)),
        ("ComplementNB", ComplementNB(alpha=0.3)),
        ("LogisticRegression", LogisticRegression(
            C=10.0, max_iter=2000, class_weight="balanced",
            random_state=RANDOM_STATE)),
        ("LinearSVC", LinearSVC(
            C=1.0, class_weight="balanced", random_state=RANDOM_STATE)),
    ]


# ═══════════════════════════════════════════════════════════════
# Scoring & Schwellenwert
# ═══════════════════════════════════════════════════════════════

def spam_scores(estimator, X) -> np.ndarray:
    """
    Kontinuierlicher Spam-Score.

    ``predict_proba`` wo vorhanden (NB, LogReg), sonst ``decision_function``
    (LinearSVC). Die absolute Skala unterscheidet sich, die Rangfolge — und
    damit ROC-AUC und Schwellenwahl — funktioniert in beiden Fällen.
    """
    if hasattr(estimator, "predict_proba"):
        return estimator.predict_proba(X)[:, 1]
    return estimator.decision_function(X)


def tune_threshold(y_true, scores, target_precision: float = TARGET_PRECISION
                   ) -> tuple[float, float, float]:
    """
    Sucht die Schwelle mit der besten Recall bei Precision ≥ target.

    Wichtig: Das passiert auf Out-of-Fold-Vorhersagen der *Trainingsdaten*.
    Auf dem Testset kalibrieren wäre Leakage — der Testwert wäre geschönt.

    Rückgabe: (threshold, precision, recall) am gewählten Punkt.
    """
    precisions, recalls, thresholds = precision_recall_curve(y_true, scores)
    # precision_recall_curve liefert einen Wert mehr als thresholds
    precisions, recalls = precisions[:-1], recalls[:-1]

    ok = precisions >= target_precision
    if not ok.any():  # Ziel unerreichbar → auf max. F1 zurückfallen
        f1s = 2 * precisions * recalls / np.clip(precisions + recalls, 1e-12, None)
        best = int(np.argmax(f1s))
    else:
        candidates = np.flatnonzero(ok)
        best = int(candidates[np.argmax(recalls[candidates])])

    return float(thresholds[best]), float(precisions[best]), float(recalls[best])


def metrics_at(y_true, y_pred, scores=None) -> dict:
    """Standard-Metriken für eine binäre Vorhersage."""
    out = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }
    if scores is not None:
        out["roc_auc"] = roc_auc_score(y_true, scores)
        out["avg_precision"] = average_precision_score(y_true, scores)
    return out


# ═══════════════════════════════════════════════════════════════
# Interpretierbarkeit
# ═══════════════════════════════════════════════════════════════

def top_features(pipeline: Pipeline, n: int = 15
                 ) -> tuple[list[tuple[str, float]], list[tuple[str, float]]]:
    """
    Die stärksten Spam- und Ham-Features eines linearen Modells.

    Rückgabe: (top_spam, top_ham) als Listen von (Featurename, Gewicht).
    """
    clf = pipeline.named_steps["clf"]
    if not hasattr(clf, "coef_"):
        return [], []

    names = pipeline.named_steps["features"].get_feature_names_out()
    coef = np.asarray(clf.coef_).ravel()
    order = np.argsort(coef)

    top_ham = [(str(names[i]), float(coef[i])) for i in order[:n]]
    top_spam = [(str(names[i]), float(coef[i])) for i in order[-n:][::-1]]
    return top_spam, top_ham


def error_analysis(df_test: pd.DataFrame, y_true, y_pred, scores,
                   n: int = 3) -> None:
    """Zeigt die gravierendsten Fehlklassifikationen mit Betreff."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    scores = np.asarray(scores)

    for name, mask, order_desc in [
        ("False Positives (Ham → Spam)", (y_true == 0) & (y_pred == 1), True),
        ("False Negatives (Spam → Ham)", (y_true == 1) & (y_pred == 0), False),
    ]:
        idx = np.flatnonzero(mask)
        print(f"\n   {name}: {len(idx)}")
        if len(idx) == 0:
            continue
        # Nach Score sortieren: die selbstbewusstesten Fehler zuerst
        idx = idx[np.argsort(-scores[idx] if order_desc else scores[idx])]
        for i in idx[:n]:
            row = df_test.iloc[i]
            subject = (row["subject"] or "(kein Betreff)")[:70]
            print(f"   • [{row['source']}] score={scores[i]:+.3f} \"{subject}\"")


# ═══════════════════════════════════════════════════════════════
# Training
# ═══════════════════════════════════════════════════════════════

def train(quick: bool = False, use_char_ngrams: bool = True,
          use_wandb: bool = False, save: bool = True) -> dict:
    """
    Trainiert, vergleicht und evaluiert die Modelle.

    Ablauf:
      1. Echte E-Mails laden und parsen
      2. 80/20 Train-Test-Split (stratifiziert)
      3. Modellvergleich per 5-facher Cross-Validation auf dem Trainingsteil
      4. Bestes Modell auf ganzem Train fitten
      5. Schwelle auf Out-of-Fold-Scores des Trainingsteils kalibrieren
      6. Einmalige Auswertung auf dem Testset

    Rückgabe: Dict mit Pipeline, Schwelle, Metriken und CV-Ergebnissen.
    """
    print("=" * 68)
    print("  E-Mail-Klassifikation — Supervised Learning (scikit-learn)")
    print("=" * 68)

    tracker = None
    if use_wandb and WANDB_AVAILABLE:
        tracker = WandBTracker(
            project="spam-klassifikation",
            config={"task": "email", "corpus": "SpamAssassin public corpus",
                    "char_ngrams": use_char_ngrams},
            tags=["email", "nlp", "sklearn"],
            group="email-klassifikation",
            notes="E-Mail-Spam-Klassifikation auf echten Mails",
        )

    # ── 1. Daten ─────────────────────────────────────────────────
    print("\n📦 Lade echte E-Mails (SpamAssassin Public Corpus)...")
    df = load_email_data(quiet=True)
    df = add_meta_features(df)
    spam_ratio = df["label"].mean()
    print(f"   {len(df):,} E-Mails | {int(df['label'].sum()):,} Spam / "
          f"{int((1 - df['label']).sum()):,} Ham ({spam_ratio:.1%} Spam)")
    print("   Quellen: " + ", ".join(
        f"{src}={cnt}" for src, cnt in df["source"].value_counts().items()))

    # ── 2. Split ─────────────────────────────────────────────────
    X_cols = ["subject", "body"] + META_COLS
    df_train, df_test = train_test_split(
        df, test_size=0.2, random_state=RANDOM_STATE, stratify=df["label"])
    X_train, y_train = df_train[X_cols], df_train["label"].values
    X_test, y_test = df_test[X_cols], df_test["label"].values
    print(f"   Train: {len(X_train):,} | Test: {len(X_test):,}")

    models = candidate_models()
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    # ── 3. Modellvergleich (Cross-Validation) ────────────────────
    cv_results = []
    if quick:
        best_name, best_model = models[-1]  # LinearSVC als Standardwahl
        print(f"\n⚡ Quick-Modus: nur {best_name}, keine Cross-Validation")
    else:
        print(f"\n🤖 Modellvergleich ({cv.get_n_splits()}-fache "
              f"Cross-Validation auf dem Trainingsteil)...")
        print(f"   {'Modell':<20s} {'F1 (CV)':>16s} {'ROC-AUC':>9s} {'Fit':>7s}")
        print("   " + "-" * 56)
        for name, model in models:
            scores = cross_validate(
                build_pipeline(model, use_char_ngrams), X_train, y_train,
                cv=cv, scoring=["f1", "roc_auc"], n_jobs=-1,
            )
            entry = {
                "name": name,
                "f1_mean": float(scores["test_f1"].mean()),
                "f1_std": float(scores["test_f1"].std()),
                "roc_auc_mean": float(scores["test_roc_auc"].mean()),
                "fit_time": float(scores["fit_time"].mean()),
            }
            cv_results.append(entry)
            print(f"   {name:<20s} {entry['f1_mean']:.4f} ±{entry['f1_std']:.4f}"
                  f"   {entry['roc_auc_mean']:.4f} {entry['fit_time']:6.1f}s")
            if tracker and tracker.is_active:
                tracker.log_model_result(
                    model_name=name, accuracy=0.0, f1=entry["f1_mean"],
                    train_time=entry["fit_time"],
                    params={"roc_auc_cv": entry["roc_auc_mean"]})

        best_name = max(cv_results, key=lambda r: r["f1_mean"])["name"]
        best_model = dict(models)[best_name]
        print(f"\n🏆 Bestes Modell: {best_name}")

    pipeline = build_pipeline(best_model, use_char_ngrams)

    # ── 4./5. Fit + Schwellenkalibrierung ────────────────────────
    print("\n🔧 Kalibriere Entscheidungsschwelle "
          f"(Ziel-Precision ≥ {TARGET_PRECISION:.0%})...")
    oof_scores = cross_val_predict(
        pipeline, X_train, y_train, cv=cv, n_jobs=-1,
        method="predict_proba" if hasattr(best_model, "predict_proba")
        else "decision_function",
    )
    if oof_scores.ndim == 2:
        oof_scores = oof_scores[:, 1]
    threshold, cal_precision, cal_recall = tune_threshold(y_train, oof_scores)
    print(f"   Schwelle = {threshold:+.4f} "
          f"(Out-of-Fold: Precision {cal_precision:.3f}, Recall {cal_recall:.3f})")

    t0 = time.time()
    pipeline.fit(X_train, y_train)
    fit_time = time.time() - t0
    n_features = pipeline.named_steps["features"].transform(
        X_train.head(1)).shape[1]
    print(f"   {best_name} auf {len(X_train):,} Mails trainiert "
          f"({fit_time:.1f}s, {n_features:,} Features)")

    # ── 6. Auswertung auf dem Testset ────────────────────────────
    test_scores = spam_scores(pipeline, X_test)
    y_pred_default = pipeline.predict(X_test)
    y_pred_tuned = (test_scores >= threshold).astype(int)

    m_default = metrics_at(y_test, y_pred_default, test_scores)
    m_tuned = metrics_at(y_test, y_pred_tuned, test_scores)

    print("\n📊 Ergebnisse auf dem Testset "
          f"({len(y_test):,} nie gesehene E-Mails):")
    print(f"   {'Betriebspunkt':<24s} {'Acc':>7s} {'Prec':>7s} {'Rec':>7s} {'F1':>7s}")
    print("   " + "-" * 56)
    for label, m in [("Standard (0.5 / 0)", m_default),
                     (f"Precision ≥ {TARGET_PRECISION:.0%}", m_tuned)]:
        print(f"   {label:<24s} {m['accuracy']:7.4f} {m['precision']:7.4f} "
              f"{m['recall']:7.4f} {m['f1']:7.4f}")
    print(f"\n   ROC-AUC: {m_default['roc_auc']:.4f} | "
          f"Average Precision: {m_default['avg_precision']:.4f}")

    tn, fp, fn, tp = confusion_matrix(y_test, y_pred_tuned).ravel()
    print(f"\n   Konfusionsmatrix (kalibrierte Schwelle):")
    print(f"                  vorhergesagt Ham   vorhergesagt Spam")
    print(f"   echt Ham   {tn:>14,}      {fp:>14,}")
    print(f"   echt Spam  {fn:>14,}      {tp:>14,}")
    print("\n" + "\n".join("   " + line for line in classification_report(
        y_test, y_pred_tuned, target_names=["Ham", "Spam"], digits=4,
        zero_division=0).splitlines()))

    # ── Fehleranalyse ────────────────────────────────────────────
    print("\n🔍 Fehleranalyse (kalibrierte Schwelle):")
    error_analysis(df_test, y_test, y_pred_tuned, test_scores)

    # ── Top-Features ─────────────────────────────────────────────
    spam_feats, ham_feats = top_features(pipeline)
    if spam_feats:
        print("\n📝 Stärkste Spam-Indikatoren:")
        for name, weight in spam_feats[:10]:
            print(f"   • {name:<40s} {weight:+.3f}")
        print("\n📝 Stärkste Ham-Indikatoren:")
        for name, weight in ham_feats[:10]:
            print(f"   • {name:<40s} {weight:+.3f}")

    # ── Speichern ────────────────────────────────────────────────
    result = {
        "pipeline": pipeline,
        "model_name": best_name,
        "threshold": threshold,
        "metrics_default": m_default,
        "metrics_tuned": m_tuned,
        "cv_results": cv_results,
        "n_features": int(n_features),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
    }
    if save:
        import joblib
        os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
        joblib.dump({k: v for k, v in result.items() if k != "cv_results"},
                    MODEL_PATH, compress=3)
        size_mb = os.path.getsize(MODEL_PATH) / 1e6
        print(f"\n💾 Modell gespeichert: {os.path.relpath(MODEL_PATH)} "
              f"({size_mb:.1f} MB)")

    if tracker and tracker.is_active:
        tracker.log_feature_stats(num_features=int(n_features),
                                  num_samples=len(df),
                                  spam_ratio=float(spam_ratio))
        tracker.log_model_result(
            model_name=f"{best_name}-test", accuracy=m_tuned["accuracy"],
            f1=m_tuned["f1"], precision=m_tuned["precision"],
            recall=m_tuned["recall"], train_time=fit_time)
        if spam_feats:
            tracker.log_top_features(
                best_name, [n for n, _ in spam_feats], [w for _, w in spam_feats])
        tracker.finish()

    print("\n✅ Fertig.")
    return result


# ═══════════════════════════════════════════════════════════════
# Inferenz
# ═══════════════════════════════════════════════════════════════

def load_model(path: str = MODEL_PATH) -> dict:
    """Lädt ein gespeichertes Modell samt kalibrierter Schwelle."""
    import joblib
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Kein Modell unter {path}. Erst 'python email_classifier.py' laufen "
            "lassen.")
    return joblib.load(path)


def classify(records: list[dict], model: dict | None = None) -> pd.DataFrame:
    """
    Klassifiziert Records (aus ``parse_email`` oder mit subject/body von Hand).

    Rückgabe: DataFrame mit Spalten score, is_spam, label.
    """
    model = model or load_model()
    df = add_meta_features(pd.DataFrame(records))
    for col in ["subject", "body"]:
        if col not in df.columns:
            df[col] = ""
    X = df[["subject", "body"] + META_COLS]

    scores = spam_scores(model["pipeline"], X)
    is_spam = scores >= model["threshold"]
    return pd.DataFrame({
        "score": scores,
        "is_spam": is_spam,
        "label": np.where(is_spam, "SPAM", "HAM"),
    })


def predict_file(path: str, model: dict | None = None) -> pd.DataFrame:
    """Klassifiziert eine rohe .eml-Datei."""
    with open(path, "rb") as f:
        record = parse_email(f.read())
    return classify([record], model)


# ═══════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(
        description="E-Mail-Klassifikation (Spam/Ham) mit scikit-learn")
    parser.add_argument("--predict", metavar="EML",
                        help="rohe E-Mail-Datei klassifizieren")
    parser.add_argument("--text", metavar="TEXT",
                        help="freien Text als E-Mail-Body klassifizieren")
    parser.add_argument("--subject", default="", metavar="BETREFF",
                        help="Betreff zu --text")
    parser.add_argument("--quick", action="store_true",
                        help="Training ohne Modellvergleich (nur LinearSVC)")
    parser.add_argument("--no-char-ngrams", action="store_true",
                        help="Zeichen-n-Gramme weglassen (schneller)")
    parser.add_argument("--wandb", action="store_true",
                        help="Metriken zu Weights & Biases loggen")
    parser.add_argument("--no-save", action="store_true",
                        help="Modell nicht auf die Platte schreiben")
    args = parser.parse_args()

    if args.predict or args.text:
        model = load_model()
        if args.predict:
            result = predict_file(args.predict, model)
            source = args.predict
        else:
            result = classify([{"subject": args.subject, "body": args.text}],
                              model)
            source = f"\"{(args.subject or args.text)[:50]}...\""
        row = result.iloc[0]
        print(f"\n{row['label']}  (Score {row['score']:+.4f}, Schwelle "
              f"{model['threshold']:+.4f})")
        print(f"Modell: {model['model_name']} | Eingabe: {source}")
        return

    train(quick=args.quick, use_char_ngrams=not args.no_char_ngrams,
          use_wandb=args.wandb, save=not args.no_save)


if __name__ == "__main__":
    main()
