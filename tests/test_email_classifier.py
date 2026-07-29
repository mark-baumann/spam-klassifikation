"""
Tests für email_data.py und email_classifier.py

Die Tests laufen ohne Netzwerk: alles bis auf die als ``needs_corpus``
markierten Tests arbeitet mit synthetischen E-Mails.
"""
import os
import textwrap

import numpy as np
import pandas as pd
import pytest
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC

from email_classifier import (
    META_COLS,
    add_meta_features,
    build_pipeline,
    classify,
    metrics_at,
    spam_scores,
    top_features,
    tune_threshold,
)
from email_data import _cache_dir, html_to_text, load_email_data, parse_email

# ═══════════════════════════════════════════════════════════════
# Test-Fixtures: synthetische, aber strukturell echte E-Mails
# ═══════════════════════════════════════════════════════════════

PLAIN_MAIL = textwrap.dedent("""\
    From: anna@example.org
    To: mark@example.com
    Subject: Re: Meeting am Donnerstag
    Date: Tue, 3 Jun 2003 10:12:00 +0200
    Content-Type: text/plain; charset="utf-8"

    Hallo Mark,

    passt Donnerstag 14 Uhr fuer dich?

    Gruss, Anna
    """).encode()

HTML_MAIL = textwrap.dedent("""\
    From: winner@promo.example
    To: victim@example.com, other@example.com
    Reply-To: cash@promo.example
    Subject: YOU WON!!!
    Content-Type: text/html; charset="iso-8859-1"

    <html><head><style>b { color: red }</style></head>
    <body><h1>CONGRATULATIONS</h1>
    <p>Click <a href="http://1.2.3.4/claim">HERE</a> to claim $1000 &amp; more!</p>
    <script>alert('x')</script>
    </body></html>
    """).encode()

MULTIPART_MAIL = textwrap.dedent("""\
    From: bob@example.org
    To: alice@example.org
    Subject: Bericht
    MIME-Version: 1.0
    Content-Type: multipart/mixed; boundary="BOUND"

    --BOUND
    Content-Type: text/plain; charset="us-ascii"

    Anhang im Anhang.
    --BOUND
    Content-Type: application/pdf; name="report.pdf"
    Content-Disposition: attachment; filename="report.pdf"
    Content-Transfer-Encoding: base64

    JVBERi0xLjQK
    --BOUND--
    """).encode()

# Kaputte Header (RFC-2047 defekt, kein Subject) — im echten Spam Alltag
BROKEN_MAIL = (
    b"From: =?bogus?X?\xff\xfe?=\r\n"
    b"To: \xff invalid <a@b.c>\r\n"
    b"Content-Type: text/plain\r\n\r\n"
    b"Body mit \xff kaputtem Byte\r\n"
)


def _toy_frame() -> pd.DataFrame:
    """Kleiner, klar trennbarer Datensatz für Pipeline-Tests."""
    spam = [
        ("FREE MONEY NOW", "Click here to win free cash prize viagra now!!!"),
        ("WINNER!!!", "You won a free lottery prize, click http://1.2.3.4 now"),
        ("Cheap pills", "Buy cheap viagra pills online now, free shipping!!!"),
        ("URGENT $$$", "Wire transfer money urgent free offer click here"),
    ]
    ham = [
        ("Re: Meeting", "Passt Donnerstag um 14 Uhr fuer das Review Meeting?"),
        ("Protokoll", "Anbei das Protokoll der Sitzung von gestern. Gruss Anna"),
        ("Re: Build kaputt", "Der Build laeuft wieder, war ein Tippfehler."),
        ("Mittagessen", "Gehen wir um zwoelf zusammen Mittagessen? Bob"),
    ]
    rows = [{"subject": s, "body": b, "label": 1} for s, b in spam]
    rows += [{"subject": s, "body": b, "label": 0} for s, b in ham]
    return add_meta_features(pd.DataFrame(rows))


@pytest.fixture(scope="module")
def toy_data():
    df = _toy_frame()
    return df[["subject", "body"] + META_COLS], df["label"].values


@pytest.fixture(scope="module")
def toy_model(toy_data):
    """Trainierte Pipeline im Modell-Dict-Format von load_model()."""
    X, y = toy_data
    pipeline = build_pipeline(LinearSVC(C=1.0, random_state=0),
                             use_char_ngrams=False)
    pipeline.fit(X, y)
    return {"pipeline": pipeline, "model_name": "LinearSVC", "threshold": 0.0}


# ═══════════════════════════════════════════════════════════════
# html_to_text
# ═══════════════════════════════════════════════════════════════

class TestHtmlToText:

    def test_strips_tags(self):
        assert html_to_text("<p>Hallo <b>Welt</b></p>") == "Hallo Welt"

    def test_drops_script_and_style(self):
        html = "<style>b{color:red}</style><p>Text</p><script>evil()</script>"
        assert html_to_text(html) == "Text"

    def test_unescapes_entities(self):
        assert "&" in html_to_text("<p>Tom &amp; Jerry</p>")

    def test_collapses_whitespace(self):
        assert html_to_text("<p>a\n\n   b</p>") == "a b"

    def test_survives_broken_markup(self):
        # Darf nicht werfen, egal wie kaputt das Markup ist
        assert isinstance(html_to_text("<p><<>unclosed <b>x"), str)

    def test_empty_input(self):
        assert html_to_text("") == ""


# ═══════════════════════════════════════════════════════════════
# parse_email
# ═══════════════════════════════════════════════════════════════

class TestParseEmail:

    def test_plain_mail_fields(self):
        rec = parse_email(PLAIN_MAIL)
        assert rec["subject"] == "Re: Meeting am Donnerstag"
        assert rec["sender"] == "anna@example.org"
        assert "passt Donnerstag" in rec["body"]
        assert rec["is_html"] == 0
        assert rec["num_recipients"] == 1
        assert rec["has_reply_to"] == 0

    def test_html_mail_is_converted_to_text(self):
        rec = parse_email(HTML_MAIL)
        assert rec["is_html"] == 1
        assert "CONGRATULATIONS" in rec["body"]
        assert "<h1>" not in rec["body"]
        assert "alert" not in rec["body"]  # <script> entfernt
        assert rec["num_recipients"] == 2
        assert rec["has_reply_to"] == 1

    def test_multipart_attachment_is_counted_not_read(self):
        rec = parse_email(MULTIPART_MAIL)
        assert rec["is_multipart"] == 1
        assert rec["num_attachments"] == 1
        assert "Anhang im Anhang" in rec["body"]
        assert "JVBERi" not in rec["body"]  # Base64-Payload nicht im Body

    def test_broken_headers_do_not_raise(self):
        rec = parse_email(BROKEN_MAIL)
        assert isinstance(rec["subject"], str)  # fehlt → ""
        assert rec["subject"] == ""
        assert "kaputtem Byte" in rec["body"]

    def test_all_keys_present(self):
        expected = {
            "subject", "body", "sender", "is_html", "is_multipart",
            "num_attachments", "num_recipients", "num_headers", "has_reply_to",
        }
        assert expected <= set(parse_email(PLAIN_MAIL))

    def test_leaky_headers_are_not_counted(self):
        """Received/X-Spam & Co. dürfen num_headers nicht beeinflussen."""
        leaky = (b"Received: from evil by relay\r\n"
                 b"X-Spam-Status: Yes\r\n"
                 b"List-Id: <ham.example.org>\r\n") + PLAIN_MAIL
        assert parse_email(leaky)["num_headers"] == \
            parse_email(PLAIN_MAIL)["num_headers"]

    def test_body_is_truncated(self):
        from email_data import MAX_BODY_CHARS
        huge = (b"Subject: x\r\nContent-Type: text/plain\r\n\r\n"
                + b"spam " * (MAX_BODY_CHARS // 2))
        assert len(parse_email(huge)["body"]) <= MAX_BODY_CHARS


# ═══════════════════════════════════════════════════════════════
# Metafeatures
# ═══════════════════════════════════════════════════════════════

class TestMetaFeatures:

    def test_all_columns_created(self):
        df = add_meta_features(pd.DataFrame([{"subject": "Hi", "body": "Text"}]))
        assert set(META_COLS) <= set(df.columns)

    def test_original_frame_untouched(self):
        original = pd.DataFrame([{"subject": "Hi", "body": "Text"}])
        add_meta_features(original)
        assert list(original.columns) == ["subject", "body"]

    def test_counts_are_correct(self):
        df = add_meta_features(pd.DataFrame([{
            "subject": "WIN NOW!!",
            "body": "FREE money http://1.2.3.4 and www.spam.example $$$ !",
        }]))
        row = df.iloc[0]
        assert row["subject_num_exclam"] == 2
        assert row["body_num_urls"] == 2      # http:// und www.
        assert row["body_num_ip_urls"] == 1
        assert row["body_num_currency"] == 3
        assert row["body_num_shouted_words"] == 1  # "FREE"

    def test_reply_prefixes_detected(self):
        df = add_meta_features(pd.DataFrame([
            {"subject": "Re: Hallo", "body": ""},
            {"subject": "AW: Hallo", "body": ""},
            {"subject": "Fwd: Hallo", "body": ""},
            {"subject": "Neue Mail", "body": ""},
        ]))
        assert list(df["subject_is_reply"]) == [1, 1, 1, 0]

    def test_empty_text_gives_zero_ratios_not_nan(self):
        df = add_meta_features(pd.DataFrame([{"subject": "", "body": ""}]))
        assert df[META_COLS].notna().all().all()
        assert df.iloc[0]["body_caps_ratio"] == 0.0
        assert df.iloc[0]["body_avg_word_len"] == 0.0

    def test_missing_struct_columns_filled_with_zero(self):
        """Bei --text gibt es keine MIME-Struktur — Features müssen 0 sein."""
        df = add_meta_features(pd.DataFrame([{"subject": "x", "body": "y"}]))
        assert df.iloc[0]["is_html"] == 0
        assert df.iloc[0]["num_attachments"] == 0

    def test_ratios_are_bounded(self):
        df = add_meta_features(pd.DataFrame([{"subject": "ABC", "body": "AB!"}]))
        for col in ["body_caps_ratio", "body_digit_ratio", "subject_caps_ratio",
                    "body_nonascii_ratio"]:
            assert 0.0 <= df.iloc[0][col] <= 1.0


# ═══════════════════════════════════════════════════════════════
# Pipeline
# ═══════════════════════════════════════════════════════════════

class TestPipeline:

    def test_fits_and_predicts(self, toy_data):
        X, y = toy_data
        pipeline = build_pipeline(LinearSVC(random_state=0),
                                 use_char_ngrams=False)
        pipeline.fit(X, y)
        pred = pipeline.predict(X)
        assert pred.shape == y.shape
        assert set(np.unique(pred)) <= {0, 1}

    def test_features_are_non_negative_for_naive_bayes(self, toy_data):
        """MultinomialNB verlangt Features ≥ 0 — sonst wirft fit()."""
        X, y = toy_data
        pipeline = build_pipeline(MultinomialNB(alpha=0.1),
                                  use_char_ngrams=False)
        pipeline.fit(X, y)
        matrix = pipeline.named_steps["features"].transform(X)
        assert matrix.min() >= 0

    def test_char_ngrams_add_features(self, toy_data):
        X, _y = toy_data
        # Der Zeichen-Vektorisierer nutzt min_df=5, braucht also mehr Dokumente
        # als der Spielzeug-Datensatz hat.
        X_big = pd.concat([X] * 3, ignore_index=True)
        without = build_pipeline(LinearSVC(), use_char_ngrams=False)
        with_char = build_pipeline(LinearSVC(), use_char_ngrams=True)
        n_without = without.named_steps["features"].fit_transform(X_big).shape[1]
        n_with = with_char.named_steps["features"].fit_transform(X_big).shape[1]
        assert n_with > n_without

    def test_meta_features_are_part_of_the_matrix(self, toy_data):
        X, _y = toy_data
        features = build_pipeline(LinearSVC(),
                                  use_char_ngrams=False).named_steps["features"]
        features.fit(X)
        names = list(features.get_feature_names_out())
        assert any(name.startswith("meta__") for name in names)

    def test_separates_the_toy_classes(self, toy_data):
        """Sanity-Check: das Modell lernt überhaupt etwas."""
        X, y = toy_data
        pipeline = build_pipeline(LinearSVC(random_state=0),
                                 use_char_ngrams=False)
        pipeline.fit(X, y)
        assert (pipeline.predict(X) == y).mean() == 1.0


# ═══════════════════════════════════════════════════════════════
# Scores, Schwelle, Metriken
# ═══════════════════════════════════════════════════════════════

class TestScoresAndThreshold:

    def test_spam_scores_uses_decision_function_for_svc(self, toy_model, toy_data):
        X, _y = toy_data
        scores = spam_scores(toy_model["pipeline"], X)
        assert scores.shape == (len(X),)
        assert scores.dtype.kind == "f"

    def test_spam_scores_uses_probabilities_for_nb(self, toy_data):
        X, y = toy_data
        pipeline = build_pipeline(MultinomialNB(alpha=0.1),
                                 use_char_ngrams=False).fit(X, y)
        scores = spam_scores(pipeline, X)
        assert ((scores >= 0) & (scores <= 1)).all()

    def test_threshold_reaches_target_precision(self):
        y = np.array([0] * 50 + [1] * 50)
        scores = np.concatenate([np.linspace(0, 0.5, 50),
                                 np.linspace(0.5, 1.0, 50)])
        threshold, precision, _recall = tune_threshold(y, scores,
                                                       target_precision=0.95)
        assert precision >= 0.95
        pred = (scores >= threshold).astype(int)
        assert (pred[y == 0].sum()) <= 2  # kaum False Positives

    def test_threshold_maximises_recall_at_target(self):
        y = np.array([0, 0, 0, 1, 1, 1])
        scores = np.array([0.1, 0.2, 0.3, 0.4, 0.8, 0.9])
        _t, precision, recall = tune_threshold(y, scores, target_precision=1.0)
        assert precision == 1.0
        assert recall == 1.0  # perfekt trennbar → volle Recall möglich

    def test_falls_back_to_best_f1_when_target_unreachable(self):
        y = np.array([0, 1, 0, 1])
        scores = np.array([0.5, 0.5, 0.5, 0.5])  # keine Trennung möglich
        threshold, precision, recall = tune_threshold(y, scores,
                                                      target_precision=0.99)
        assert precision < 0.99          # Ziel wurde verfehlt ...
        assert np.isfinite(threshold)    # ... aber es kommt ein Wert zurück
        assert recall > 0

    def test_metrics_at_returns_all_keys(self):
        y = np.array([0, 1, 1, 0])
        pred = np.array([0, 1, 0, 0])
        scores = np.array([0.1, 0.9, 0.4, 0.2])
        m = metrics_at(y, pred, scores)
        assert set(m) == {"accuracy", "precision", "recall", "f1", "roc_auc",
                          "avg_precision"}
        assert m["accuracy"] == 0.75

    def test_metrics_at_without_scores(self):
        m = metrics_at(np.array([0, 1]), np.array([0, 1]))
        assert "roc_auc" not in m
        assert m["f1"] == 1.0


# ═══════════════════════════════════════════════════════════════
# Interpretierbarkeit & Inferenz
# ═══════════════════════════════════════════════════════════════

class TestTopFeatures:

    def test_returns_spam_and_ham_features(self, toy_model):
        spam, ham = top_features(toy_model["pipeline"], n=5)
        assert len(spam) == 5 and len(ham) == 5
        assert spam[0][1] > 0 > ham[0][1]
        assert spam[0][1] >= spam[-1][1]  # absteigend sortiert

    def test_empty_for_models_without_coefficients(self, toy_data):
        from sklearn.ensemble import RandomForestClassifier
        X, y = toy_data
        pipeline = build_pipeline(RandomForestClassifier(n_estimators=5,
                                                        random_state=0),
                                 use_char_ngrams=False).fit(X, y)
        assert top_features(pipeline) == ([], [])


class TestClassify:

    def test_classifies_obvious_spam_and_ham(self, toy_model):
        records = [
            {"subject": "FREE VIAGRA!!!",
             "body": "Click here to win free money now viagra prize"},
            {"subject": "Re: Meeting",
             "body": "Passt Donnerstag um 14 Uhr fuer das Review Meeting?"},
        ]
        result = classify(records, model=toy_model)
        assert list(result["label"]) == ["SPAM", "HAM"]
        assert result["score"].iloc[0] > result["score"].iloc[1]

    def test_works_without_struct_columns(self, toy_model):
        """Nur subject/body → fehlende Features werden ergänzt, kein Fehler."""
        result = classify([{"subject": "Hallo", "body": "Kurzer Text"}],
                          model=toy_model)
        assert len(result) == 1
        assert result["label"].iloc[0] in {"SPAM", "HAM"}

    def test_accepts_parsed_email_records(self, toy_model):
        result = classify([parse_email(PLAIN_MAIL), parse_email(HTML_MAIL)],
                          model=toy_model)
        assert len(result) == 2
        assert set(result.columns) == {"score", "is_spam", "label"}

    def test_respects_threshold(self, toy_model):
        record = [{"subject": "Hallo", "body": "Kurzer Text"}]
        strict = dict(toy_model, threshold=1e9)
        lenient = dict(toy_model, threshold=-1e9)
        assert classify(record, model=strict)["is_spam"].iloc[0] == False  # noqa: E712
        assert classify(record, model=lenient)["is_spam"].iloc[0] == True   # noqa: E712


# ═══════════════════════════════════════════════════════════════
# Echter Korpus (nur wenn der Cache schon gefüllt ist)
# ═══════════════════════════════════════════════════════════════

needs_corpus = pytest.mark.skipif(
    not os.path.isdir(os.path.join(_cache_dir(), "easy_ham")),
    reason="Korpus nicht im Cache — 'python email_data.py' lädt ihn herunter",
)


@pytest.fixture(scope="module")
def corpus():
    return load_email_data(quiet=True)


@needs_corpus
class TestRealCorpus:

    def test_has_expected_shape(self, corpus):
        assert len(corpus) > 3000
        assert {"label", "subject", "body", "source"} <= set(corpus.columns)

    def test_labels_are_binary_and_both_present(self, corpus):
        assert set(corpus["label"].unique()) == {0, 1}
        assert corpus["label"].sum() > 400          # Spam
        assert (1 - corpus["label"]).sum() > 2000   # Ham

    def test_ham_sources_are_labelled_zero(self, corpus):
        ham_sources = corpus[corpus["source"].str.endswith("ham")]
        assert (ham_sources["label"] == 0).all()

    def test_no_duplicate_messages(self, corpus):
        assert not corpus.duplicated(subset=["subject", "body"]).any()

    def test_bodies_are_mostly_non_empty(self, corpus):
        assert (corpus["body"].str.len() > 0).mean() > 0.95

    def test_meta_features_computable_on_real_data(self, corpus):
        df = add_meta_features(corpus)
        assert df[META_COLS].notna().all().all()
        assert np.isfinite(df[META_COLS].to_numpy(dtype=float)).all()
