"""
Tests für spam_classifier.py
"""
import numpy as np
import pandas as pd
import pytest
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB

from spam_classifier import (
    create_features,
    evaluate_model,
    load_spam_data,
)


class TestLoadSpamData:
    """Tests für load_spam_data()."""

    def test_returns_dataframe(self):
        """Gibt einen DataFrame zurück."""
        df = load_spam_data()
        assert isinstance(df, pd.DataFrame)

    def test_has_required_columns(self):
        """DataFrame hat die Spalten 'label' und 'message'."""
        df = load_spam_data()
        assert "label" in df.columns
        assert "message" in df.columns

    def test_label_is_binary(self):
        """label ist 0 oder 1."""
        df = load_spam_data()
        assert set(df["label"].unique()).issubset({0, 1})

    def test_has_both_classes(self):
        """Enthält sowohl Spam (1) als auch Ham (0)."""
        df = load_spam_data()
        assert df["label"].sum() > 0
        assert (df["label"] == 0).sum() > 0

    def test_no_empty_messages(self):
        """Keine leeren Nachrichten."""
        df = load_spam_data()
        assert (df["message"].str.len() > 0).all()

    def test_reasonable_size(self):
        """Mindestens 1000 Nachrichten."""
        df = load_spam_data()
        assert len(df) >= 1000


class TestCreateFeatures:
    """Tests für create_features()."""

    @pytest.fixture
    def sample_df(self):
        """Erzeugt einen kleinen Test-DataFrame."""
        return pd.DataFrame({
            "label": [0, 1, 0, 1],
            "message": [
                "Hey, how are you?",
                "WIN A FREE IPHONE NOW!!! Click http://spam.com",
                "See you at 8pm",
                "URGENT! Call 0800-123456 to claim your prize!",
            ],
        })

    def test_returns_sparse_matrix(self, sample_df):
        """Gibt eine sparse Matrix zurück."""
        X, _vec = create_features(sample_df)
        from scipy.sparse import issparse
        assert issparse(X)

    def test_returns_vectorizer(self, sample_df):
        """Gibt einen TfidfVectorizer zurück."""
        _X, vec = create_features(sample_df)
        assert isinstance(vec, TfidfVectorizer)

    def test_feature_count_matches(self, sample_df):
        """Feature-Dimension = TF-IDF (max 5000) + 7 Metafeatures."""
        X, vec = create_features(sample_df)
        n_tfidf = len(vec.get_feature_names_out())
        assert X.shape[1] == n_tfidf + 7

    def test_row_count_matches_input(self, sample_df):
        """Anzahl Zeilen entspricht der Eingabe."""
        X, _vec = create_features(sample_df)
        assert X.shape[0] == len(sample_df)


class TestEvaluateModel:
    """Tests für evaluate_model()."""

    @pytest.fixture
    def dummy_data(self):
        """Erzeugt synthetische Train/Test-Daten (nicht-negativ für MultinomialNB)."""
        np.random.seed(42)
        X_train = csr_matrix(np.abs(np.random.randn(100, 20)))
        X_test = csr_matrix(np.abs(np.random.randn(30, 20)))
        y_train = np.random.randint(0, 2, 100)
        y_test = np.random.randint(0, 2, 30)
        return X_train, X_test, y_train, y_test

    def test_returns_dict(self, dummy_data):
        """Gibt ein Dictionary zurück."""
        X_train, X_test, y_train, y_test = dummy_data
        result = evaluate_model(
            "Test", MultinomialNB(), X_train, X_test, y_train, y_test
        )
        assert isinstance(result, dict)

    def test_has_required_keys(self, dummy_data):
        """Enthält alle erforderlichen Schlüssel."""
        X_train, X_test, y_train, y_test = dummy_data
        result = evaluate_model(
            "Test", MultinomialNB(), X_train, X_test, y_train, y_test
        )
        for key in ["name", "accuracy", "precision", "recall", "f1", "predictions"]:
            assert key in result

    def test_name_matches(self, dummy_data):
        """Name wird korrekt übernommen."""
        X_train, X_test, y_train, y_test = dummy_data
        result = evaluate_model(
            "MeinModell", MultinomialNB(), X_train, X_test, y_train, y_test
        )
        assert result["name"] == "MeinModell"

    def test_predictions_have_correct_length(self, dummy_data):
        """predictions hat die Länge von y_test."""
        X_train, X_test, y_train, y_test = dummy_data
        result = evaluate_model(
            "Test", MultinomialNB(), X_train, X_test, y_train, y_test
        )
        assert len(result["predictions"]) == len(y_test)

    def test_metrics_in_range(self, dummy_data):
        """Alle Metriken liegen zwischen 0 und 1."""
        X_train, X_test, y_train, y_test = dummy_data
        result = evaluate_model(
            "Test", MultinomialNB(), X_train, X_test, y_train, y_test
        )
        for metric in ["accuracy", "precision", "recall", "f1"]:
            assert 0.0 <= result[metric] <= 1.0

    def test_works_with_logistic_regression(self, dummy_data):
        """Funktioniert auch mit LogisticRegression."""
        X_train, X_test, y_train, y_test = dummy_data
        result = evaluate_model(
            "LR", LogisticRegression(max_iter=1000),
            X_train, X_test, y_train, y_test
        )
        assert result["name"] == "LR"
        assert len(result["predictions"]) == len(y_test)


class TestMainIntegration:
    """Integrationstest: main() läuft ohne Fehler durch."""

    def test_main_runs_without_error(self):
        """main() wirft keine Exception."""
        from spam_classifier import main
        # main() läuft durch (Daten sind bereits gecached)
        main()  # Kein AssertionError/Exception = bestanden
