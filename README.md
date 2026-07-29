# Spam-Klassifikation 📧

**Textklassifikation von Grund auf: TF-IDF, Naive Bayes, Logistic Regression, LinearSVC**

Lernprojekt zur binären Textklassifikation — der Grundlage für Embeddings, RAG und moderne NLP-Systeme.

Zwei Datensätze, zwei Skripte:

| Skript | Daten | Aufgabe |
|---|---|---|
| `spam_classifier.py` | SMS Spam Collection (UCI) | kurze SMS-Texte |
| `email_classifier.py` | **SpamAssassin Public Corpus** | echte E-Mails mit Headern, MIME und HTML |

---

## 📬 E-Mail-Klassifikation (`email_classifier.py`)

Supervised Learning auf **3.227 echten E-Mails** — vollständige RFC-822-Nachrichten
aus dem [SpamAssassin Public Corpus](https://spamassassin.apache.org/old/publiccorpus/),
inklusive Headern, MIME-Multipart-Struktur, HTML-Bodys und Anhängen.

| Quelle | Mails | Label |
|---|---|---|
| `easy_ham` | 2.472 | Ham — normale Mails |
| `hard_ham` | 250 | Ham — sieht spam-ähnlich aus (Newsletter, HTML-Werbung) |
| `spam_1` + `spam_2` | 505 | Spam |

Der Download passiert beim ersten Start automatisch (~5 MB, Cache in `.data_cache/`).
Exakte Duplikate werden entfernt, damit dieselbe Mail nicht in Train *und* Test landet.

### Warum scikit-learn und nicht PyTorch?

Die Frage ist berechtigt — hier ist sklearn objektiv die bessere Wahl:

- **Datenmenge**: 2.581 Trainingsmails. Ein neuronales Netz, das eigene Embeddings
  von null lernt, braucht Zehntausende Beispiele, um TF-IDF + LinearSVC zu schlagen.
  Bei dieser Größe overfittet es zuverlässig.
- **Datenstruktur**: Das Spam-Signal steckt in einzelnen Tokens (`viagra`,
  `click here`, `$$$`). Genau dafür sind spärliche lineare Modelle gebaut — sie
  nutzen 65.000 Features direkt, ohne Kompression in dichte Vektoren.
- **Kosten/Nutzen**: 8 Sekunden Training auf der CPU. Keine GPU, kein Batching,
  kein Early Stopping, kein Learning-Rate-Schedule.
- **Interpretierbarkeit**: Die Koeffizienten sind direkt lesbar (siehe Top-Features
  unten). Bei einem Filter, der echte Mails wegwerfen kann, ist das kein Luxus.

PyTorch wäre richtig bei deutlich mehr Daten oder beim Feintuning eines
vortrainierten Transformers (BERT & Co.) — der Gewinn läge dann aber im
Pretraining, nicht in der Architektur.

### Features

Vier Blöcke, kombiniert in einem `ColumnTransformer`:

| Block | Was |
|---|---|
| `subject_word` | TF-IDF über den Betreff (1–2-Gramme) — kurz, aber extrem aussagekräftig, darf nicht im Body untergehen |
| `body_word` | TF-IDF über den Body (1–2-Gramme, englische Stopwords) — das Hauptsignal |
| `body_char` | TF-IDF über Zeichen-3–5-Gramme — fängt Verschleierung wie `V1AGRA` oder `F R E E` |
| `meta` | 22 handgebaute Features: HTML-Anteil, Link-Dichte, IP-URLs, `GROSSBUCHSTABEN`-Wörter, `$`-Zeichen, Anhänge, Empfängeranzahl, `Re:`-Präfix … |

Die Zeichen-n-Gramme kosten Rechenzeit, aber sie zahlen sich aus: LinearSVC
erreicht mit ihnen 0.9608 F1 in der Cross-Validation, ohne sie 0.9450 (Fit 1.1 s
statt 6.2 s → `--no-char-ngrams`, wenn Geschwindigkeit wichtiger ist).

**Leakage-Vermeidung**: Rohe Header dieses Korpus verraten die Klasse — Ham stammt
größtenteils aus Mailinglisten (`List-Id`, `X-Mailman-*`), Spam trägt teils schon
`X-Spam-*`-Header der Sammler. Ein Modell darauf erreicht ~100 % und lernt die
Sammelmethode statt Spam. `email_data.py` ignoriert deshalb `Received`,
`Delivered-To`, `Return-Path`, `X-Spam-*`, `List-*` & Co. und nutzt nur Felder,
die in jeder echten Mailbox vorhanden sind.

### Ergebnisse

**Modellvergleich** (5-fache Cross-Validation, nur Trainingsdaten):

| Modell | F1 (CV) | ROC-AUC | Fit |
|---|---|---|---|
| MultinomialNB | 0.9322 ± 0.0188 | 0.9949 | 6.8 s |
| ComplementNB | 0.9406 ± 0.0107 | 0.9932 | 5.7 s |
| LogisticRegression | 0.9543 ± 0.0177 | 0.9969 | 6.1 s |
| **LinearSVC** 🏆 | **0.9608 ± 0.0126** | 0.9969 | 6.2 s |

**Testset** (646 nie gesehene E-Mails, LinearSVC, 65.211 Features):

| Betriebspunkt | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|
| Standard-Schwelle | 0.9876 | 0.9604 | 0.9604 | 0.9604 |
| Precision ≥ 99 % | 0.9814 | **1.0000** | 0.8812 | 0.9368 |

ROC-AUC 0.9997 · Average Precision 0.9982

Der zweite Betriebspunkt ist der interessante: Für einen Spamfilter ist ein
False Positive (echte Mail landet im Spam) viel teurer als ein False Negative
(Spam im Postfach). Die Schwelle wird deshalb **nicht** auf maximales F1 gesetzt,
sondern auf hohe Precision — kalibriert auf Out-of-Fold-Vorhersagen der
Trainingsdaten, niemals auf dem Testset (das wäre Leakage). Ergebnis: **0 von 545
echten Mails** wurden falsch als Spam markiert, dafür rutschen 12 von 101 Spams
durch.

**Stärkste Spam-Indikatoren** (LinearSVC-Koeffizienten):

```
meta__is_html                  +0.702    meta__num_recipients        +0.522
meta__body_nonascii_ratio      +0.660    meta__body_num_ip_urls      +0.521
meta__num_attachments          +0.514    meta__subject_num_exclam    +0.477
meta__body_num_shouted_words   +0.443    body_word__remove           +0.435
```

Bemerkenswert: Die handgebauten Metafeatures dominieren die stärksten Gewichte —
HTML-Mails, viele Empfänger, IP-Adressen als Links und Ausrufezeichen im Betreff
sind 2003 wie heute die verlässlichsten Spam-Signale.

### Verwendung

```bash
uv venv && uv pip install -r requirements.txt

# Trainieren, evaluieren, Modell nach models/ speichern (~90 s inkl. Vergleich)
python email_classifier.py

# Schneller: nur LinearSVC, ohne Cross-Validation (~30 s)
python email_classifier.py --quick

# Eine rohe .eml-Datei klassifizieren
python email_classifier.py --predict mail.eml

# Freien Text klassifizieren
python email_classifier.py --subject "You WON \$\$\$" --text "Click here now!!!"
```

```
SPAM  (Score +1.1406, Schwelle +0.5176)
Modell: LinearSVC | Eingabe: "You WON $$$..."
```

Als Bibliothek:

```python
from email_data import parse_email
from email_classifier import classify, load_model

model = load_model()
with open("mail.eml", "rb") as f:
    record = parse_email(f.read())
print(classify([record], model))
#       score  is_spam label
# 0  1.503906     True  SPAM
```

Weitere Flags: `--no-char-ngrams` (schneller), `--wandb` (Metriken zu Weights &
Biases), `--no-save`.

---

## 📱 SMS-Klassifikation (`spam_classifier.py`)

```bash
python spam_classifier.py
```

| Modell | Accuracy | F1-Score |
|---|---|---|
| Naive Bayes | ~97 % | ~0.92 |
| Logistic Regression | ~98 % | ~0.94 |

Zusätzlich: Streamlit-Demo mit `streamlit run app.py`.

---

## 🧪 Tests

```bash
python -m pytest -q     # 68 Tests
```

Die E-Mail-Tests laufen ohne Netzwerk auf synthetischen, aber strukturell echten
Mails (HTML, MIME-Multipart, kaputte RFC-2047-Header). Die Tests gegen den echten
Korpus überspringen sich selbst, solange `.data_cache/` leer ist.

## 🧠 Lernziele

1. **Bag-of-Words**: Wie wird Text zu Zahlen?
2. **TF-IDF**: Warum ist "the" weniger wichtig als "free"?
3. **Naive Bayes**: Warum "naive"? (Unabhängigkeitsannahme!)
4. **Precision vs. Recall**: Warum ist bei Spam die Precision wichtiger — und wie
   kalibriert man eine Schwelle darauf, ohne das Testset zu verbrennen?
5. **Data Leakage**: Warum ein Modell mit 100 % Accuracy meist ein Bug ist
6. **Feature Engineering**: Wann schlagen handgebaute Features das Sprachmodell?
7. **Embeddings**: Der Sprung von TF-IDF zu dichten Vektoren
