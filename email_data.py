"""
E-Mail-Daten: SpamAssassin Public Corpus
========================================
Lädt und parst echte E-Mails (vollständige RFC-822-Nachrichten mit Headern,
MIME-Struktur und HTML) aus dem öffentlichen SpamAssassin-Korpus.

Quelle: https://spamassassin.apache.org/old/publiccorpus/
Lizenz: frei zur Forschung/Ausbildung, Empfängeradressen sind anonymisiert.

Zusammensetzung (3.754 Mails, ~27 % Spam):
    easy_ham  (2.501)  normale Mails, klar als Ham erkennbar
    hard_ham    (251)  Ham, das spam-ähnlich aussieht (Newsletter, HTML, Werbung)
    spam      (1.002)  echter Spam aus zwei disjunkten Sammlungen

Verwendung:
    from email_data import load_email_data
    df = load_email_data()          # DataFrame: label, subject, body, ...

⚠️  Leakage-Warnung
Rohe E-Mail-Header dieses Korpus sind *verräterisch*: Ham stammt größtenteils
aus Mailinglisten (List-Id, X-Mailman-*), Spam trägt teils schon X-Spam-Header
der Sammler, und die Received-Ketten unterscheiden sich systematisch. Ein Modell
auf allen Headern erreicht ~100 % — lernt aber die Sammelmethode, nicht Spam.
Deshalb extrahiert dieser Parser nur Felder, die in jeder echten Mailbox
verfügbar und nicht sammlungsspezifisch sind (Subject, Body, MIME-Struktur,
Empfängeranzahl) und ignoriert Received / Delivered-To / X-Spam-* / List-*.
"""

from __future__ import annotations

import email
import email.header
import email.utils
import os
import re
import shutil
import tarfile
from email import policy
from html.parser import HTMLParser
from typing import ClassVar
from urllib.request import urlopen

import pandas as pd

BASE_URL = "https://spamassassin.apache.org/old/publiccorpus"

# (Archivname, Zielordner, Label)  — label: 1 = Spam, 0 = Ham
ARCHIVES = [
    ("20030228_easy_ham.tar.bz2", "easy_ham", 0),
    ("20030228_hard_ham.tar.bz2", "hard_ham", 0),
    ("20030228_spam.tar.bz2", "spam_1", 1),
    ("20021010_spam.tar.bz2", "spam_2", 1),
]

# Header, die in diesem Korpus die Klasse verraten würden (siehe Modul-Docstring)
LEAKY_HEADERS = (
    "received", "delivered-to", "return-path", "x-spam", "list-", "x-mailman",
    "x-beenthere", "sender", "errors-to", "precedence", "x-original-to",
)

MAX_BODY_CHARS = 20_000  # Bodys kappen: begrenzt Laufzeit, kostet keine Qualität


# ═══════════════════════════════════════════════════════════════
# Download
# ═══════════════════════════════════════════════════════════════

def _cache_dir() -> str:
    return os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        ".data_cache", "spamassassin")


def download_corpus(cache_dir: str | None = None, quiet: bool = False) -> str:
    """
    Lädt die Korpus-Archive (falls nicht im Cache) und entpackt sie.

    Jedes Archiv landet in einem eigenen Unterordner, weil beide Spam-Archive
    intern denselben Ordnernamen ``spam/`` benutzen.
    """
    cache_dir = cache_dir or _cache_dir()
    os.makedirs(cache_dir, exist_ok=True)

    for archive, target, _label in ARCHIVES:
        archive_path = os.path.join(cache_dir, archive)
        target_path = os.path.join(cache_dir, target)

        if os.path.isdir(target_path):
            continue

        if not os.path.exists(archive_path):
            if not quiet:
                print(f"  ↓ {archive}")
            url = f"{BASE_URL}/{archive}"
            tmp_path = archive_path + ".part"
            with urlopen(url, timeout=120) as response, open(tmp_path, "wb") as f:
                shutil.copyfileobj(response, f)
            os.replace(tmp_path, archive_path)

        tmp_dir = target_path + ".part"
        shutil.rmtree(tmp_dir, ignore_errors=True)
        with tarfile.open(archive_path, "r:bz2") as tf:
            tf.extractall(tmp_dir)
        os.replace(tmp_dir, target_path)

    return cache_dir


# ═══════════════════════════════════════════════════════════════
# HTML → Text
# ═══════════════════════════════════════════════════════════════

class _HTMLTextExtractor(HTMLParser):
    """Zieht sichtbaren Text aus HTML; verwirft <script>/<style>."""

    SKIP_TAGS: ClassVar[set[str]] = {"script", "style", "head"}

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self._parts: list[str] = []
        self._skip_depth = 0

    def handle_starttag(self, tag, attrs):
        if tag in self.SKIP_TAGS:
            self._skip_depth += 1

    def handle_endtag(self, tag):
        if tag in self.SKIP_TAGS and self._skip_depth:
            self._skip_depth -= 1

    def handle_data(self, data):
        if not self._skip_depth:
            self._parts.append(data)

    @property
    def text(self) -> str:
        return " ".join(self._parts)


def html_to_text(html: str) -> str:
    """HTML → Klartext. Fällt bei kaputtem Markup auf Regex-Stripping zurück."""
    parser = _HTMLTextExtractor()
    try:
        parser.feed(html)
        parser.close()
        text = parser.text
    except Exception:  # noqa: BLE001
        text = re.sub(r"<[^>]+>", " ", html)
    return re.sub(r"\s+", " ", text).strip()


# ═══════════════════════════════════════════════════════════════
# E-Mail-Parsing
# ═══════════════════════════════════════════════════════════════

def _decode_header(msg: email.message.Message, name: str) -> str:
    """Dekodiert einen Header (RFC-2047, z. B. =?iso-8859-1?q?...?=)."""
    raw = msg.get(name)
    if not raw:
        return ""
    try:
        parts = email.header.decode_header(str(raw))
        out = []
        for value, charset in parts:
            if isinstance(value, bytes):
                out.append(value.decode(charset or "latin-1", errors="replace"))
            else:
                out.append(str(value))
        text = " ".join(out)
    except Exception:  # noqa: BLE001
        text = str(raw)
    return re.sub(r"\s+", " ", text).strip()


def _part_text(part: email.message.Message) -> str:
    """Dekodiert den Payload eines MIME-Parts zu Text."""
    try:
        payload = part.get_payload(decode=True)
    except Exception:  # noqa: BLE001
        payload = None
    if payload is None:
        payload = part.get_payload()
        return payload if isinstance(payload, str) else ""
    charset = part.get_content_charset() or "latin-1"
    try:
        return payload.decode(charset, errors="replace")
    except (LookupError, UnicodeDecodeError):
        return payload.decode("latin-1", errors="replace")


def parse_email(raw: bytes) -> dict:
    """
    Parst eine rohe E-Mail zu einem flachen Record.

    Rückgabe:
        subject          Betreffzeile (dekodiert)
        body             Klartext-Body (HTML-Teile werden gestrippt)
        sender           From-Header (dekodiert)
        is_html          1, wenn die Mail HTML enthält
        is_multipart     1 bei MIME-multipart
        num_attachments  Anzahl Anhänge (Content-Disposition: attachment)
        num_recipients   Empfänger in To + Cc
        num_headers      Anzahl nicht-verräterischer Header
        has_reply_to     1, wenn Reply-To gesetzt ist
    """
    msg = email.message_from_bytes(raw, policy=policy.compat32)

    text_parts: list[str] = []
    html_parts: list[str] = []
    num_attachments = 0

    for part in msg.walk():
        if part.get_content_maintype() == "multipart":
            continue
        disposition = str(part.get("Content-Disposition") or "")
        if "attachment" in disposition.lower():
            num_attachments += 1
            continue
        ctype = part.get_content_type()
        if ctype == "text/plain":
            text_parts.append(_part_text(part))
        elif ctype == "text/html":
            html_parts.append(_part_text(part))

    if text_parts:
        body = "\n".join(text_parts)
    elif html_parts:
        body = html_to_text("\n".join(html_parts))
    else:
        # Reine Nicht-Text-Mail (z. B. nur Bild): Body bleibt leer
        body = ""

    if html_parts and text_parts:
        # Multipart/alternative: HTML-Text zusätzlich anhängen, er trägt Signal
        body = body + "\n" + html_to_text("\n".join(html_parts))

    # str() nötig: bei kaputten Headern liefert compat32 ein Header-Objekt
    recipients = ", ".join(
        str(value) for value in (msg.get("To"), msg.get("Cc")) if value
    )
    num_recipients = len(email.utils.getaddresses([recipients])) if recipients else 0

    num_headers = sum(
        1 for key in msg
        if not key.lower().startswith(LEAKY_HEADERS)
    )

    return {
        "subject": _decode_header(msg, "Subject"),
        "body": body[:MAX_BODY_CHARS].strip(),
        "sender": _decode_header(msg, "From"),
        "is_html": int(bool(html_parts)),
        "is_multipart": int(msg.is_multipart()),
        "num_attachments": num_attachments,
        "num_recipients": num_recipients,
        "num_headers": num_headers,
        "has_reply_to": int(bool(msg.get("Reply-To"))),
    }


# ═══════════════════════════════════════════════════════════════
# Laden
# ═══════════════════════════════════════════════════════════════

def load_email_data(cache_dir: str | None = None,
                    quiet: bool = False) -> pd.DataFrame:
    """
    Lädt den SpamAssassin-Korpus als DataFrame.

    Spalten: label (1=Spam), source (easy_ham/hard_ham/spam_1/spam_2),
             subject, body, sender + Struktur-Metafeatures aus parse_email().
    """
    cache_dir = download_corpus(cache_dir, quiet=quiet)

    records = []
    for _archive, folder, label in ARCHIVES:
        root = os.path.join(cache_dir, folder)
        for dirpath, _dirnames, filenames in os.walk(root):
            for filename in sorted(filenames):
                if filename == "cmds":  # Hilfsskript des Korpus, keine Mail
                    continue
                path = os.path.join(dirpath, filename)
                try:
                    with open(path, "rb") as f:
                        raw = f.read()
                    record = parse_email(raw)
                except Exception as exc:  # noqa: BLE001 — eine kaputte Mail stoppt nicht alles
                    if not quiet:
                        print(f"  ⚠️  {filename} übersprungen: {exc}")
                    continue
                record["label"] = label
                record["source"] = folder
                record["message_id"] = filename
                records.append(record)

    df = pd.DataFrame.from_records(records)
    if df.empty:
        raise RuntimeError("Keine E-Mails geladen — ist der Cache leer?")

    # Exakte Duplikate entfernen (die Spam-Sammlungen enthalten Wiederholungen)
    before = len(df)
    df = df.drop_duplicates(subset=["subject", "body"]).reset_index(drop=True)
    if not quiet and len(df) < before:
        print(f"  {before - len(df)} exakte Duplikate entfernt")

    return df


if __name__ == "__main__":
    frame = load_email_data()
    print(f"\n{len(frame):,} E-Mails geladen")
    print(frame.groupby(["source", "label"]).size())
    print("\nBeispiel-Spam:")
    print(frame[frame.label == 1].iloc[0][["subject", "sender", "is_html"]])
    print(frame[frame.label == 1].iloc[0]["body"][:300])
