import os
from typing import Any
import regex as re
from fastwarc.warc import ArchiveIterator, WarcRecordType
from resiliparse.extract.html2text import extract_plain_text
from resiliparse.parse.encoding import detect_encoding
import fasttext
import nltk
from .common import get_shared_assets_path

DIRECTORY = get_shared_assets_path()
language_identify_model = fasttext.load_model(str(DIRECTORY / "classifiers/lid.176.bin"))
nsfw_identify_model = fasttext.load_model(str(DIRECTORY / "classifiers/dolma_fasttext_nsfw_jigsaw_model.bin"))
toxic_speech_identify_model = fasttext.load_model(str(DIRECTORY / "classifiers/dolma_fasttext_hatespeech_jigsaw_model.bin"))
email_pattern = re.compile(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9-]+(\.[a-zA-Z]+)+")
phone_number_pattern = re.compile(r"(\+?1[\- ])?(\(\d{3}\)|\d{3})[.\- ]?\d{3}[.\- ]?\d{4}")
ip_pattern = re.compile(r"((25[0-5]|2[0-4][0-9]|1[0-9]{2}|[1-9][0-9]|[0-9])\.){3}(25[0-5]|2[0-4][0-9]|1[0-9]{2}|[1-9][0-9]|[0-9])")


def extract_text_from_html_bytes(html_bytes: bytes) -> str:
    try:
        html = html_bytes.decode("utf-8")
    except UnicodeDecodeError:
        encoding = detect_encoding(html_bytes)
        html = html_bytes.decode(encoding, errors="replace")

    return extract_plain_text(html)


def identify_language(text: str)-> tuple[Any, float]:
    labels, scores = language_identify_model.predict(text.replace("\n", " "))
    language = labels[0].removeprefix("__label__")
    score = float(scores[0])

    return language, score


def mask_emails(text: str) -> tuple[str, int]:
    new_text, count = email_pattern.subn("|||EMAIL_ADDRESS|||", text)
    return new_text, count


def mask_phone_numbers(text: str) -> tuple[str, int]:
    new_text, count = phone_number_pattern.subn("|||PHONE_NUMBER|||", text)
    return new_text, count


def mask_ips(text: str) -> tuple[str, int]:
    new_text, count = ip_pattern.subn("|||IP_ADDRESS|||", text)
    return new_text, count


def classify_nsfw(text: str) -> tuple[Any, float]:
    labels, scores = nsfw_identify_model.predict(text.replace("\n", " "))
    language = labels[0].removeprefix("__label__")
    score = float(scores[0])

    return language, score


def classify_toxic_speech(text: str) -> tuple[Any, float]:
    labels, scores = toxic_speech_identify_model.predict(text.replace("\n", " "))
    language = labels[0].removeprefix("__label__")
    score = float(scores[0])

    return language, score


def gopher_quality_filter(text: str) -> bool:
    words = nltk.word_tokenize(text)
    if not (len(words) >= 50 and len(words) <= 1e5):
        return False

    word_len_avg = sum(len(word) for word in words) / len(words)
    if not (word_len_avg >= 3 and word_len_avg <= 10):
        return False

    lines = text.splitlines()
    ellipsis_end_ratio = sum(1 for line in lines if line.endswith("...")) / len(lines)
    if ellipsis_end_ratio > 0.3:
        return False

    one_alphabetic_ratio = sum(1 for word in words if word.isalpha()) / len(words)
    if one_alphabetic_ratio < 0.8:
        return False

    return True


def classify_quality(text: str) -> tuple[Any, float]:
    raise NotImplementedError