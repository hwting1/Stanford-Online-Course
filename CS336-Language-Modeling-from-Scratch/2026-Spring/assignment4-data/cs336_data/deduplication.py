import os
import mmh3, random
from pathlib import Path
from collections import Counter
import itertools
import unicodedata
import string
import regex as re
import nltk
import numpy as np


def exact_line_deduplication(
    input_files: list[os.PathLike], output_directory: os.PathLike
):
    counter = Counter()
    for file in input_files:
        with open(file, "r", encoding="utf-8") as f:
            for line in f:
                h = mmh3.hash(line)
                counter[h] += 1

    for file in input_files:
        file = Path(file)
        with open(output_directory / file.name, "w", encoding="utf-8") as wf:
            with open(file, "r", encoding="utf-8") as rf:
                for line in rf:
                    if counter[mmh3.hash(line)] == 1:
                        wf.write(line)



def normalize(text: str) -> str:
    # 1. lowercase
    text = text.lower()

    # 2. NFD unicode normalization
    text = unicodedata.normalize("NFD", text)

    # 3. remove accents / combining marks
    text = "".join(
        ch for ch in text
        if not unicodedata.combining(ch)
    )

    # 4. remove punctuation
    text = re.sub(rf"[{re.escape(string.punctuation)}]", "", text)

    # 5. normalize whitespace
    text = re.sub(r"\s+", " ", text).strip()

    return text


def make_ngrams(text: str, n: int) -> set[str]:
    words = nltk.word_tokenize(text)
    n_grams = set()
    for i in range(len(words) - n + 1):
        n_grams.add(" ".join(words[i : i + n]))
    return n_grams


def compute_jaccard(A: set, B: set) -> float:
    intersection = len(A & B)
    union = len(A | B)
    return intersection / union


def minhash(S: set[str], seed: int):
    return min(mmh3.hash(x, seed) for x in S)


def minhash_deduplication(
    input_files: list[os.PathLike],
    num_hashes: int,
    num_bands: int,
    ngrams: int,
    jaccard_threshold: float,
    output_directory: os.PathLike,
):
    assert num_hashes % num_bands == 0

    buckets = {}
    document_ngrams = {}
    document_texts = {}

    # --------------------------------------------------
    # 1. Compute n-grams + MinHash signatures + LSH buckets
    # --------------------------------------------------

    for file in input_files:
        file = Path(file)

        with open(file, "r", encoding="utf-8") as f:
            original_text = f.read()

        document_texts[file] = original_text
        text = normalize(original_text)
        n_grams = make_ngrams(text, ngrams)
        document_ngrams[file] = n_grams

        # If the document is too short to produce any n-grams,
        # skip MinHash/LSH for it.
        if not n_grams:
            continue

        signature = [
            minhash(n_grams, seed)
            for seed in range(num_hashes)
        ]

        bands = np.array(signature).reshape(num_bands, -1)

        for band_idx, band in enumerate(bands):
            key = (band_idx, tuple(band.tolist()))
            buckets.setdefault(key, []).append(file)

    # --------------------------------------------------
    # 2. Find candidate pairs from LSH collisions
    # --------------------------------------------------

    candidate_pairs = set()

    for files in buckets.values():
        if len(files) < 2:
            continue

        for i in range(len(files)):
            for j in range(i + 1, len(files)):
                candidate_pairs.add((files[i], files[j]))

    # --------------------------------------------------
    # 3. Compute true Jaccard similarity
    # --------------------------------------------------

    duplicate_pairs = []

    for file_a, file_b in candidate_pairs:
        similarity = compute_jaccard(
            document_ngrams[file_a],
            document_ngrams[file_b],
        )

        if similarity > jaccard_threshold:
            duplicate_pairs.append((file_a, file_b))

    # --------------------------------------------------
    # 4. Cluster duplicates
    #
    # Union-Find / Disjoint Set
    # --------------------------------------------------

    parent = {Path(file): Path(file) for file in input_files}

    def find(x):
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]

    def union(a, b):
        root_a = find(a)
        root_b = find(b)

        if root_a != root_b:
            parent[root_b] = root_a

    for file_a, file_b in duplicate_pairs:
        union(file_a, file_b)

    # --------------------------------------------------
    # 5. Gather documents into clusters
    # --------------------------------------------------

    clusters = {}

    for file in parent:
        root = find(file)
        clusters.setdefault(root, []).append(file)

    # --------------------------------------------------
    # 6. Keep one random document from each cluster
    # --------------------------------------------------

    files_to_keep = set()

    for cluster in clusters.values():
        files_to_keep.add(random.choice(cluster))

    # --------------------------------------------------
    # 7. Write retained documents
    # --------------------------------------------------

    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)

    for file in files_to_keep:
        output_path = output_directory / file.name

        with open(output_path, "w", encoding="utf-8") as f:
            f.write(document_texts[file])



def minhash_deduplication(
    input_files: list[os.PathLike],
    num_hashes: int,
    num_bands: int,
    ngrams: int,
    jaccard_threshold: float,
    output_directory: os.PathLike,
):
    buckets = {}
    document_ngrams = {}
    document_texts = {}
    for file in input_files:
        with open(file, "r", encoding="utf-8") as f:
            original_text = f.read()

        document_texts[file] = original_text
        text = normalize(original_text)
        n_grams = make_ngrams(text, ngrams)
        document_ngrams[file] = n_grams
        signature = [minhash(n_grams, seed) for seed in range(num_hashes)]
        bands = np.array(signature).reshape(num_bands, -1)

        for band_idx, band in enumerate(bands):
            key = (band_idx, tuple(band))
            buckets.setdefault(key, []).append(file)

    candidate_pairs = set()
    for groups in buckets.values():
        if len(groups) > 1:
            for comb in itertools.combinations(groups, 2):
                candidate_pairs.add(tuple(comb))

    duplicate_groups = []
    for file_a, file_b in candidate_pairs:
        ngrams_a, ngrams_b = document_ngrams[file_a], document_ngrams[file_b]
        sim = compute_jaccard(ngrams_a, ngrams_b)
        if sim > jaccard_threshold:
            group_a = None
            group_b = None

            for group in duplicate_groups:
                if file_a in group:
                    group_a = group
                if file_b in group:
                    group_b = group

            if group_a is None and group_b is None:
                duplicate_groups.append({file_a, file_b})

            elif group_b is None:
                group_a.add(file_b)

            elif group_a is None:
                group_b.add(file_a)

            elif group_a is not group_b:
                group_a.update(group_b)
                duplicate_groups.remove(group_b)

    files_to_remove = set()
    for files in duplicate_groups:
        keep_file = random.choice(list(files))
        files_to_remove.update(files - {keep_file})

    for file in document_texts:
        if file not in files_to_remove:
            file = Path(file)
            with open(output_directory / file.name, "w", encoding="utf-8") as f:
                f.write(document_texts[file])