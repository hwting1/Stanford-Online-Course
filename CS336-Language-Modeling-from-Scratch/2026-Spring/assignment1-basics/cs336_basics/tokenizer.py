import os
import regex as re
from typing import BinaryIO
import multiprocessing as mp
from collections import Counter, defaultdict

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
PAT = re.compile(PAT)

def find_chunk_boundaries(
    file: BinaryIO,
    desired_num_chunks: int,
    split_special_token: bytes,
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    assert isinstance(split_special_token, bytes), "Must represent special token as a bytestring"

    # Get total file size in bytes
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks

    # Initial guesses for chunk boundary locations, uniformly spaced
    # Chunks start on previous index, don't include last index
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)  # Start at boundary guess
        while True:
            mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

            # If EOF, this boundary should be at the end of the file
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            # Find the special token in the mini chunk
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))


def pre_tokenize(
    input_path,
    boundary,
    special_pattern,

):
    counts = Counter()
    with open(input_path, "rb") as f:
        start, end = boundary
        f.seek(start)
        raw_bytes = f.read(end - start)
        corpus = raw_bytes.decode("utf-8", errors="ignore")
        segments = special_pattern.split(corpus)
        for segment in segments:
            for match in PAT.finditer(segment):
                token_bytes = match.group(0).encode("utf-8")
                token_tuple = tuple(token_bytes)
                counts[token_tuple] += 1

    return counts


def _pre_tokenize_job(job):
    input_path, boundary, special_pattern = job
    return pre_tokenize(input_path, boundary, special_pattern)


def count_pretokens_multiprocessing(
    input_path,
    boundaries,
    special_pattern,
    num_processes,
):
    jobs = (
        (input_path, (start, end), special_pattern)
        for start, end in zip(boundaries[:-1], boundaries[1:])
    )

    pretoken_counts = Counter()
    with mp.Pool(processes=num_processes) as pool:
        for local_counts in pool.imap_unordered(
            _pre_tokenize_job,
            jobs,
            chunksize=1,
        ):
            pretoken_counts.update(local_counts)

    return pretoken_counts


def merge(
        pretoken_counts,
        pair_counts,
        pair_to_pretokens,
        pair: tuple[int, int],
        new_index: int
) -> list[int]:

    old_pretokens, new_pretokens, freqs = set(), set(), []
    affected = list(pair_to_pretokens[pair])
    for pretoken in affected:
        i = 0
        new_indices, new_pairs, old_pairs,  = [], [], []
        freq = pretoken_counts[pretoken]

        for j in range(len(pretoken)-1):
            old_pairs.append((pretoken[j], pretoken[j+1]))
        while i < len(pretoken):
            if i + 1 < len(pretoken) and pretoken[i] == pair[0] and pretoken[i + 1] == pair[1]:
                new_indices.append(new_index)
                i += 2
            else:
                new_indices.append(pretoken[i])
                i += 1
        new_indices = tuple(new_indices)
        for j in range(len(new_indices)-1):
            new_pairs.append((new_indices[j], new_indices[j+1]))

        pretoken_counts[new_indices] += freq
        del pretoken_counts[pretoken]
        old_pretokens.add(pretoken)
        new_pretokens.add(new_indices)

        for old_pair in old_pairs:
            pair_counts[old_pair] -= freq
            if pair_counts[old_pair] <= 0:
                del pair_counts[old_pair]
            pair_to_pretokens[old_pair].discard(pretoken)
            if not pair_to_pretokens:
                del pair_to_pretokens[old_pair]

        for new_pair in new_pairs:
            pair_counts[new_pair] += freq
            pair_to_pretokens[new_pair].add(new_indices)

    return pretoken_counts, pair_counts, pair_to_pretokens


def train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
    **kwargs,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:

    merges: list[tuple[bytes, bytes]] = []
    vocab: dict[int, bytes] = {x: bytes([x]) for x in range(256)}
    num_processes = kwargs.get("num_processes", 4)
    start_idx = len(vocab)
    for i, token in enumerate(special_tokens):
        vocab[start_idx + i] = token.encode("utf-8")

    # max_merges = vocab_size - 256 - len(special_tokens)
    end_token = special_tokens[0]
    special_pattern = re.compile(
        "|".join(re.escape(token) for token in special_tokens)
    )

    with open(input_path, "rb") as f:
        boundaries = find_chunk_boundaries(f, num_processes, end_token.encode("utf-8"))

    pretoken_counts = count_pretokens_multiprocessing(
        input_path=input_path,
        boundaries=boundaries,
        special_pattern=special_pattern,
        num_processes=num_processes,
    )

    new_token_id = len(vocab)
    pair_counts = Counter()
    pair_to_pretokens = defaultdict(set)
    for pretoken, freq in pretoken_counts.items():
        for i in range(len(pretoken) - 1):
            pair = (pretoken[i], pretoken[i + 1])
            pair_counts[pair] += freq
            pair_to_pretokens[pair].add(pretoken)

    while new_token_id < vocab_size:
        if not pair_counts:
            break

        best_pair = max(pair_counts, key=lambda p: (pair_counts[p], vocab[p[0]], vocab[p[1]]))
        left, right = best_pair
        merges.append((vocab[left],vocab[right]))
        vocab[new_token_id] = vocab[left] + vocab[right]
        pretoken_counts, pair_counts, pair_to_pretokens = merge(pretoken_counts, pair_counts,
                                                                pair_to_pretokens, best_pair, new_token_id)
        new_token_id += 1

    return vocab, merges
