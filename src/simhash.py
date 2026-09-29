from hashlib import blake2b

import numpy as np
import pandas as pd
from numba import njit
from pyspark.sql.functions import pandas_udf
from pyspark.sql.types import BinaryType

# There is no simhash for embeddings - embeddings already return one fingerprint for each text.


@njit(fastmath=True)
def _numba_fastsimhash(byte_matrix: np.ndarray, n_hashes: int, hash_len: int) -> np.ndarray:
    """Implementation of FastSimHash compiled at runtime with Numba, using LLVM.
    The actual SIMD instructions are created by LLVM, optimizing the code during compile-time.

    Args:
        byte_matrix (np.ndarray): list of hashes to merge
        n_hashes (int): number of hashes in byte_matrix
        hash_len (int): length of an hash, in **bytes**

    Returns:
        np.ndarray: a fingerprint
    """
    # One 32-bit counter for every bit (8 for each byte)
    counters = np.zeros(hash_len * 8, dtype=np.int32)

    if n_hashes == 0:
        raise ValueError("[_numba_fastsimhash] n_hashes has value zero. Is byte_matrix empty?")
    if hash_len == 0:
        raise ValueError("[_numba_fastsimhash] hash_len has value zero. Is byte_matrix empty?")

    # Iterate over each hash and each byte index
    for i in range(n_hashes):
        for j in range(hash_len):
            # Get j-th byte
            val = byte_matrix[i, j]
            # Skips previous counters
            base_idx = j * 8

            # Updates each counter of the bits inside the byte,
            # shifting from left to right and masking with 1
            counters[base_idx] += (val >> 7) & 1  # Most significant (left)
            counters[base_idx + 1] += (val >> 6) & 1
            counters[base_idx + 2] += (val >> 5) & 1
            counters[base_idx + 3] += (val >> 4) & 1
            counters[base_idx + 4] += (val >> 3) & 1
            counters[base_idx + 5] += (val >> 2) & 1
            counters[base_idx + 6] += (val >> 1) & 1
            counters[base_idx + 7] += val & 1  # Least significant (right)

    output = np.zeros(hash_len, dtype=np.uint8)
    threshold = (n_hashes + 1) // 2
    for i in range(hash_len):
        base_idx = i * 8
        part_sum = 0  # Integer value of a byte
        # For each bit, if the counter value is bigger than the threshold
        # the bit will be set to 1 in the result fingerprint.
        if counters[base_idx] >= threshold:  # Most significant
            part_sum |= 128
        if counters[base_idx + 1] >= threshold:
            part_sum |= 64
        if counters[base_idx + 2] >= threshold:
            part_sum |= 32
        if counters[base_idx + 3] >= threshold:
            part_sum |= 16
        if counters[base_idx + 4] >= threshold:
            part_sum |= 8
        if counters[base_idx + 5] >= threshold:
            part_sum |= 4
        if counters[base_idx + 6] >= threshold:
            part_sum |= 2
        if counters[base_idx + 7] >= threshold:  # Least significant
            part_sum |= 1
        output[i] = part_sum  # The sum is saved as an integer
    return output


def fastsimhash(hashes: list[bytes]) -> bytes:
    """Aggregates a batch of hashes into a single fingerprint.
    Based on the FastSimHash algorithm but uses 32 bit counters instead of 8 bit ones,
    avoiding the carry problem but using more space than necessary.
    It's about 8x more efficient than *simhash* on average, using SIMD operations (SWAR) to update the counters.

    Args:
        hashes (list[bytes]): same-length binary values

    Returns:
        bytes: a fingerprint
    """
    if hashes is None or len(hashes) == 0:
        raise ValueError("[fastsimhash] An empty byte_list has been passed.")
    n = len(hashes)
    hash_len = len(hashes[0])  # Returns num of bytes in an hash
    # Aggregates hashes into a matrix.
    # byte_matrix[i][j] = j-th bit of hash i
    byte_matrix = np.frombuffer(b"".join(hashes), dtype=np.uint8).reshape(n, hash_len)
    fingerprint_list = _numba_fastsimhash(byte_matrix, n, hash_len)
    # Merges list of integers (parts of fingerprint) into a single bytes Python value (the fingerprint)
    return fingerprint_list.tobytes()


@pandas_udf(BinaryType())  # type: ignore
def simhash_tf_idf(tokens: pd.Series, weights: pd.Series, fingerprint_size: pd.Series) -> pd.Series:
    # The simhash for TF-IDF is calculated on the TF-IDF weights, which are floats.
    # Therefore, it's not possible to use fastsimhash here, since it requires bit strings.
    def compute(token_list: list[str], weight_list: list[float], fingerprint_size: int):
        if token_list is None or len(token_list) == 0:
            raise ValueError("[simhash_tf_idf] Token list is None or empty.")

        if len(token_list) != len(weight_list):
            raise ValueError(
                f"[simhash_tf_idf] Token list ({len(token_list)}) length mismatches with weights list ({len(weight_list)})."
            )

        hashes = [
            blake2b(token.encode(), digest_size=int(fingerprint_size / 8)).digest()
            for token in token_list
        ]

        # Unpacks binary value as array of integers (each integer is a bit)
        bit_arrays = [np.unpackbits(np.frombuffer(b, dtype=np.uint8)) for b in hashes]
        bits_matrix = np.vstack(bit_arrays)
        ints_matrix = np.where(bits_matrix == 1, 1, -1)  # Map 0 to -1 for negative weight

        # Transform list to a vertical array (column)
        weights_col = np.array(weight_list, dtype=np.float64)[:, np.newaxis]
        weighted_matrix = ints_matrix * weights_col  # Matrix product

        # Sum each column vertically
        col_sum = weighted_matrix.sum(axis=0)

        # Quantize
        fingerprint_bits = (col_sum > 0).astype(np.uint8)

        # Pack integers into a binary value
        fingerprint = np.packbits(fingerprint_bits).tobytes()

        return fingerprint

    return pd.Series([compute(t, w, fs) for t, w, fs in zip(tokens, weights, fingerprint_size)])


@pandas_udf(BinaryType())  # type: ignore
def simhash_shingles_over_chars(
    text: pd.Series, window_size: pd.Series, fingerprint_size: pd.Series
) -> pd.Series:
    def compute(text: str, window_size: int, fingerprint_size: int) -> bytes:
        if text is None or text.strip() == "":
            raise ValueError(f"[simhash_shingles_over_chars] The input text ({text}) is empty.")
        if window_size <= len(text):
            shingles = [text]
        else:
            shingles = [text[i : i + window_size] for i in range(len(text) - window_size + 1)]
        if shingles == []:
            raise Exception("[simhash_shingles_over_chars] Error during shingles extraction.")
        hashes = [
            blake2b(shingle.encode(), digest_size=int(fingerprint_size / 8)).digest()
            for shingle in shingles
        ]
        return fastsimhash(hashes=hashes)

    return pd.Series([compute(t, ws, fs) for t, ws, fs in zip(text, window_size, fingerprint_size)])


@pandas_udf(BinaryType())  # type: ignore
def simhash_shingles_over_words(
    text: pd.Series, window_size: pd.Series, fingerprint_size: pd.Series
) -> pd.Series:
    def compute(text: str, window_size: int, fingerprint_size: int) -> bytes:
        if text is None or not text.strip():
            raise ValueError(f"[simhash_shingles_over_words] The input text ({text}) is empty.")
        words = text.split()
        if len(words) <= window_size:
            shingles = words
        else:
            shingles = [
                "".join(words[i : i + window_size]) for i in range(len(words) - window_size + 1)
            ]
        hashes = [
            blake2b(shingle.encode(), digest_size=int(fingerprint_size / 8)).digest()
            for shingle in shingles
        ]
        return fastsimhash(hashes=hashes)

    return pd.Series([compute(t, ws, fs) for t, ws, fs in zip(text, window_size, fingerprint_size)])


@pandas_udf(BinaryType())  # type: ignore
def simhash_bow(text: pd.Series, fingerprint_size: pd.Series) -> pd.Series:
    def compute(text: str, fingerprint_size: int) -> bytes:
        if text is None or not text.strip():
            raise ValueError(f"[simhash_bow] The input text ({text}) is empty.")
        tokens = [x for x in text.split() if x]
        hashes = [
            blake2b(token.encode(), digest_size=int(fingerprint_size / 8)).digest()
            for token in tokens
        ]
        return fastsimhash(hashes=hashes)

    return pd.Series([compute(t, fs) for t, fs in zip(text, fingerprint_size)])
