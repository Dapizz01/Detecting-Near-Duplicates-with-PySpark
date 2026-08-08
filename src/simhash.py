from hashlib import blake2b

import numpy as np
import pandas as pd
from numba import njit
from pyspark.sql.functions import pandas_udf
from pyspark.sql.types import BinaryType


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


# https://www.dynatrace.com/news/blog/speeding-up-simhash-by-10x/
@pandas_udf(BinaryType())  # type: ignore
def fastsimhash(hashes: pd.Series) -> pd.Series:
    """Aggregates a batch of hashes into a single fingerprint.
    Based on the FastSimHash algorithm but uses 32 bit counters instead of 8 bit ones,
    avoiding the carry problem but using more space than necessary.
    It's about 8x more efficient than *simhash* on average, using SIMD operations (SWAR) to update the counters.

    Args:
        hashes (pd.Series): series of hashes (of the same length)

    Returns:
        pd.Series: a fingerprint
    """

    def fast_merge(byte_list: list[bytes]) -> bytes:
        """Helper function, bundles all the hashes into a matrix and calls *_numba_fastsimhash*.

        Args:
            byte_list (list[bytes]): list of hashes

        Returns:
            bytes: fingerprint
        """
        if byte_list is None or len(byte_list) == 0:
            raise Exception("An empty byte_list has been passed to fast_merge")
        n = len(byte_list)
        hash_len = len(byte_list[0])  # Returns num of bytes in an hash
        # Aggregates hashes into a matrix.
        # byte_matrix[i][j] = j-th bit of hash i
        byte_matrix = np.frombuffer(b"".join(byte_list), dtype=np.uint8).reshape(n, hash_len)
        fingerprint_list = _numba_fastsimhash(byte_matrix, n, hash_len)
        # Merges list of integers (parts of fingerprint) into a single bytes Python value (the fingerprint)
        return fingerprint_list.tobytes()

    return hashes.apply(fast_merge)  # TODO: replace apply for list comprehension? ...maybe?


# TODO: Refactor name
def fastsimhash_test(hashes: list[bytes]) -> bytes:
    """Helper function, bundles all the hashes into a matrix and calls *_numba_fastsimhash*.

    Args:
        byte_list (list[bytes]): list of hashes

    Returns:
        bytes: fingerprint
    """
    if hashes is None or len(hashes) == 0:
        raise Exception("An empty byte_list has been passed to fast_merge")
    n = len(hashes)
    hash_len = len(hashes[0])  # Returns num of bytes in an hash
    # Aggregates hashes into a matrix.
    # byte_matrix[i][j] = j-th bit of hash i
    byte_matrix = np.frombuffer(b"".join(hashes), dtype=np.uint8).reshape(n, hash_len)
    fingerprint_list = _numba_fastsimhash(byte_matrix, n, hash_len)
    # Merges list of integers (parts of fingerprint) into a single bytes Python value (the fingerprint)
    return fingerprint_list.tobytes()


@pandas_udf(BinaryType())  # type: ignore
def simhash(hashes: pd.Series) -> pd.Series:
    """Reduces a list of hashes into a single hash.
    Each bit of the result has is 1 iff at least half of the input hashes have value 1 in the same index, 0 otherwise.

    Args:
        hashes (pd.Series): list of BinaryType hashes. All hashes must have the same length.

    Returns:
        pd.Series: a fingerprinta fingerprint, BinaryType.
    """

    def merge(byte_list):
        if byte_list is None or len(byte_list) == 0:
            return None

        n = len(byte_list)

        # Each hash is casted into an array of single bits
        bit_arrays = [np.unpackbits(np.frombuffer(b, dtype=np.uint8)) for b in byte_list]

        # Stacks the arrays vertically (row = hash index, col = bit index)
        stacked = np.vstack(bit_arrays)

        # Vertical sum
        bit_sum = stacked.sum(axis=0)

        fingerprint_bits = (bit_sum * 2 >= n).astype(np.uint8)

        # Casts from array of bits to binary
        fingerprint = np.packbits(fingerprint_bits).tobytes()
        return fingerprint

    return hashes.apply(merge)


# TODO: Refactor name
@pandas_udf(BinaryType())  # type: ignore
def simhash_shingles(
    text: pd.Series, window_size: pd.Series, fingerprint_size: pd.Series
) -> pd.Series:
    def compute(text: str, window_size: int, fingerprint_size: int) -> bytes:
        if not text or window_size >= len(text):
            raise Exception("[compute_shingles_simhash]: invalid text or window_size")
        shingles = [text[i : i + window_size] for i in range(len(text) - window_size + 1)]
        # TODO: change with faster hash function
        hashes = [
            blake2b(shingle.encode(), digest_size=int(fingerprint_size / 8)).digest()
            for shingle in shingles
        ]
        return fastsimhash_test(hashes=hashes)

    return pd.Series([compute(t, ws, fs) for t, ws, fs in zip(text, window_size, fingerprint_size)])


@pandas_udf(BinaryType())  # type: ignore
def simhash_bow(text: pd.Series, fingerprint_size: pd.Series) -> pd.Series:
    def compute(text: str, fingerprint_size: int) -> bytes:
        if not text:
            raise Exception("[compute]: invalid text for bag_of_words")
        tokens = text.split()
        # TODO: change with faster hash function
        hashes = [
            blake2b(token.encode(), digest_size=int(fingerprint_size / 8)).digest()
            for token in tokens
        ]
        return fastsimhash_test(hashes=hashes)

    return pd.Series([compute(t, fs) for t, fs in zip(text, fingerprint_size)])
