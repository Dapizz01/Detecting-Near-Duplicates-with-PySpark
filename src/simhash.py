import numpy as np
import pandas as pd
from pyspark.sql.functions import pandas_udf
from pyspark.sql.types import BinaryType


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
