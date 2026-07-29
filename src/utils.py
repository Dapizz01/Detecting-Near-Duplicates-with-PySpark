from pyspark.sql import Column
from pyspark.sql import functions as F

MAX_INT64 = 9223372036854775807  # 2^63 - 1
TWO_POW_64 = 18446744073709551616  # 2^64


def hexchunk_to_bigint(chunk: Column) -> Column:
    """Transforms an hex chunk of 16 chars (64 bits) in a 64-bit signed integer (BigInt in PySpark)
    Equivalent to *F.conv(chunk, 16, 10)*, but with an overflow check due to BigInt being signed.

    Args:
        chunk (Column): input column, a String value

    Returns:
        Column: output column, a 64-bit signed integer value
    """
    unsigned = F.conv(chunk, 16, 10).cast("decimal(38,0)")
    return (
        F.when(unsigned > F.lit(MAX_INT64), unsigned - F.lit(TWO_POW_64))
        .otherwise(unsigned)
        .cast("long")
    )


def hex_to_bigint(col: Column) -> Column:
    """Transforms a list of hex strings in a nested list of 64-bit signed integers (BigInt, in PySpark).
    Each hex string must be a multiple of divisible by 64.

    Args:
        col (Column): input column, array<string>

    Returns:
        Column: output column, array<array<BigInt>>
    """
    return F.transform(
        col, lambda hex_str: F.transform(F.split(hex_str, "(?<=\\G.{16})"), hexchunk_to_bigint)
    )


# TODO: implement weight for TF-IDF
def merge_fingerprints_fragments(col: Column, n_blocks: int, K: int, weights) -> Column:
    assert n_blocks in (1, 2, 4, 8)
    len_fragment = 64
    acc = F.lit(0).cast("bigint")
    for b in range(len_fragment):
        bit_vals = F.transform(
            col,
            lambda bit_val: F.when(
                (F.shiftright(bit_val.getItem(K), b) & F.lit(1)) == 1, F.lit(1)
            ).otherwise(F.lit(-1)),
        )
        bit_sum = F.aggregate(bit_vals, F.lit(0), lambda prev, v: prev + v)
        quant_bit = F.when(bit_sum > 0, F.lit(1)).otherwise(F.lit(0)).cast("bigint")
        acc = acc.bitwiseOR(F.shiftleft(quant_bit, b))
    return acc


def merge_fingerprints(col: Column, n_blocks: int, K: int) -> Column:
    fragments = [merge_fingerprints_fragments(col, n_blocks, K, None) for K in range(n_blocks)]
    return F.array(fragments)
