import math
from functools import reduce
from itertools import combinations
from typing import List, Tuple

import pandas as pd
import pyspark.sql.functions as F
from pyspark.sql import DataFrame
from pyspark.sql.types import BinaryType, IntegerType

from src.fingerprint import FINGERPRINT_SIZES
from src.metrics import Metrics, extract_metrics


@F.pandas_udf(IntegerType())  # type: ignore
def hamming_distance_bin(col1: pd.Series, col2: pd.Series) -> pd.Series:
    def hamming(b1, b2):
        if b1 is None or b2 is None or pd.isna(b1) or pd.isna(b2):
            return None
        return (int.from_bytes(b1, "big") ^ int.from_bytes(b2, "big")).bit_count()

    return pd.Series([hamming(b1, b2) for b1, b2 in zip(col1, col2)], dtype="Int64")


def near_duplicates_pairs(df: DataFrame, f1: str, f2: str, K: int) -> DataFrame:
    assert f1 in df.columns
    assert f2 in df.columns
    df = df.withColumn(
        "hamming_distance",
        hamming_distance_bin(F.col(f1), F.col(f2)),  # type: ignore
    )
    df = df.withColumn(
        "computed_near_duplicate",
        F.when(
            F.col("hamming_distance") <= K,  # type: ignore
            F.lit(True),
        ).otherwise(F.lit(False)),
    )
    return df


# TODO: finish (and update) docstring
def near_duplicates_benchmark(
    df: DataFrame, f1_col: str, f2_col: str, truth_col: str, K: int
) -> Metrics:
    """Returns metrics about the efficacy...

    Args:
        df (DataFrame): the input PySpark DataFrame
        f1_col (str): column name of *df* containing a fingerprint (BinaryType)
        f2_col (str): column name of *df* containing the other fingerprint (BinaryType)
        truth_col (str): column name of *df* (IntType), source of truth -> 1 means near-duplicate, 0 otherwise
        K (int): specifies the maximum hamming distance allowed between two fingerprints to be considered near-duplicates

    Returns:
        Metrics: the result metrics
    """
    assert f1_col in df.columns
    assert f2_col in df.columns
    assert truth_col in df.columns

    df = df.withColumn(
        "hamming_distance",
        hamming_distance_bin(F.col(f1_col), F.col(f2_col)),  # type: ignore
    )

    df = df.withColumn(
        "computed_near_duplicate",
        F.when(
            F.col("hamming_distance") <= K,  # type: ignore
            F.lit(True),
        ).otherwise(F.lit(False)),
    )

    df = df.drop("hamming_distance")
    return extract_metrics(df, test_col="computed_near_duplicate", truth_col=truth_col)


def _partition_blocks(fingerprint_size: int, n_blocks: int) -> List[Tuple[int, int]]:
    base, remainder = divmod(fingerprint_size, n_blocks)
    sizes = [base + 1 if i < remainder else base for i in range(n_blocks)]
    blocks = []
    carry = 0
    for s in sizes:
        blocks.append((carry, s))
        carry += s
    return blocks


def _block_combinations(n_blocks: int, K: int) -> List[Tuple[int, ...]]:
    if n_blocks <= K:
        raise ValueError()
    r = n_blocks - K
    return list(combinations(range(n_blocks), r))


# TODO: add return type hint
def _make_bit_key_udf(bit_ranges: List[Tuple[int, int]], fingerprint_size: int):
    key_len_bits = sum(length for _, length in bit_ranges)
    key_len_bytes = (key_len_bits + 7) // 8

    def extract(fp: bytes):
        if fp is None:
            return None
        val = int.from_bytes(fp, "big")
        key_val = 0
        for start, length in bit_ranges:
            shift = fingerprint_size - start - length
            mask = (1 << length) - 1
            key_val = (key_val << length) | ((val >> shift) & mask)
        return key_val.to_bytes(key_len_bytes, "big")

    @F.pandas_udf(BinaryType())  # type: ignore
    def bit_key(col: pd.Series) -> pd.Series:
        return pd.Series([extract(fp) for fp in col], dtype="object")

    return bit_key


def _optimal_n_blocks(
    fingerprint_size: int, K: int, d: float, max_tables: int = 256, per_table_overhead: float = 1.0
) -> int:
    best_m, best_cost = None, math.inf
    for m in range(K + 1, fingerprint_size + 1):
        t = math.comb(m, m - K)
        if t > max_tables:
            break
        p_i = ((m - K) * fingerprint_size) // m
        expected_candidates = 2 ** (d - p_i)
        cost = t * (per_table_overhead + expected_candidates)
        if cost < best_cost:
            best_m, best_cost = m, cost
    if best_m is None:
        raise ValueError()
    return best_m


def near_duplicates_optimized(
    df: DataFrame,
    query: DataFrame,
    id_col: str,
    df_fingerprint_col: str,
    query_fingerprint_col: str,
    fingerprint_size: int,
    K: int,
    broadcast: bool,
    max_tables: int = 256,
    per_table_overhead: float = 1.0,
) -> DataFrame:
    # TODO: asserts
    d = math.log2(max(df.count(), 1))
    n_blocks = _optimal_n_blocks(
        fingerprint_size=fingerprint_size,
        K=K,
        d=d,
        max_tables=max_tables,
        per_table_overhead=per_table_overhead,
    )
    blocks = _partition_blocks(fingerprint_size=fingerprint_size, n_blocks=n_blocks)
    combos = _block_combinations(n_blocks=n_blocks, K=K)

    results = []
    for combo in combos:
        bit_ranges = [blocks[i] for i in combo]
        key_udf = _make_bit_key_udf(bit_ranges=bit_ranges, fingerprint_size=fingerprint_size)

        df_table = df.select(
            id_col,
            F.col(df_fingerprint_col).alias("f1"),
            key_udf(F.col(df_fingerprint_col)).alias("block_key"),  # type: ignore
        )

        query_table = query.select(
            id_col,
            F.col(query_fingerprint_col).alias("f2"),
            key_udf(F.col(query_fingerprint_col)).alias("block_key"),  # type: ignore
        )

        if broadcast:
            query_table = F.broadcast(query_table)

        join_result = (
            df_table.alias("a")
            .join(query_table.alias("b"), on="block_key")
            .select(F.col("a.f1"), F.col("b.f2"))
        )
        results.append(join_result)

    candidates = reduce(lambda a, b: a.unionByName(b), results).drop_duplicates(["f1", "f2"])
    return near_duplicates_pairs(candidates, f1="f1", f2="f2", K=K)


def near_duplicates(
    df: DataFrame,
    query: DataFrame,
    id_col: str,
    df_fingerprint_col: str,
    query_fingerprint_col: str,
    fingerprint_size: int,
    K: int,
    broadcast: bool,
    optimized: bool,
) -> DataFrame:
    assert df.count() > 0
    assert query.count() > 0
    assert df_fingerprint_col in df.columns
    assert id_col in df.columns
    assert query_fingerprint_col in query.columns
    assert id_col in query.columns
    assert fingerprint_size in FINGERPRINT_SIZES

    def partition_tables(fingerprint_size: int, n_partitions: int) -> list:
        base, remainder = divmod(fingerprint_size, n_partitions)
        sizes = [base + 1 if i < remainder else base for i in range(n_partitions)]
        carry = 0
        partitions = []
        for s in sizes:
            partitions.append([carry, s])
            carry += s
        return partitions

    nd_df = df
    if not optimized:
        nd_df = (
            df.alias("a")
            .crossJoin(query.select(query_fingerprint_col).alias("b"))
            .select(
                F.col(id_col),
                F.col(f"a.{df_fingerprint_col}").alias("f1"),
                F.col(f"b.{query_fingerprint_col}").alias("f2"),
            )
        )
        nd_df = near_duplicates_pairs(nd_df, "f1", "f2", K=K)
    else:
        nd_df = near_duplicates_optimized(
            df=df,
            query=query,
            id_col=id_col,
            df_fingerprint_col=df_fingerprint_col,
            query_fingerprint_col=query_fingerprint_col,
            fingerprint_size=fingerprint_size,
            K=K,
            broadcast=broadcast,
        )

    return nd_df
