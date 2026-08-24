import itertools
import math
from functools import lru_cache
from typing import List

import numpy as np
import pandas as pd
import pyspark.sql.functions as F
from pyspark.sql import DataFrame
from pyspark.sql.types import ArrayType, BinaryType, IntegerType

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
    ).filter(F.col("hamming_distance") <= K)
    """
    df = df.withColumn(
        "computed_near_duplicate",
        F.when(
            F.col("hamming_distance") <= K,  # type: ignore
            F.lit(True),
        ).otherwise(F.lit(False)),
    )
    """
    return df.drop("hamming_distance")


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


# TODO: it doesn't seem to work well on some inputs
def estimate_optimal_schema(f: int, K: int, d: int, p_min: int) -> tuple[int, list[dict]]:
    """Estimates the optimal join schema to find near-duplicates.
    Can return multiple schemas, to be applied sequentially.

    Args:
        f (int): Length of fingerprints in bits
        K (int): Maximum hamming distance allowed
        d (int): Number of fingerprints (in log2)
        p_min (int): Maximum number of candidate fingerprints (in log2)

    Raises:
        ValueError: Instances for which there doesn't exist a schema - a cartesian product is required.

    Returns:
        tuple[int, list[dict]]: Computed schema
    """

    @lru_cache(maxsize=None)
    def X(f: int, d: int) -> tuple[int, list[dict]]:
        # Terminate, the candidates are already small enough,
        # there's no need to partition further
        if d <= p_min:
            return 1, []
        best: tuple[int, list[dict]] | None = None
        # "r" is the number of partitions
        # Needs to be bigger than K, but no more than half the fingerprint length (block_bits = 2)
        for r in range(K + 1, f // 2):
            # Bits for each block (rounded down, blocks can have different number of bits)
            block_bits = f // r
            # Possible combinations (without repetition) for this partition schema
            tables_comb = math.comb(r, K)
            # Recursive call to X
            sub_cost, sub_params = X(K * block_bits, d - ((r - K) * block_bits))
            cost = tables_comb * sub_cost
            params = [
                {"f": f, "d": d, "r": r, "block_bits": block_bits, "tables": tables_comb},
                *sub_params,
            ]
            if best is None or cost < best[0]:
                best = (cost, params)
        if best is None:
            raise ValueError(
                f"It's impossible to determine the optimal schema for f: {f} and d: {d}"
            )
        return best

    return X(f=f, d=d)


def build_tables(
    df: DataFrame, fingerprint_col: str, fingerprint_size: int, K: int, num_tables: int
) -> DataFrame:
    schema: List[int] = []

    # Splits fs into n parts evenly, putting larger blocks first
    def apply_table_schema(fs: int, n: int):
        base, remainder = divmod(fs, n)
        return [base + 1 if i < remainder else base for i in range(n)]

    # Split fingeprints into tokens, reading the schema from the container function
    @F.pandas_udf(ArrayType(BinaryType()))  # type: ignore
    def helper(fingerprints: pd.Series) -> pd.Series:
        # Aggregates all fingerprints into a single binary string,
        # unpacked as list of uint8 (each value is a bit)
        bits = np.unpackbits(np.frombuffer(b"".join(fingerprints.tolist()), np.uint8))
        # Reshape list of bits as a matrix, with *len(fingerprints)* rows
        # and sum(schema) columns
        bits = bits.reshape(len(fingerprints), sum(schema))

        cols, start = [], 0
        for partition_size in schema:
            # Create partition, padding bits if necessary
            packed = np.packbits(bits[:, start : start + partition_size], axis=1)
            # Transforms column into a list
            cols.append(packed.view(f"S{packed.shape[1]}").ravel().tolist())
            start += partition_size
        # Zip columns together, recomposing the original rows
        return pd.Series(map(list, zip(*cols)), index=fingerprints.index, dtype=object)  # type: ignore

    assert fingerprint_col in df.columns
    combos = list(itertools.combinations([i for i in range(num_tables)], num_tables - K))
    schema = apply_table_schema(fs=fingerprint_size, n=num_tables)
    df = df.withColumn("tokens", helper(F.col(fingerprint_col)))  # type: ignore
    return df.select(
        "*",
        F.posexplode(
            F.array(*[F.array(F.col("tokens")[a], F.col("tokens")[b]) for a, b in combos])
        ).alias("token_id", "token"),
    )


def near_duplicate_batch(
    corpus: DataFrame,
    query: DataFrame,
    corpus_fp_col: str,
    query_fp_col: str,
    corpus_id_col: str,
    query_id_col: str,
    fingerprint_size: int,
    K: int,
) -> DataFrame:
    corpus_size = corpus.count()
    """
    schema = estimate_optimal_schema(
        f=fingerprint_size, K=K, d=int(math.log(corpus_size, 2)), p_min=10
    )
    """
    # TODO: use other schema function
    schema = optimal_design(f=fingerprint_size, k=K, d=int(math.log(corpus_size, 2)), p_min=10)
    if schema[0] == 1:
        raise NotImplementedError(f"Need to do a cartesian product, the schema is {schema}")
    num_partitions = schema[1][0]["tau"]  # TODO: ugly, fix this
    corpus_tables = build_tables(
        df=corpus,
        fingerprint_col=corpus_fp_col,
        fingerprint_size=fingerprint_size,
        K=K,
        num_tables=num_partitions,
    )
    query_tables = build_tables(
        df=query,
        fingerprint_col=query_fp_col,
        fingerprint_size=fingerprint_size,
        K=K,
        num_tables=num_partitions,
    )  # .drop(query_fp_col)

    near_dups = (
        corpus_tables.alias("a")
        .join(
            query_tables.alias("b"),
            (F.col("a.token") == F.col("b.token")) & (F.col("a.token_id") == F.col("b.token_id")),
        )
        .where(F.col(f"a.{corpus_id_col}") < F.col(f"b.{query_id_col}"))
        .select(
            F.col(f"a.{corpus_id_col}").alias("id1"),
            F.col(f"b.{query_id_col}").alias("id2"),
            F.col(f"a.{corpus_fp_col}").alias("fingerprint1"),
            F.col(f"b.{query_fp_col}").alias("fingerprint2"),
        )
    )

    # print(corpus_tables.groupBy("token").count().sort(F.col("count").desc()).show())
    return near_duplicates_pairs(near_dups, f1="fingerprint1", f2="fingerprint2", K=K)


def optimal_design(
    f: int, k: int, d: int, p_min: int, *, nibble: bool = False
) -> tuple[int, list[dict]]:
    """
    Like optimal_table_count, but also returns the chosen decomposition:
    a list of levels [{f, d, tau, block_bits, tables_at_this_level}, ...]
    ordered outermost-first. Total tables = product of tables_at_level.

    Returns (total_tables, levels).
    """

    def _divisors_at_least(f: int, lo: int) -> list[int]:
        """All divisors of f that are >= lo."""
        return [t for t in range(lo, f + 1) if f % t == 0]

    @lru_cache(maxsize=None)
    def X(f: int, d: int) -> tuple[int, list[dict]]:
        if d <= p_min:
            return 1, []

        best: tuple[int, list[dict]] | None = None
        for tau in _divisors_at_least(f, k + 1):
            block = f // tau
            if nibble and block % 4 != 0:
                continue
            f_next, d_next = block * k, max(d - (tau - k) * block, 0)
            if f_next < 1:
                continue
            sub_cost, sub_levels = X(f_next, d_next)
            cost, levels = (
                math.comb(tau, k) * sub_cost,
                [
                    {"f": f, "d": d, "tau": tau, "block_bits": block, "tables": math.comb(tau, k)},
                    *sub_levels,
                ],
            )
            if best is None or cost < best[0]:
                best = (cost, levels)
        if best is None:
            raise ValueError(f"no valid decomposition for f={f}, k={k}")
        return best

    return X(f, d)


# Make function that chops fingerprints given a schema and posexplodes them

# Make function that prepares the fingerprint database, creating schema and tables, saving it on disk if it's not there already.

# Make function that prepares the query

# Make function that joins database and query, dropping duplicates and checking K distance.
