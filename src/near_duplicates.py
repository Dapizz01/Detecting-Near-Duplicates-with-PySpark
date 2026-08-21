import pandas as pd
import pyspark.sql.functions as F
from pyspark.sql import DataFrame
from pyspark.sql.types import IntegerType

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
    return df


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
        # K = 2 -> 3 blocks
        # K = 3 -> 4 blocks
        #
        if K < int(fingerprint_size / 4):
            df_chunks = [
                F.substring(F.col(df_fingerprint_col), i, 1)
                for i in range(1, int(fingerprint_size / 8) + 1)
            ]
            query_chunks = [
                F.substring(F.col(query_fingerprint_col), i, 1)
                for i in range(1, int(fingerprint_size / 8) + 1)
            ]

            df_join = df.select(
                id_col, df_fingerprint_col, F.posexplode(F.array(*df_chunks)).alias("pos", "chunk")
            )
            query_join = query.select(
                id_col,
                query_fingerprint_col,
                F.posexplode(F.array(*query_chunks)).alias("pos", "chunk"),
            )

            join_result = (
                df_join.alias("a")
                .join(F.broadcast(query_join).alias("b"), on=["pos", "chunk"])
                .select(
                    F.col(f"a.{df_fingerprint_col}").alias("f1"),
                    F.col(f"b.{query_fingerprint_col}").alias("f2"),
                )
                .drop_duplicates(["f1", "f2"])
            )

            join_result.show()
            nd_df = near_duplicates_pairs(join_result, f1="f1", f2="f2", K=K)

        else:  # Multi-Index Hashing
            pass

        pass

    return nd_df
