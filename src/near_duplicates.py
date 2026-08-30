import itertools
import math
from functools import lru_cache
from typing import List

import numpy as np
import pandas as pd
import pyspark.sql.functions as F
from pyspark.sql import DataFrame
from pyspark.sql.types import ArrayType, BinaryType, IntegerType, StructField, StructType

from src.metrics import Metrics, extract_metrics
from src.spark_utils import get_spark


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
    return df  # .drop("hamming_distance")


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


def tables_schema(f: int, K: int, d: int, p_min: int) -> tuple[int, list[dict]]:
    """Estimates the optimal tables schema to find near-duplicates.
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
        for r in range(K + 1, (f // 2) + 1):
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
            return 1, []
        return best

    return X(f=f, d=d)


def get_block_shifts_masks(tables_sizes: list[int]) -> list[tuple[int, int]]:
    shifts_masks = []
    shift = 0
    for length in reversed(tables_sizes):
        shifts_masks.append((shift, (1 << length) - 1))
        shift += length
    shifts_masks.reverse()
    return shifts_masks


def get_combo_plans(tables_sizes, combos) -> list[list[tuple[int, int]]]:
    return [[(idx, tables_sizes[idx]) for idx in combo] for combo in combos]


def extract_blocks(fingerprint: int, shifts_masks) -> list[int]:
    return [(fingerprint >> shift) & mask for shift, mask in shifts_masks]


def extract_tokens_fast(blocks, combo_plans) -> list[int]:
    tokens = []
    for plan in combo_plans:
        token = 0
        for idx, size in plan:
            token = (token << size) | blocks[idx]
        tokens.append(token)
    return tokens


def get_tables_sizes(fingerprint_size: int, num_tables: int) -> list[int]:
    base, remainder = divmod(fingerprint_size, num_tables)
    return [base + 1 if i < remainder else base for i in range(num_tables)]


def extract_tokens(fingerprint: int, tables_sizes: list[int], combos: list[tuple]) -> list[int]:
    blocks = []
    current_fingerprint = fingerprint

    for length in reversed(tables_sizes):
        mask = (1 << length) - 1
        blocks.insert(0, current_fingerprint & mask)
        current_fingerprint >>= length

    tokens = []
    for combo in combos:
        token = 0
        for idx in combo:
            token = (token << tables_sizes[idx]) | blocks[idx]
        tokens.append(token)
    return tokens


def build_broadcast_tables(
    query: DataFrame, fingerprint_size: int, K: int, num_tables: int
) -> tuple[dict, dict]:
    # Add select for id/fingerprint columns
    query_rows = query.collect()
    tables_sizes = get_tables_sizes(fingerprint_size=fingerprint_size, num_tables=num_tables)
    combos = list(itertools.combinations(range(num_tables), num_tables - K))
    query_reference = {}
    tables = [{} for _ in range(len(combos))]
    shifts_masks = get_block_shifts_masks(tables_sizes=tables_sizes)
    combo_plans = get_combo_plans(tables_sizes=tables_sizes, combos=combos)

    for row in query_rows:
        query_id = row[0]
        query_fingerprint = int.from_bytes(row[1], "big")
        tokens = extract_tokens_fast(
            extract_blocks(query_fingerprint, shifts_masks=shifts_masks),
            combo_plans=combo_plans,
        )
        # TODO: in order to implement nested schemas, it's necessary to just save
        # the part of query_fingerprint that is yet to check
        query_reference[query_id] = query_fingerprint
        tokens = extract_tokens(query_fingerprint, tables_sizes=tables_sizes, combos=combos)

        for table_idx, token in enumerate(tokens):
            if token not in tables[table_idx]:
                tables[table_idx][token] = []
            tables[table_idx][token].append(query_id)
    return query_reference, tables


def detect_near_duplicate_batch(
    corpus: DataFrame, query: DataFrame, fingerprint_size: int, K: int
) -> DataFrame:
    spark = get_spark()
    # TODO: implement this better
    num_tables = tables_schema(f=fingerprint_size, K=K, d=20, p_min=10)[1][0]["r"]
    query_reference, tables = build_broadcast_tables(
        query=query, fingerprint_size=fingerprint_size, K=K, num_tables=num_tables
    )
    broadcast_refs = spark.sparkContext.broadcast(query_reference)
    broadcast_tables = spark.sparkContext.broadcast(tables)
    schema_blocks = get_tables_sizes(fingerprint_size=fingerprint_size, num_tables=num_tables)
    combos = list(itertools.combinations(range(num_tables), num_tables - K))
    shifts_masks = get_block_shifts_masks(schema_blocks)
    combo_plans = get_combo_plans(schema_blocks, combos)

    def probe_corpus(iterator):
        refs = broadcast_refs.value
        tables = broadcast_tables.value

        for row in iterator:
            corpus_id = row[0]
            corpus_fingerprint = int.from_bytes(row[1], "big")

            tokens = extract_tokens_fast(
                extract_blocks(corpus_fingerprint, shifts_masks=shifts_masks),
                combo_plans=combo_plans,
            )
            seen_query_ids = set()

            for table_idx, token in enumerate(tokens):
                # candidates = tables[table_idx].get(token, [])
                candidates = tables[table_idx].get(token)
                if not candidates:
                    continue

                for query_id in candidates:
                    # Prevent evaluating the same pair multiple times
                    if query_id in seen_query_ids:
                        continue
                    seen_query_ids.add(query_id)

                    # Fetch heavy fingerprint from flat map and evaluate native XOR
                    query_fingerprint = refs[query_id]
                    if (corpus_fingerprint ^ query_fingerprint).bit_count() <= K:
                        yield (corpus_id, query_id)

    result_rdd = corpus.select("ID", "fingerprint").rdd.mapPartitions(probe_corpus)
    corpus_id_type = corpus.schema["ID"].dataType
    query_id_type = query.schema["ID"].dataType

    schema = StructType(
        [StructField("ID", corpus_id_type, False), StructField("ID", query_id_type, False)]
    )

    return spark.createDataFrame(result_rdd, schema=schema)


def build_partitions_dataframe(
    df: DataFrame, fingerprint_col: str, fingerprint_size: int, K: int, num_tables: int
) -> DataFrame:
    schema: List[int] = []

    # Splits fs into n parts evenly, putting larger blocks first
    def apply_partition_schema(fs: int, n: int):
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
    schema = apply_partition_schema(fs=fingerprint_size, n=num_tables)
    df = df.withColumn("tokens", helper(F.col(fingerprint_col)))  # type: ignore
    return df.select(
        "*",
        F.posexplode(
            F.array(*[F.array(*[F.col("tokens")[i] for i in combo]) for combo in combos])
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
    # TODO: use other schema function
    schema = optimal_design(f=fingerprint_size, k=K, d=int(math.log(corpus_size, 2)), p_min=10)
    if schema[0] == 1:
        # It's not possible to partition the fingerprints, and a cartesian product might be too expensive
        raise NotImplementedError(f"Need to do a cartesian product, the schema is {schema}")
    num_partitions = schema[1][0]["tau"]  # TODO: ugly, fix this
    corpus_exploded = build_partitions_dataframe(
        df=corpus,
        fingerprint_col=corpus_fp_col,
        fingerprint_size=fingerprint_size,
        K=K,
        num_tables=num_partitions,
    )
    query_exploded = build_partitions_dataframe(
        df=query,
        fingerprint_col=query_fp_col,
        fingerprint_size=fingerprint_size,
        K=K,
        num_tables=num_partitions,
    )

    # Join corpus and query, finding common tokens
    # (both token and token_id must be equal)
    near_dups = (
        corpus_exploded.alias("a")
        .join(
            query_exploded.alias("b"),
            (F.col("a.token") == F.col("b.token")) & (F.col("a.token_id") == F.col("b.token_id")),
        )
        # Prevents duplicates
        .where(F.col(f"a.{corpus_id_col}") < F.col(f"b.{query_id_col}"))
        .select(
            F.col(f"a.{corpus_id_col}").alias("id1"),
            F.col(f"b.{query_id_col}").alias("id2"),
            F.col(f"a.{corpus_fp_col}").alias("fingerprint1"),
            F.col(f"b.{query_fp_col}").alias("fingerprint2"),
        )
    )

    # print(corpus_tables.groupBy("token").count().sort(F.col("count").desc()).show())
    return near_duplicates_pairs(near_dups, f1="fingerprint1", f2="fingerprint2", K=K).distinct()


# TODO: for now, this function can't make nested partitions. Is this fine, or should this be fixed?
# Either way, it needs to be pointed out.
def near_duplicate_mirror(
    df: DataFrame,
    df_fp_col: str,
    df_id_col: str,
    fingerprint_size: int,
    K: int,
    df_size: int,
) -> DataFrame:
    # TODO: use other schema function
    schema = optimal_design(f=fingerprint_size, k=K, d=int(math.log(df_size, 2)), p_min=10)
    if schema[0] == 1:
        # It's not possible to partition the fingerprints, and a cartesian product might be too expensive
        raise NotImplementedError(f"Need to do a cartesian product, the schema is {schema}")
    num_partitions = schema[1][0]["tau"]  # TODO: ugly, fix this
    df_exploded = build_partitions_dataframe(
        df=df,
        fingerprint_col=df_fp_col,
        fingerprint_size=fingerprint_size,
        K=K,
        num_tables=num_partitions,
    )
    # df_test = df_exploded.select(df_id_col, "token_id", "token")
    # df_exploded = df_exploded.cache()

    # Join corpus and query, finding common tokens
    # (both token and token_id must be equal)
    near_dups = (
        df_exploded.alias("a")
        .join(
            df_exploded.alias("b"),
            (F.col("a.token") == F.col("b.token")) & (F.col("a.token_id") == F.col("b.token_id")),
        )
        # Prevents duplicates
        .where(F.col(f"a.{df_id_col}") < F.col(f"b.{df_id_col}"))
        .select(
            F.col(f"a.{df_id_col}").alias("id1"),
            F.col(f"b.{df_id_col}").alias("id2"),
            F.col(f"a.{df_fp_col}").alias("fingerprint1"),
            F.col(f"b.{df_fp_col}").alias("fingerprint2"),
        )
        .drop_duplicates(["id1", "id2"])
        # .distinct()
    )

    # fp1 = df.select(F.col(df_id_col).alias("id1"), F.col(df_fp_col).alias("fingerprint1"))
    # fp2 = df.select(F.col(df_id_col).alias("id2"), F.col(df_fp_col).alias("fingerprint2"))

    # print(corpus_tables.groupBy("token").count().sort(F.col("count").desc()).show())
    result = near_duplicates_pairs(near_dups, f1="fingerprint1", f2="fingerprint2", K=K)
    return result


# Make function that chops fingerprints given a schema and posexplodes them

# Make function that prepares the fingerprint database, creating schema and tables, saving it on disk if it's not there already.

# Make function that prepares the query

# Make function that joins database and query, dropping duplicates and checking K distance.
