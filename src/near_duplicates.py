import bisect
import itertools
import math
from collections import defaultdict
from functools import lru_cache

import pandas as pd
import pyspark.sql.functions as F
from pyspark.sql import DataFrame
from pyspark.sql.types import BinaryType, IntegerType, StringType, StructField, StructType

from src.config import FINGERPRINT_SIZES, SPARK_SHUFFLE_PARTITIONS
from src.metrics import Metrics, extract_metrics
from src.spark_utils import get_spark


@F.pandas_udf(IntegerType())  # type: ignore
def hamming_distance_bin(col1: pd.Series, col2: pd.Series) -> pd.Series:
    def hamming(b1, b2):
        if b1 is None or b2 is None or pd.isna(b1) or pd.isna(b2):
            return None
        return (int.from_bytes(b1, "big") ^ int.from_bytes(b2, "big")).bit_count()

    return pd.Series([hamming(b1, b2) for b1, b2 in zip(col1, col2)], dtype="Int64")


def detect_near_duplicates_pairs(df: DataFrame, f1: str, f2: str, K: int) -> DataFrame:
    assert f1 in df.columns
    assert f2 in df.columns
    df = df.withColumn(
        "hamming_distance",
        hamming_distance_bin(F.col(f1), F.col(f2)),  # type: ignore
    ).filter(F.col("hamming_distance") <= K)
    return df  # .drop("hamming_distance")


def detect_near_duplicates_benchmark(
    df: DataFrame, f1_col: str, f2_col: str, truth_col: str, K: int
) -> Metrics:
    """Benchmarks feature models, by comparing already established pairs of data.

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

    # df = df.drop("hamming_distance")
    return extract_metrics(df, test_col="computed_near_duplicate", truth_col=truth_col)


### Batch Near-Duplicate Detection ###


class FingerprintPartitioner:
    """
    Contains the logic for splitting a fingerprint into blocks.
    Both dataset and query must use the same partitioner.
    """

    def __init__(self, fingerprint_size: int, num_partitions: int, K: int):
        self.fingerprint_size = fingerprint_size
        self.num_partitions = num_partitions

        # Determine partitions sizes, distributing remainder evenly
        base, remainder = divmod(fingerprint_size, num_partitions)
        self.partitions_sizes = [base + 1 if i < remainder else base for i in range(num_partitions)]

        # Compute bit shifts and masks for O(1) look-up using dicts
        self.shifts_masks = []
        shift = 0
        for length in reversed(self.partitions_sizes):
            # 1000... - 1 = 0111...
            self.shifts_masks.append((shift, (1 << length) - 1))
            shift += length
        self.shifts_masks.reverse()

        # 3. Precompute combinations of partitions
        self.combos = list(itertools.combinations(range(num_partitions), num_partitions - K))
        self.partitions_blocks = [
            [(idx, self.partitions_sizes[idx]) for idx in combo] for combo in self.combos
        ]

    def extract_partitions(self, fingerprint: int) -> list[int]:
        """Extracts all combined partition tokens for a given fingerprint."""
        # Extract individual blocks
        blocks = [(fingerprint >> shift) & mask for shift, mask in self.shifts_masks]

        # Combine blocks into partition keys
        tokens = []
        for combo in self.partitions_blocks:
            token = 0
            for idx, offset in combo:
                token = (token << offset) | blocks[idx]
            tokens.append(token)
        return tokens


def partitions_schema(f: int, K: int, d: int, p_min: int, depth: int) -> tuple[int, list[dict]]:
    """Estimates the optimal tables schema using the equation from the original paper."""

    @lru_cache(maxsize=None)
    def _search_schema(f_curr: int, d_curr: int, depth: int) -> tuple[int, list[dict]]:
        if d_curr <= p_min or depth == 0:
            return 1, []

        best = None
        for r in range(K + 1, (f_curr // 2) + 1):
            block_bits = f_curr // r
            tables_comb = math.comb(r, K)

            sub_cost, sub_params = _search_schema(
                K * block_bits, d_curr - ((r - K) * block_bits), depth - 1
            )
            cost = tables_comb * sub_cost
            params = [
                {
                    "f": f_curr,
                    "d": d_curr,
                    "r": r,
                    "block_bits": block_bits,
                    "tables": tables_comb,
                },
                *sub_params,
            ]

            if best is None or cost < best[0]:
                best = (cost, params)

        return best if best is not None else (1, [])

    return _search_schema(f, d, depth)


def build_broadcast_tables(
    query: DataFrame, partitioner: FingerprintPartitioner
) -> tuple[dict, list[dict]]:
    """Builds the hash tables for query."""

    # Query dataset collected in the driver
    query_rows = query.collect()
    query_reference = {}

    # Initialize dict for every partition combo
    all_partitions = [defaultdict(list) for _ in range(len(partitioner.combos))]

    for row in query_rows:
        query_id = row[0]
        fingerprint = int.from_bytes(row[1], "big")

        # Save fingerprint for O(1) lookup
        query_reference[query_id] = fingerprint

        # Partition index and partition value are the keys
        for p_idx, partition_val in enumerate(partitioner.extract_partitions(fingerprint)):
            # No base case due to defaultdict
            all_partitions[p_idx][partition_val].append(query_id)

    # Convert defaultdicts back to standard dicts
    return query_reference, [dict(p) for p in all_partitions]


def detect_near_duplicate_batch(
    dataset: DataFrame,
    query: DataFrame,
    fingerprint_size: int,
    K: int,
    d: int,
    p_min: int,
    rec_depth: int = 1,
) -> DataFrame:

    assert fingerprint_size in FINGERPRINT_SIZES, (
        f"[detect_near_duplicate_batch] The specified fingerprint size ({fingerprint_size}) is not valid: it must be one of ({FINGERPRINT_SIZES})"
    )
    assert K > 0 and K <= fingerprint_size, (
        f"[detect_near_duplicate_batch] The selected K value ({K}) is invalid."
    )
    assert d > 0, f"[detect_near_duplicate_batch] The chosen d value ({d}) must be positive."
    assert p_min > 0, (
        f"[detect_near_duplicate_batch] The chosen p_min value ({p_min}) must be positive."
    )
    assert ("ID, fingerprint") in dataset.columns, (
        "[detect_near_duplicate_batch] The dataset DataFrame schema is invalid: it must contain columns 'ID' and 'fingerprint'"
    )
    assert ("ID, fingerprint") in query.columns, (
        "[detect_near_duplicate_batch] The query DataFrame schema is invalid: it must contain columns 'ID' and 'fingerprint'"
    )

    if rec_depth != 1:
        raise NotImplementedError(
            "[detect_near_duplicate_batch] The chosen rec_depth value ({rec_depth}) is invalid: it must be 1."
            "Unfortunately, it's tricky to implement deeper recursions without fixing other parameters, since fingerprint size become uneven in the dataset."
            "This is deemed as out of scope for this project."
        )

    spark = get_spark()

    schema_cost, schema_params = partitions_schema(
        f=fingerprint_size, K=K, d=d, p_min=p_min, depth=rec_depth
    )

    # There's no need to partition, perform cartesian product
    if schema_cost == 1:
        cartesian = dataset.alias("a").crossJoin(query.alias("b"))
        return cartesian.withColumn(
            "hamming_distance",
            hamming_distance_bin(F.col("a.fingerprint"), F.col("b.fingerprint")),  # type: ignore
        ).filter(F.col("hamming_distance") <= K)

    # For the reason explained above, only the first-layer schema is considered.
    num_tables = schema_params[0]["r"]
    partitioner = FingerprintPartitioner(fingerprint_size, num_tables, K)

    # Build query broadcast tables (look at build_broadcast_tables for details)
    # using the created partitioner.
    query_reference, tables = build_broadcast_tables(query, partitioner)

    # Broadcast query refs and tables.
    # refs uses query IDs as key and query fingerprints as values.
    # tables uses query IDs and query partition as key, returning query IDs with matching partitions.
    broadcast_refs = spark.sparkContext.broadcast(query_reference)
    broadcast_tables = spark.sparkContext.broadcast(tables)

    def probe_corpus(iterator):
        # Load broadcasted values
        query_refs = broadcast_refs.value
        query_tables = broadcast_tables.value

        # For each row of worker dataset partition
        for row in iterator:
            corpus_id = row[0]
            corpus_fingerprint = int.from_bytes(row[1], "big")

            # Keep history of seen query IDs: they will be skipped if encountered again.
            seen_query_ids = set()

            corpus_partitions = partitioner.extract_partitions(corpus_fingerprint)

            for p_idx, partition_val in enumerate(corpus_partitions):
                # Get all query IDs whose partition value matches current dataset partition value (and index).
                candidates = query_tables[p_idx].get(partition_val)

                if not candidates:
                    continue

                for query_id in candidates:
                    if query_id in seen_query_ids:
                        continue
                    seen_query_ids.add(query_id)

                    # Compute Hamming distance and compare it to K: if lower, yield it.
                    # Otherwise, discard it and go on.
                    query_fingerprint = query_refs[query_id]
                    hamming_distance = (corpus_fingerprint ^ query_fingerprint).bit_count()
                    if hamming_distance <= K:
                        yield (corpus_id, query_id, hamming_distance)

    # Map all workers dataset partitions (NOTE: not to be confused with dataset logical partitions)
    # to find near-duplicates.
    # This ensures no shuffle is performed.
    result_rdd = dataset.select("ID", "fingerprint").rdd.mapPartitions(probe_corpus)

    schema = StructType(
        [
            StructField("ID_dataset", dataset.schema["ID"].dataType, False),
            StructField("ID_query", query.schema["ID"].dataType, False),
            StructField("hamming_distance", IntegerType(), False),
        ]
    )

    return spark.createDataFrame(result_rdd, schema=schema)


### LEGACY METHODS ###


def get_partitions_index_offset(lengths: list[int]) -> list[tuple[int, int]]:
    raise NotImplementedError(
        "[binary_search_method] Legacy support method used by detect_near_duplicates_alternative."
    )
    base = 0
    result = []
    for length in lengths:
        result.append((base, length))
        base += length
    return result


def extract_bits(fingerprint: bytes, masks: list[int], schema: list[tuple[int, int]]) -> list[int]:
    raise NotImplementedError(
        "[binary_search_method] Legacy support method used by detect_near_duplicates_alternative."
    )
    fp = int.from_bytes(fingerprint, "big")
    return [(fp & mask) >> part[0] for mask, part in zip(masks, schema)]


def binary_search_match(record):
    raise NotImplementedError(
        "[binary_search_method] Legacy support method used by detect_near_duplicates_alternative."
    )
    # The RDD must have been previously cogrouped
    # (all query and corpus rows with the same partition_ID are on the same worker)
    _, (query_iter, corpus_iter) = record

    # Load corpus iterator in worker memory,
    # and sort based on the partition value
    corpus_list = list(corpus_iter)
    corpus_list.sort(key=lambda x: x[0])

    # Extract all corpus partition values for binary search
    corpus_keys = [x[0] for x in corpus_list]
    results = []

    # Iterate over all query rows
    for q_part_val, q_payload in query_iter:
        # Binary search for the matching corpus partition value
        idx = bisect.bisect_left(corpus_keys, q_part_val)

        # If the query partition key is not found in the corpus, bisect will returns
        # the index where "it should be".
        # Therefore, it's necessary to check the value indexed by bisect.
        while idx < len(corpus_keys) and corpus_keys[idx] == q_part_val:
            # Extract all corpus rows with same partition value.
            _, c_payload = corpus_list[idx]

            # Unpack the nested [(ID, fingerprint)] lists
            for q_id, q_val in q_payload:
                for c_id, c_val in c_payload:
                    results.append((q_id, q_val, c_id, c_val))

            # bisect_left returns the leftest element with matching key,
            # the next element may also have a matching key.
            idx += 1

    return results


def hamming_distance(val1: bytes, val2: bytes, K: int) -> bool:
    return (int.from_bytes(val1, "big") ^ int.from_bytes(val2, "big")).bit_count() <= K


# Assumes that corpus and query have the same id_col and fingerprint_col
def detect_near_duplicate_alternative(
    corpus: DataFrame,
    query: DataFrame,
    fingerprint_size: int,
    K: int,
    id_col: str,
    fingerprint_col: str,
    p_min: int,
    corpus_size: int,
) -> DataFrame:
    raise NotImplementedError(
        "This is an experimental method, much less efficient compared to the already implemented method. \n"
        "This method strives to lower the computational complexity of detect_near_duplicate_batch, while keeping the same shuffle cost. \n"
        "It's possible by using binary search on the sorted dataset, for each query partition entry. \n"
        "However, in PySpark it's not feasable: loading a sorted DataFrame does not guaranteed that the data each worker receives is sorted. \n"
        "Of course, sorting the whole dataset is very expensive too. \n"
        "If one wishes, uncomment this raised exception and move forward."
    )
    assert id_col, fingerprint_col in corpus.columns
    assert id_col, fingerprint_col in query.columns
    assert K < fingerprint_size

    corpus_log_size = int(math.log(corpus_size, 2))
    schema = partitions_schema(f=fingerprint_size, K=K, d=corpus_log_size, p_min=p_min)
    if schema[0] == 1:
        print("[detect_near_duplicate] Fallback to Cartesian product.")
        # TODO: doesn't seem to work, must be fixed and put in a separate function
        result = (
            corpus.alias("a")
            .crossJoin(query.alias("b"))
            .filter(
                hamming_distance_bin(F.col(f"a.{fingerprint_col}"), F.col(f"b.{fingerprint_col}"))  # type: ignore
            )
            .withColumnsRenamed(
                {
                    f"b.{id_col}": "query_ID",
                    f"b.{fingerprint_col}": "query_fingerprint",
                    f"a.{id_col}": "corpus_ID",
                    f"a.{fingerprint_col}": "corpus_fingerprint",
                }
            )
        )
        return result

    partitions_lengths = get_partitions_index_offset(
        get_blocks_sizes(fingerprint_size=fingerprint_size, num_blocks=schema[1][0]["r"])  # noqa: F821
    )
    masks = [((1 << offset) - 1) << index for index, offset in partitions_lengths]

    corpus = corpus.select(id_col, fingerprint_col)
    query = query.select(id_col, fingerprint_col)
    corpus_rdd = corpus.rdd
    query_rdd = query.rdd

    corpus_partitions = (
        corpus_rdd.flatMap(
            lambda row: [
                (partition_idx, partition, row[0], row[1])
                for partition_idx, partition in enumerate(
                    extract_bits(row[1], masks, partitions_lengths)
                )
            ]
        )
        # [(p_ID, p, fp_ID, fp), ...]
        .groupBy(lambda row: (row[0], row[1]))
        # [((p_ID, p), [(p_ID, p, fp_ID, fp), (p_ID, p, fp_ID, fp), ...]), ...]
        .mapValues(lambda group: [item[2:] for item in group])
        # [((p_ID, p), [(fp_ID, fp), (fp_ID, fp), ...]), ...]
        .map(lambda x: (x[0][0], (x[0][1], x[1])))
        .partitionBy(numPartitions=int(SPARK_SHUFFLE_PARTITIONS))
        # [(p_ID, (query_iterator, corpus_iterator)), ...]
    )
    query_partitions = (
        query_rdd.flatMap(
            lambda row: [
                (partition_idx, partition, row[0], row[1])
                for partition_idx, partition in enumerate(
                    extract_bits(row[1], masks, partitions_lengths)
                )
            ]
        )
        # [(p_ID, p, fp_ID, fp), ...]
        .groupBy(lambda row: (row[0], row[1]))
        # [((p_ID, p), [(p_ID, p, fp_ID, fp), (p_ID, p, fp_ID, fp), ...]), ...]
        .mapValues(lambda group: [item[2:] for item in group])
        # [((p_ID, p), [(fp_ID, fp), (fp_ID, fp), ...]), ...]
        .map(lambda x: (x[0][0], (x[0][1], x[1])))
        .partitionBy(numPartitions=int(SPARK_SHUFFLE_PARTITIONS))
        # [(p_ID, (query_iterator, corpus_iterator)), ...]
    )
    grouped_rdd = query_partitions.cogroup(corpus_partitions)
    result = (
        grouped_rdd.flatMap(binary_search_match)
        .distinct()
        .filter(lambda row: hamming_distance(row[1], row[3], K))
    )
    result_schema = StructType(
        [
            StructField("query_ID", StringType(), True),
            StructField("query_fingerprint", BinaryType(), True),
            StructField("corpus_ID", StringType(), True),
            StructField("corpus_fingerprint", BinaryType(), True),
        ]
    )
    return result.toDF(result_schema)
