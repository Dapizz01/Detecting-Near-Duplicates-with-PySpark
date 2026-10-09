import bisect
import itertools
import math
from functools import lru_cache

import pandas as pd
import pyspark.sql.functions as F
from pyspark.sql import DataFrame
from pyspark.sql.types import BinaryType, IntegerType, Row, StringType, StructField, StructType

from src.config import SPARK_SHUFFLE_PARTITIONS
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


def partitions_schema(f: int, K: int, d: int, p_min: int) -> tuple[int, list[dict]]:
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


### DETECT_NEAR_DUPLICATE_NEW_V2 ###


def create_partition_map(entries: list[Row], schema: tuple) -> dict[tuple, list]:
    return {(1, 2, 3): [1, 2, 3]}


# Like detect_near_duplicate_batch, but with support for multiple tables.
def detect_near_duplicate_batch_v2(
    dataset: DataFrame,
    query: DataFrame,
    fingerprint_col: str,
    id_col: str,
    fingerprint_size: int,
    dataset_size: int,
    K: int,
    max_match_per_probe: int,
) -> DataFrame:
    raise NotImplementedError("This function is still under development!")
    assert (
        id_col != fingerprint_col
    ), f"[detect_near_duplicate_new_v2] Fingerprint column and ID column names are the same ({id_col})"
    assert (
        (id_col, fingerprint_col) in dataset.columns
    ), f"[detect_near_duplicate_new_v2] The specified column names ({id_col} and {fingerprint_col}) are not present in the dataset DataFrame schema ({query.columns})"
    assert (
        (id_col, fingerprint_col) in query.columns
    ), f"[detect_near_duplicate_new_v2] The specified column names ({id_col} and {fingerprint_col}) are not present in the query DataFrame schema ({query.columns})"
    assert (
        fingerprint_size in (32, 64, 128, 256, 512)
    ), f"[detect_near_duplicate_new_v2] Fingerprint size {fingerprint_size} is invalid; it must be (32, 64, 128, 256, 512)"
    assert (
        K <= fingerprint_size
    ), f"[detect_near_duplicate_new_v2] Parameter K ({K}) cannot be larger than the fingerprint size ({fingerprint_size})"
    assert dataset_size >= 0, "[detect_near_duplicate_new_v2] Dataset size must be positive."

    # Clean dataframes, taking only the needed columns
    query = query.select([id_col, fingerprint_col])
    dataset = dataset.select([id_col, fingerprint_col])

    spark = get_spark()
    schema = partitions_schema(
        f=fingerprint_size,
        K=K,
        d=int(math.log(dataset_size, 2)),
        p_min=int(math.log(max_match_per_probe, 2)),
    )

    if schema[0] == 1:
        print("[detect_near_duplicate_new_v2] Fallback to Cartesian product.")
        # TODO: make Cartesian product function
        ...
    query_partition_map = create_partition_map(entries=query.collect(), schema=schema)
    return dataset


### DETECT_NEAR_DUPLICATES_NEW ###


def get_blocks_masks(blocks_sizes: list[int]) -> list[tuple[int, int]]:
    """Translates a list of block sizes into a list of offsets and masks.
    Useful to split fingerprints into blocks.

    Args:
        blocks_sizes (list[int]): list of block sizes

    Returns:
        list[tuple[int, int]]: list of (offset, bit_mask)
    """
    shifts_masks = []
    shift = 0
    # Iterate reversed, it makes shift computation easier
    for length in reversed(blocks_sizes):
        # 1000000... - 1 -> 011111...
        shifts_masks.append((shift, (1 << length) - 1))
        shift += length
    # Reverse back to get increasing order
    shifts_masks.reverse()
    return shifts_masks


def get_partitions_blocks_sizes(
    blocks_sizes: list[int], combos: list[tuple[int, ...]]
) -> list[list[tuple[int, int]]]:
    """Returns all blocks (index and size) of all partitions combinations.

    Args:
        blocks_sizes (list[int]): list of all blocks sizes
        combos (list[tuple[int, ...]]): list of all partitions combinations

    Returns:
        list[list[tuple[int, int]]]: list of all blocks (index and size) of all partitions combinations.
    """
    return [[(idx, blocks_sizes[idx]) for idx in combo] for combo in combos]


def extract_blocks(fingerprint: int, masks: list[tuple[int, int]]) -> list[int]:
    """Extracts all blocks from a fingerprint, following the provided masks (obtained via get_blocks_masks)

    Args:
        fingerprint (int): a fingerprint
        shifts_masks (list[tuple[int, int]]): a list of masks, obtainable with get_blocks_masks(...)

    Returns:
        list[int]: ordered list of all blocks in the fingerprint
    """
    return [(fingerprint >> shift) & mask for shift, mask in masks]


def extract_partitions(blocks: list[int], combos: list[list[tuple[int, int]]]) -> list[int]:
    """Extracts all blocks of all partitions (listed in combos) of blocks.

    Args:
        blocks (list[int]): List of block values.
        combos (list[list[tuple[int, int]]]): All combos. Each combo is a list of (block_id, offset).

    Returns:
        list[int]: List of key blocks values
    """
    tokens = []
    for combo in combos:
        token = 0
        for idx, offset in combo:
            token = (token << offset) | blocks[idx]
        tokens.append(token)
    return tokens


def get_blocks_sizes(fingerprint_size: int, num_blocks: int) -> list[int]:
    base, remainder = divmod(fingerprint_size, num_blocks)
    return [base + 1 if i < remainder else base for i in range(num_blocks)]


def build_broadcast_tables(
    query: DataFrame, fingerprint_size: int, K: int, num_blocks: int
) -> tuple[dict, list[dict]]:
    assert (
        len(query.columns) == 2
    ), "The query dataframe must have schema: (ID: int, fingerprint: binary)"
    # Move the query DataFrame in the driver memory
    query_rows = query.collect()
    # Obtains sizes of each block (ex: [11, 11, 11, 11, 10, 10])
    blocks_sizes = get_blocks_sizes(fingerprint_size=fingerprint_size, num_blocks=num_blocks)
    # Computes all partition combinations of num_blocks - K blocks (pigeonhole)
    combos = list(itertools.combinations(range(num_blocks), num_blocks - K))
    query_reference = {}
    # List of all partitions.
    # A partition is represented as a dict, having as key all the values of the key blocks of that partition,
    # and as value a list of query rows IDs whose key block value is the key.
    # Ex: all_partitions[partition_idx][token_key] = [query_id1, query_id2, ...]
    all_partitions = [{} for _ in range(len(combos))]
    # Takes all masks to extract bits from blocks.
    blocks_masks = get_blocks_masks(blocks_sizes=blocks_sizes)
    partitions_blocks_sizes = get_partitions_blocks_sizes(blocks_sizes=blocks_sizes, combos=combos)

    for row in query_rows:
        query_id = row[0]
        # Parse the fingerprint from binary to integer (Python integers can be larger than 64 bits)
        query_fingerprint = int.from_bytes(row[1], "big")
        partitions = extract_partitions(
            extract_blocks(query_fingerprint, masks=blocks_masks),
            combos=partitions_blocks_sizes,
        )
        # Save the fingerprint for O(1) access during map phase
        query_reference[query_id] = query_fingerprint

        # For each block extracted, add it to partitions
        for partition_idx, partition in enumerate(partitions):
            if partition not in all_partitions[partition_idx]:
                all_partitions[partition_idx][partition] = []
            all_partitions[partition_idx][partition].append(query_id)
    return query_reference, all_partitions


def detect_near_duplicate_batch(
    corpus: DataFrame, query: DataFrame, fingerprint_size: int, K: int, d: int, p_min: int
) -> DataFrame:
    spark = get_spark()
    corpus = corpus.select(["ID", "fingerprint"])
    query = query.select(["ID", "fingerprint"])
    # TODO: implement multiple tables
    schema = partitions_schema(f=fingerprint_size, K=K, d=d, p_min=p_min)

    # No partitions needed - Cartesian product is performed
    if schema[0] == 1:
        cartesian = corpus.alias("a").crossJoin(query.alias("b"))
        cartesian = cartesian.withColumn(
            "hamming_distance", hamming_distance_bin(F.col("a.fingerprint"), F.col("b.fingerprint"))
        )
        cartesian = cartesian.filter(F.col("hamming_distance") <= K)
        return cartesian

    num_tables = schema[1][0]["r"]
    # query_reference is a dict that allows O(1) reading when comparing fingerprints.
    query_reference, tables = build_broadcast_tables(
        query=query, fingerprint_size=fingerprint_size, K=K, num_blocks=num_tables
    )
    blocks_sizes = get_blocks_sizes(fingerprint_size=fingerprint_size, num_blocks=num_tables)
    combos = list(itertools.combinations(range(num_tables), num_tables - K))
    blocks_masks = get_blocks_masks(blocks_sizes=blocks_sizes)
    partitions_blocks_sizes = get_partitions_blocks_sizes(blocks_sizes=blocks_sizes, combos=combos)

    # Broadcast to the executors all the query fingerprints and the partition tables.
    # The size of query is supposed to be small enough to be broadcasted (max 10MB, configurable)
    broadcast_refs = spark.sparkContext.broadcast(query_reference)
    broadcast_tables = spark.sparkContext.broadcast(tables)

    def probe_corpus(iterator):
        # Access broadcasted values
        query_refs = broadcast_refs.value
        query_tables = broadcast_tables.value

        for row in iterator:
            corpus_id = row[0]
            corpus_fingerprint = int.from_bytes(row[1], "big")

            corpus_partitions = extract_partitions(
                extract_blocks(corpus_fingerprint, masks=blocks_masks),
                combos=partitions_blocks_sizes,
            )

            # Keeps track of past matches - prevents multiple hits for the same query ID.
            seen_query_ids = set()

            for partition_idx, partition in enumerate(corpus_partitions):
                # Check if there's a matching partition value in query, with the same partition_idx
                candidates = query_tables[partition_idx].get(partition)
                # No match, skip.
                if not candidates:
                    continue

                # Check all candidates
                for query_id in candidates:
                    # Prevent evaluating the same pair multiple times.
                    if query_id in seen_query_ids:
                        continue
                    seen_query_ids.add(query_id)

                    # Fetch query fingerprint from dict and XOR them.
                    query_fingerprint = query_refs[query_id]
                    if (corpus_fingerprint ^ query_fingerprint).bit_count() <= K:
                        yield (corpus_id, query_id)

    # Maps every row of corpus into a (corpus_id, query_id) pair, executing probe_corpus.
    # This could be further optimized with mapInPandas, but it's fine for small/medium sized datasets.
    result_rdd = corpus.select("ID", "fingerprint").rdd.mapPartitions(probe_corpus)
    corpus_id_type = corpus.schema["ID"].dataType
    query_id_type = query.schema["ID"].dataType

    schema = StructType(
        [
            StructField("ID_corpus", corpus_id_type, False),
            StructField("ID_query", query_id_type, False),
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
        get_blocks_sizes(fingerprint_size=fingerprint_size, num_blocks=schema[1][0]["r"])
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
