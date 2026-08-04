import pyspark.sql.functions as F
from pyspark.sql import DataFrame

from src.config import data_path
from src.spark_utils import get_spark


def read_qqp() -> DataFrame:
    """Reads the Quora Question Pair dataset as a PySpark DataFrame.

    Returns:
        DataFrame: The QQP dataset, with the schema ("ID": int, "data1": string, "data2": string)
    """
    spark = get_spark()
    df = spark.read.csv(
        str(data_path("quora_duplicate_questions.tsv")),
        header=True,
        sep="\\t",
        inferSchema=True,
    )
    df = df.drop("qid1", "qid2")
    return df.withColumnsRenamed({"id": "ID", "question1": "data1", "question2": "data2"})


def read_qqp_benchmark(benchmark_id: int):
    """Reads the Quora Question Pair benchmark dataset as a PySpark DataFrame.

    Args:
        benchmark_id (int): Which QQP benchmark dataset to choose (can be 1, 2, 3)

    Returns:
        DataFrame: The QQP dataset, with the schema ("ID": int, "is_near_duplicate_semantic": int, "is_near_duplicate_syntax": int, "data1": string, "data2": string)
    """
    assert benchmark_id in (1, 2, 3)
    spark = get_spark()
    df = spark.read.csv(
        str(data_path(f"qqp_benchmark_{benchmark_id}.csv")),
        header=True,
        sep=",",
        inferSchema=True,
    )
    df = df.withColumn(
        "is_near_duplicate_semantic",
        F.when(F.col("is_near_duplicate_semantic") == 1, True).otherwise(False),
    )
    df = df.withColumn(
        "is_near_duplicate_syntax",
        F.when(F.col("is_near_duplicate_syntax") == 1, True).otherwise(False),
    )
    return df.withColumnsRenamed({"id": "ID", "question1": "data1", "question2": "data2"})
