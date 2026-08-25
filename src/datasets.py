import pyspark.sql.functions as F
from pyspark.sql import DataFrame

from src.config import data_path
from src.spark_utils import get_spark


def read_synthetic() -> DataFrame:
    spark = get_spark()
    df = spark.read.csv(str(data_path("synthetic.csv")), header=True)
    df = df.withColumn("fingerprint", F.unhex(F.lpad(F.conv(F.col("fingerprint"), 2, 16), 16, "0")))
    return df


def read_c5() -> DataFrame:
    spark = get_spark()
    df = spark.read.csv(str(data_path("c5_ita.csv")), inferSchema=True, header=True)
    return df


def read_paws() -> DataFrame:
    spark = get_spark()
    df = spark.read.csv(str(data_path("paws.csv")), inferSchema=True, header=True)
    df = df.withColumn(
        "is_near_duplicate",
        F.when(F.col("label") == 1, F.lit(True)).otherwise(F.lit(False)),
    )
    df = df.drop("label")
    return df.withColumnsRenamed({"id": "ID", "sentence1": "data1", "sentence2": "data2"})


def read_atn_sampled() -> DataFrame:
    spark = get_spark()
    df = spark.read.csv(
        str(data_path("all-the-news-sampled.csv")),
        inferSchema=True,
    )
    return df.withColumnsRenamed({"id": "ID", "question1": "data1", "question2": "data2"})


def read_wikipedia(variant: str) -> DataFrame:
    spark = get_spark()
    assert variant in ("5percent", "10percent")
    df = spark.read.csv(
        str(data_path(f"wikipedia_near_duplicate_{variant}.csv")), inferSchema=True, header=True
    )
    df = df.withColumn(
        "is_dup",
        F.when(F.col("is_near_duplicate") == 1, F.lit(True)).otherwise(F.lit(False)),
    )
    df = df.drop("is_near_duplicate")
    return df.withColumnsRenamed(
        {"id": "ID", "text": "data1", "text2": "data2", "is_dup": "is_near_duplicate"}
    )


def read_qqp(n: int) -> DataFrame:
    """Reads the Quora Question Pair dataset as a PySpark DataFrame.

    Returns:
        DataFrame: The QQP dataset, with the schema ("ID": int, "data1": string, "data2": string)
    """
    spark = get_spark()
    df = (
        spark.read.csv(
            str(data_path("quora_duplicate_questions.tsv")),
            header=True,
            sep="\\t",
            inferSchema=True,
        )
        .dropna()
        .limit(n)
    )
    df = df.withColumn(
        "is_near_duplicate", F.when(F.col("is_duplicate") == 1, F.lit(True)).otherwise(F.lit(False))
    )
    df = df.drop("qid1", "qid2", "is_duplicate")
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


def read_feature_benchmark(which: str):
    # TODO: comments
    assert which in ("easy", "medium", "hard")
    spark = get_spark()
    df = spark.read.csv(
        str(data_path(f"feature_benchmark_{which}.csv")),
        header=True,
        sep=",",
        quote='"',
        escape='"',
        inferSchema=True,
    )
    df = df.withColumn(
        "is_near_duplicate_semantic",
        F.when(F.col("is_near_duplicate_semantic") == 1, True).otherwise(False),
    )
    if which != "hard":  # "Hard" dataset doesnt have syntax near-dup
        df = df.withColumn(
            "is_near_duplicate_syntax",
            F.when(F.col("is_near_duplicate_syntax") == 1, True).otherwise(False),
        )
    return df.withColumnsRenamed({"id": "ID"})
