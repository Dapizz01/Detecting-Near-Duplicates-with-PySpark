from dataclasses import dataclass

import pyspark.sql.functions as F
from pyspark.sql import DataFrame


@dataclass
class Metrics:
    accuracy: float
    precision: float
    recall: float
    f1_score: float


def extract_metrics(df: DataFrame) -> Metrics:
    assert "detected_near_duplicate" in df.columns
    assert "is_near_duplicate" in df.columns
    bow_metrics = df.agg(
        F.sum(
            F.when(
                (F.col("is_near_duplicate") == F.lit(True))
                & (F.col("detected_near_duplicate") == F.lit(True)),
                1,
            ).otherwise(0)
        ).alias("tp"),
        F.sum(
            F.when(
                (F.col("is_near_duplicate") == F.lit(False))
                & (F.col("detected_near_duplicate") == F.lit(False)),
                1,
            ).otherwise(0)
        ).alias("tn"),
        F.sum(
            F.when(
                (F.col("is_near_duplicate") == F.lit(False))
                & (F.col("detected_near_duplicate") == F.lit(True)),
                1,
            ).otherwise(0)
        ).alias("fp"),
        F.sum(
            F.when(
                (F.col("is_near_duplicate") == F.lit(True))
                & (F.col("detected_near_duplicate") == F.lit(False)),
                1,
            ).otherwise(0)
        ).alias("fn"),
    ).collect()[0]

    tp = bow_metrics["tp"]
    tn = bow_metrics["tn"]
    fp = bow_metrics["fp"]
    fn = bow_metrics["fn"]

    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    f1 = 2 * (precision * recall) / (precision + recall)

    return Metrics(accuracy=accuracy, precision=precision, recall=recall, f1_score=f1)
