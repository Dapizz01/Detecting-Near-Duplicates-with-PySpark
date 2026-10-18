from dataclasses import dataclass

import pyspark.sql.functions as F
from pyspark.sql import DataFrame


@dataclass
class Metrics:
    accuracy: float  # How many near-duplicates detections were correct?
    precision: float  # How many near-duplicates detected as such were really near-duplicates?
    recall: float  # How many true near-duplicates were detected?
    f1_score: float  # Combined score of previous metrics.


@dataclass
class ModelMetrics:
    model: str
    K: int
    fingerprint_size: int
    shingle_size: int
    notes: str
    metrics: Metrics


def extract_metrics(df: DataFrame, test_col: str, truth_col: str) -> Metrics:
    assert test_col in df.columns, (
        f"[extract_metrics] The test column ({test_col}) isn't present in the DataFrame schema ({df.columns})."
    )
    assert truth_col in df.columns, (
        f"[extract_metrics] The truth column ({truth_col}) isn't present in the DataFrame schema ({df.columns})"
    )

    bow_metrics = df.agg(
        F.sum(
            F.when(
                (F.col(truth_col) == F.lit(True)) & (F.col(test_col) == F.lit(True)),
                1,
            ).otherwise(0)
        ).alias("tp"),
        F.sum(
            F.when(
                (F.col(truth_col) == F.lit(False)) & (F.col(test_col) == F.lit(False)),
                1,
            ).otherwise(0)
        ).alias("tn"),
        F.sum(
            F.when(
                (F.col(truth_col) == F.lit(False)) & (F.col(test_col) == F.lit(True)),
                1,
            ).otherwise(0)
        ).alias("fp"),
        F.sum(
            F.when(
                (F.col(truth_col) == F.lit(True)) & (F.col(test_col) == F.lit(False)),
                1,
            ).otherwise(0)
        ).alias("fn"),
    ).collect()[0]

    tp = bow_metrics["tp"]  # True positives
    tn = bow_metrics["tn"]  # True negatives
    fp = bow_metrics["fp"]  # False positives
    fn = bow_metrics["fn"]  # False negatives

    precision = tp / (tp + fp) if tp + fp != 0 else 0
    recall = tp / (tp + fn) if tp + fn != 0 else 0
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    f1 = 2 * (precision * recall) / (precision + recall) if precision + recall != 0 else 0

    return Metrics(accuracy=accuracy, precision=precision, recall=recall, f1_score=f1)
