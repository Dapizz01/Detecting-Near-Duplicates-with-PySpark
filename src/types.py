from dataclasses import dataclass
from typing import Literal

from pyspark.sql.types import FloatType, StructField, StructType

metrics_schema = StructType(
    [
        StructField("Accuracy", FloatType(), nullable=False),
        StructField("Precision", FloatType(), nullable=False),
        StructField("Recall", FloatType(), nullable=False),
        StructField("F1-score", FloatType(), nullable=False),
    ]
)


@dataclass
class Metrics:
    accuracy: float
    precision: float
    recall: float
    f1_score: float


_FINGERPRINT_SIZES = Literal[32, 64, 128, 256, 512]
"""Contains all of the possible fingerprint sizes. It must be a multiple of 2.
"""
# TODO: check if fingerprint_sizes must really be a multiple of 2.
