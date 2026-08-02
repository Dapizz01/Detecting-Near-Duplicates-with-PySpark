from dataclasses import dataclass

import pyspark.sql.functions as F
from pyspark.sql import DataFrame

import src.feature_models.shingles as shingles

FINGERPRINT_SIZES = (32, 64, 128, 256, 512, 1024)


@dataclass
class FingerprintParams:
    shingle_size: int
    over_words: bool


def create_fingerprints(
    df: DataFrame, method: str, input_col: str, output_col: str, size: int, other: FingerprintParams
) -> DataFrame:
    assert method in ["bag_of_words", "embeddings", "shingles", "tf-idf"]
    assert input_col in df.columns
    assert size in FINGERPRINT_SIZES

    if method == "shingles":
        assert other is not None
        assert other.shingle_size < size

    df = df.filter(F.col(input_col).isNotNull())
    df = (  # Clean column, remove all symbols but a-z and \s
        df.withColumn("cleaned_text", F.regexp_replace(F.col(input_col), r"[\n\r\t]", ""))
        .withColumn("cleaned_text", F.lower(F.col(input_col)))
        .withColumn("cleaned_text", F.regexp_replace(F.col(input_col), r"[^a-z\s]", ""))
    )
    match method:
        case "bag_of_words":
            df = df
        case "shingles":
            df = shingles.make_fingerprints(
                df,
                input_col="cleaned_text",
                output_col=output_col,
                fingerprint_size=size,
                shingle_size=other.shingle_size,
                over_words=other.over_words,
            )
        case "tf-idf":
            df = df
        case "embeddings":  # TODO: will use input_text, not cleaned_text
            df = df
    return df
