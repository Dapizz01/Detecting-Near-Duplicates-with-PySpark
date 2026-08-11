from dataclasses import dataclass

import pyspark.sql.functions as F
from pyspark.sql import DataFrame

from src.feature_models import bag_of_words, embeddings, shingles, tf_idf

FINGERPRINT_SIZES = (32, 64, 128, 256, 512, 1024)


@dataclass
class FingerprintParams:
    shingle_size: int
    over_words: bool


def create_fingerprints(
    df: DataFrame,
    method: str,
    input_col: str,
    output_col: str,
    size: int,
    other: FingerprintParams | None,
) -> DataFrame:
    """Using the data from *input_col*, creates a SimHash fingerprint inside *output_col*.

    Args:
        df (DataFrame): input dataframe
        method (str): specifies the feature model - it must be one of "bag_of_words", "embeddings", "shingles", "tf-idf"
        input_col (str): input column name
        output_col (str): output column name
        size (int): size (in bits) of the fingerprints
        other (FingerprintParams): additional params for the "shingles" feature model

    Returns:
        DataFrame: output dataframe
    """
    assert method in ["bag_of_words", "embeddings", "shingles", "tf-idf"]
    assert input_col in df.columns
    assert size in FINGERPRINT_SIZES

    df = df.filter(F.col(input_col).isNotNull())
    df = (  # Clean column, remove all symbols but a-z and \s
        df.withColumn("cleaned_text", F.regexp_replace(F.col(input_col), r"[\n\r\t]", ""))
        .withColumn("cleaned_text", F.lower(F.col("cleaned_text")))
        .withColumn("cleaned_text", F.regexp_replace(F.col("cleaned_text"), r"[^a-z\s]", ""))
    )
    match method:
        case "bag_of_words":
            df = bag_of_words.make_fingerprints(
                df,
                input_col="cleaned_text",
                output_col=output_col,
                fingerprint_size=size,
            )
        case "shingles":
            assert other is not None
            assert other.shingle_size < size

            df = shingles.make_fingerprints(
                df,
                input_col="cleaned_text",
                output_col=output_col,
                fingerprint_size=size,
                shingle_size=other.shingle_size,
                over_words=other.over_words,
            )
        case "tf-idf":
            df = tf_idf.make_fingerprints(
                df, input_col="cleaned_text", output_col=output_col, fingerprint_size=size
            )
        case "embeddings":
            df = embeddings.make_fingerprints(
                df, input_col="cleaned_text", output_col=output_col, fingerprint_size=size
            )
    df = df.drop("cleaned_text")
    return df
