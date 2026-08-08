import pyspark.sql.functions as F
from pyspark.sql import DataFrame

from src.simhash import simhash_bow


def make_fingerprints(
    df: DataFrame, input_col: str, output_col: str, fingerprint_size: int, fast: bool
) -> DataFrame:
    """Creates a fingerprint for every value inside *input_col*, using the *bag_of_words* feature model.
    Two strings are near-duplicate if they contain about the same words, in any order.

    Args:
        df (DataFrame): input dataframe
        input_col (str): input column name
        output_col (str): output column name
        fingerprint_size (int): the size of the fingerprints, in bits
        fast (bool): the preferred SimHash algorithm - True for FastSimHash, False for classic SimHash

    Returns:
        DataFrame: result dataframe
    """
    assert input_col in df.columns

    df = (
        df.withColumn("fingerprint_size", F.lit(fingerprint_size))
        .withColumn(output_col, simhash_bow(F.col(input_col), F.col("fingerprint_size")))  # type: ignore
        .drop("fingerprint_size")
    )
    return df
