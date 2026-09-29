import pyspark.sql.functions as F
from pyspark.sql import DataFrame

from src.config import FINGERPRINT_SIZES
from src.simhash import simhash_bow


def make_fingerprints(
    df: DataFrame, input_col: str, output_col: str, fingerprint_size: int
) -> DataFrame:
    """Creates a fingerprint for every value inside *input_col*, using the *bag_of_words* feature model.
    Two strings are near-duplicate if they contain about the same words, in any order.

    Args:
        df (DataFrame): input dataframe
        input_col (str): input column name
        output_col (str): output column name
        fingerprint_size (int): the size of the fingerprints, in bits

    Returns:
        DataFrame: result dataframe
    """
    assert (
        input_col in df.columns
    ), f"[make_fingerprints (bag_of_words)] The specified input column ({input_col}) is not present in the DataFrame schema ({df.columns})"

    assert (
        dict(df.dtypes).get(input_col) == "string"
    ), f"[make_fingerprints (bag_of_words)] The type of the input column ({input_col}) is ({dict(df.dtypes).get(input_col)}) instead of (string)"

    assert (
        output_col not in df.columns
    ), f"[make_fingerprints (bag_of_words)] The specified output column ({output_col}) is already present in the DataFrame schema ({df.columns})"

    assert (
        fingerprint_size in FINGERPRINT_SIZES
    ), f"[make_fingerprints (bag_of_words)] The specified fingerprint size ({fingerprint_size}) is invalid; its possible values are ({FINGERPRINT_SIZES})"

    df = (
        df.withColumn("fingerprint_size", F.lit(fingerprint_size))
        .withColumn(output_col, simhash_bow(F.col(input_col), F.col("fingerprint_size")))  # type: ignore
        .drop("fingerprint_size")
    )
    return df
