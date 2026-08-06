import pyspark.sql.functions as F
from pyspark.ml.feature import Tokenizer
from pyspark.sql import DataFrame

from src.simhash import simhash


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
    assert input_col in df.columns

    tokenizer = Tokenizer(inputCol=input_col, outputCol="words")
    df = tokenizer.transform(df)

    df = df.withColumn(  # Each word is hashed into a BinaryType
        "hashed_words",
        F.transform(
            F.col("words"),
            lambda x: F.unhex(F.substring(F.sha2(x, 512), 0, int(fingerprint_size / 4))),
        ),  # Each char in hex string is 4 bits
    )

    df = df.withColumn(output_col, simhash(F.col("hashed_words")))  # type: ignore

    df = df.drop("words", "hashed_words")
    return df
