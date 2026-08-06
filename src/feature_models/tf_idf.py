from pyspark.ml.feature import Tokenizer
from pyspark.sql import DataFrame


def create_fingerprints(
    df: DataFrame, input_col: str, output_col: str, fingerprint_size: int
) -> DataFrame:
    """Builds fingerprints using the TF-IDF (Term Frequency - Inverse Document Frequency) feature model.

    Args:
        df (DataFrame): input dataframe
        input_col (str): input column name
        output_col (str): output column name
        fingerprint_size (int): size of each fingerprint, in bits

    Returns:
        DataFrame: output dataframe
    """
    assert input_col in df.columns

    tokenizer = Tokenizer(inputCol=input_col, outputCol="tokens")
    df = tokenizer.transform(df)

    # TODO: finish this

    return df
