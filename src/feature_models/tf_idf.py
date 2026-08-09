import pyspark.sql.functions as F
from pyspark.ml.feature import Tokenizer
from pyspark.sql import DataFrame


# TODO: check that it works as it should
def make_fingerprints(
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

    N = df.count()

    idf = (
        df.select(F.explode(F.array_distinct(F.col("tokens"))).alias("word"))
        .groupby("word")
        .agg(F.count("*").alias("df"))
        .withColumn("idf", F.log(F.lit(N) / F.col("df")))
    )

    # TODO: not so efficient to use collect, optimize?
    idf_dict = {row["word"]: float(row["idf"]) for row in idf.collect()}
    idf_map = F.create_map([F.lit(x) for item in idf_dict.items() for x in item])

    df = df.withColumn(
        "tf",
        F.transform(F.col("tokens"), lambda x: F.size(F.filter(F.col("tokens"), lambda y: y == x))),
    )
    df = df.withColumn(
        "tf-idf",
        F.zip_with(F.col("tokens"), F.col("tf"), lambda word, tf: tf * idf_map.getItem(word)),
    )

    return df
