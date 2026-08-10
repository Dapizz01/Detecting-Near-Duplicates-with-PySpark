import pyspark.sql.functions as F
from pyspark.ml.feature import Tokenizer
from pyspark.sql import DataFrame

from src.simhash import simhash_tf_idf


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

    tokenizer = Tokenizer(inputCol=input_col, outputCol="words")
    df = tokenizer.transform(df)

    # Incremental ID for each row
    df = df.withColumn("_row_id", F.monotonically_increasing_id())

    N = df.count()

    # Schema: _row_id, pos (index of word in tokens), word
    df_exploded = df.select("_row_id", F.posexplode(F.col("words")).alias("pos", "word"))

    # IDF - groups by word and counts how many different _row_id it has, then inverts it.
    idf = (
        df_exploded.select("_row_id", "word")
        .distinct()
        .groupby("word")
        .agg(F.count("*").alias("df"))
        .withColumn("idf", F.log(F.lit(N) / F.col("df")))
    )

    # TF - group by (id, word) and count
    tf = df_exploded.groupby("_row_id", "word").agg(F.count("*").alias("tf"))

    # TF-IDF - joins tf with idf
    tf_idf = tf.join(F.broadcast(idf), on="word").withColumn("tf_idf", F.col("tf") * F.col("idf"))

    # Matches every term with its computed TF-IDF
    weighted = df_exploded.join(tf_idf.select("_row_id", "word", "tf_idf"), on=["_row_id", "word"])

    # Aggregates data back into a list, using a struct to preserve the rows
    agg = weighted.groupby("_row_id").agg(
        F.sort_array(F.collect_list(F.struct("pos", "word", "tf_idf"))).alias("items")
    )
    agg = (
        agg.withColumn("tokens_ordered", F.col("items.word"))
        .withColumn("tf_idf_ordered", F.col("items.tf_idf"))
        .drop("items")
    )

    # Join original df with the aggregated one
    df = df.join(agg, on="_row_id")

    df = df.withColumn("fingerprint_size", F.lit(fingerprint_size))
    df = df.withColumn(
        output_col,
        simhash_tf_idf(  # type: ignore
            F.col("tokens_ordered"),  # type: ignore
            F.col("tf_idf_ordered"),  # type: ignore
            F.col("fingerprint_size"),  # type: ignore
        ),
    )
    df = df.drop("words", "tokens_ordered", "tf_idf_ordered", "fingerprint_size", "_row_id")

    return df
