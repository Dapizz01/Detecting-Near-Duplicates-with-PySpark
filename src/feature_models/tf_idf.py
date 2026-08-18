import pyspark.sql.functions as F
from pyspark.ml.feature import Tokenizer
from pyspark.sql import DataFrame

from src.simhash import simhash_tf_idf


def make_fingerprints(
    df: DataFrame, input_col: str, output_col: str, fingerprint_size: int
) -> DataFrame:
    assert input_col in df.columns

    # Tokenize words by space
    tokenizer = Tokenizer(inputCol=input_col, outputCol="words")
    df = tokenizer.transform(df)

    # Assign id to each row
    df = df.withColumn("_row_id", F.monotonically_increasing_id())

    # Explode words, creating a row for each word in words, with schema _row_id | word
    df_exploded = df.select("_row_id", F.explode(F.col("words")).alias("word"))

    # Compute document frequency - how many documents contain a certain word
    # Group by "word" and count distinct _row_id
    df_count = df_exploded.groupBy("word").agg(F.count_distinct("_row_id").alias("df_count"))

    # Compute term frequency - how many times does a word appear in a document
    # Count how many instances of ("_row_id", "word") there are in df_exploded
    tf = df_exploded.groupBy("_row_id", "word").agg(F.count("*").alias("tf"))

    total_docs = df.count()

    # Join TF and DF (document frequency) by "word" - now each _row_id | word has their own TF and DF
    # Then, compute TF-IDF, inverting DF into IDF and then multiplying TF and IDF
    tf_idf = tf.join(F.broadcast(df_count), "word")
    tf_idf = tf_idf.withColumn("tf_idf", F.col("tf") * F.log(F.lit(total_docs) / F.col("df_count")))

    # Aggregate by _row_id, making a "terms" column whose type is list[struct]
    # Trick: F.struct "merges" columns into a struct, while F.collect_list transforms structs into lists
    agg = tf_idf.groupBy("_row_id").agg(F.collect_list(F.struct("word", "tf_idf")).alias("terms"))

    # Unwrap struct
    agg = agg.withColumn("tokens", F.col("terms.word"))
    agg = agg.withColumn("tf_idf", F.col("terms.tf_idf"))

    # Join original DataFrame with "agg" DataFrame, by _row_id
    df = df.join(agg, "_row_id", "left")

    df = df.withColumn(
        output_col,
        simhash_tf_idf(
            F.col("tokens"),  # type: ignore
            F.col("tf_idf"),  # type: ignore
            F.lit(fingerprint_size),  # type: ignore
        ),
    )
    df = df.drop("words", "tokens", "tf_idf", "_row_id")

    return df
