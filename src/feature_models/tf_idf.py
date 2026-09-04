import pyspark.sql.functions as F
from pyspark.ml.feature import IDF, CountVectorizer, StopWordsRemover, Tokenizer
from pyspark.sql import DataFrame
from pyspark.sql.types import ArrayType, DoubleType

from src.simhash import simhash_tf_idf
from src.spark_utils import get_spark


def make_fingerprints(
    df: DataFrame, input_col: str, output_col: str, fingerprint_size: int
) -> DataFrame:
    assert input_col in df.columns

    # Tokenize word by space
    tokenizer = Tokenizer(inputCol=input_col, outputCol="words")
    df_tokens = tokenizer.transform(df)

    # Removes frequent english words ("the", "and", ...)
    remover = StopWordsRemover(
        inputCol="words",
        outputCol="filtered_words",
        stopWords=StopWordsRemover.loadDefaultStopWords("english"),
    )
    df_cleaned = remover.transform(df_tokens)

    # Converts tokens into a vector of token counts (DF)
    cv = CountVectorizer(inputCol="filtered_words", outputCol="features")
    cv_model = cv.fit(df_cleaned)
    df_tf = cv_model.transform(df_cleaned)

    # Computes TF-IDF, taking DF as input (vector frequencies)
    idf = IDF(inputCol="features", outputCol="tfidf_features")
    idf_model = idf.fit(df_tf)
    df_tfidf = idf_model.transform(df_tf)

    # Extracts vocabulary (dict) of token-weight
    vocabulary = cv_model.vocabulary
    try:
        vocabulary_dict = get_spark().sparkContext.broadcast(
            {word: i for i, word in enumerate(vocabulary)}
        )
    except Exception as e:
        print(f"Error during vocabulary broadcast for TF-IDF: {e}")
        print("Most likely, the vocabulary is too large to be broadcasted.")

    @F.udf(returnType=ArrayType(DoubleType()))
    def extract_weights(tokens, tfidf_vector):
        vector_dict = dict(zip(tfidf_vector.indices, tfidf_vector.values))
        weights = []

        for token in tokens:
            idx = vocabulary_dict.value.get(token)
            # If the token is missing from the vocabulary, set weight to 0.0
            weight = float(vector_dict.get(idx, 0.0)) if idx is not None else 0.0
            weights.append(weight)
        return weights

    # The IDF method returns PySpark vectors, not lists.
    # It's necessary to grab weights form the vocabulary.
    result = df_tfidf.withColumn("tf-idf", extract_weights("filtered_words", "tfidf_features"))

    # Lastly, compute simhash
    result = result.withColumn(
        output_col,
        simhash_tf_idf(F.col("filtered_words"), F.col("tf-idf"), F.lit(fingerprint_size)),  # type: ignore
    )
    return result.drop("words", "features", "filtered_words", "tfidf_features")


# This is an old, custom-built implementation of TF-IDF.
# It works but it's really, really slow.
def make_fingerprints_legacy(
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
