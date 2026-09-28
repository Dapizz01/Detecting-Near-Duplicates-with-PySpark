from dataclasses import dataclass

from pyspark.sql import DataFrame

from src.feature_models import bag_of_words, embeddings, shingles, tf_idf

FINGERPRINT_SIZES = (32, 64, 128, 256, 512)


@dataclass
class ShingleParams:
    shingle_size: int
    over_words: bool


def create_fingerprints(
    df: DataFrame,
    model: str,
    input_col: str,
    output_col: str,
    fingerprint_size: int,
    other: ShingleParams | None,
) -> DataFrame:
    """Using the data from *input_col*, creates a SimHash fingerprint inside *output_col*.

    Args:
        df (DataFrame): input dataframe
        model (str): specifies the feature model - it must be one of "bag_of_words", "embeddings", "shingles", "tf-idf"
        input_col (str): input column name
        output_col (str): output column name
        fingerprint_size (int): size (in bits) of the fingerprints
        other (FingerprintParams): additional params for the "shingles" feature model

    Returns:
        DataFrame: output dataframe
    """
    assert model in [
        "bag_of_words",
        "embeddings",
        "shingles",
        "tf-idf",
    ], f"[create_fingerprints] The selected model ({model}) is not available."
    assert (
        input_col in df.columns
    ), f"[create_fingerprints] The input column ({input_col}) does not exist in the DataFrame."
    assert (
        fingerprint_size in FINGERPRINT_SIZES
    ), f"[create_fingerprints] The selected fingerprint size ({fingerprint_size}) is not valid. Please, select one of the following: {FINGERPRINT_SIZES}"
    assert (
        output_col not in df.columns
    ), f"[create_fingerprints] The selected output column ({output_col}) already exists in the DataFrame. In order to avoid data loss, select an unused column name."
    assert (
        dict(df.dtypes)[input_col] == "string"
    ), f"[create_fingerprints] The selected input column has type ({dict(df.dtypes)[input_col]}) instead of 'string'."

    match model:
        case "bag_of_words":
            df = bag_of_words.make_fingerprints(
                df,
                input_col=input_col,
                output_col=output_col,
                fingerprint_size=fingerprint_size,
            )
        case "shingles":
            assert (
                other is not None
            ), "[create_fingerprints] The 'other' field must contain a ShingleParams object if the selected feature model is 'shingles'."
            assert (
                other.shingle_size > 0
            ), f"[create_fingerprints] The selected shingle size ({other.shingle_size}) is not positive."

            df = shingles.make_fingerprints(
                df,
                input_col=input_col,
                output_col=output_col,
                fingerprint_size=fingerprint_size,
                shingle_size=other.shingle_size,
                over_words=other.over_words,
            )
        case "tf-idf":
            df = tf_idf.make_fingerprints(
                df,
                input_col=input_col,
                output_col=output_col,
                fingerprint_size=fingerprint_size,
            )
        case "embeddings":
            df = embeddings.make_fingerprints(
                df,
                input_col=input_col,
                output_col=output_col,
                fingerprint_size=fingerprint_size,
            )

    assert (
        output_col in df.columns
    ), "[create_fingerprints] Runtime error: the feature model failed to generate the required fingerprints."

    # df = df.drop(clean_col)
    return df
