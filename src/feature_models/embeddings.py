from typing import Iterator

import numpy as np
import pandas as pd
import pyspark.sql.functions as F
import requests
from pyspark.sql import DataFrame
from pyspark.sql.types import BinaryType

# These parameters are meant for Ollama, may need to be changed on other devices.
OLLAMA_URL = "http://localhost:11434/api/embed"
MODEL = "Qwen3-embedding:0.6b"
BATCH_SIZE = 16


@F.pandas_udf(BinaryType())  # type: ignore # noqa: F821
def get_embeddings(iterator: Iterator[pd.Series]) -> Iterator[pd.Series]:
    """Pandas UDF function that transforms batches of rows into fingerprints, using an embedding model.

    Args:
        iterator (Iterator[pd.Series]): input batch

    Yields:
        Iterator[pd.Series]: output batch
    """

    # TODO: implement sanity check (strings must be lower than a threshold, otherwise memory usage goes through the roof)
    def quantize(embedding: list[float]) -> bytes:
        if not embedding:
            return b""
        arr = np.array(embedding, dtype=np.float32)
        return np.packbits(arr >= 0).tobytes()

    # Ollama provides a handy Python API for queries, but it's not usable
    # in a Conda environment, because the embedding model is hosted in the base environment.
    # As a workaround, the HTTP API are used here.
    session = requests.Session()

    for series in iterator:
        text_list = series.tolist()
        embeddings = []

        # Iterate over each batch
        for i in range(0, len(text_list), BATCH_SIZE):
            batch = text_list[i : i + BATCH_SIZE]
            try:
                payload = {"model": MODEL, "input": batch}
                response = session.post(OLLAMA_URL, json=payload, timeout=60)
                response.raise_for_status()
                data = response.json()
                batch_res = data.get("embeddings", [])

                embeddings.extend([quantize(emb) for emb in batch_res])
                # Output: [b"\x01\x02", b"...", ...]
            except Exception:
                # Padded with None, the output series must have the same number
                # of elements as the input series
                embeddings.extend([None] * len(batch))
        yield pd.Series(embeddings)


def make_fingerprints(
    df: DataFrame, input_col: str, output_col: str, fingerprint_size: int
) -> DataFrame:
    """Builds binary fingerprints using a local transformer-based embedding model (Qwen3-embedding:0.6b) via Ollama.

    Args:
        df (DataFrame): input dataframe
        input_col (str): input column name
        output_col (str): output column name
        fingerprint_size (int): size of the fingerprint, in bits.

    Returns:
        DataFrame: output dataframe
    """
    assert input_col in df.columns

    df = df.withColumn(
        output_col,
        F.substring(get_embeddings(F.col(input_col)), 0, int(fingerprint_size / 8)),  # type: ignore
    )  # type: ignore

    return df
