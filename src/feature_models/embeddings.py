from typing import Iterator

import pandas as pd
import pyspark.sql.functions as F
import requests
from pyspark.sql import DataFrame
from pyspark.sql.types import BinaryType

OLLAMA_URL = "http://localhost:11434/api/embed"
MODEL = "Qwen3-embedding:0.6b"
BATCH_SIZE = 16  # The optimal value depends on the hardware


@F.pandas_udf(BinaryType())  # type: ignore # noqa: F821
def get_embeddings(iterator: Iterator[pd.Series]) -> Iterator[pd.Series]:
    def list_to_bin(vals: list[str]) -> bytes:
        if not vals:
            return b""
        num = int("".join(vals), 2)
        length = (len(vals) + 7) // 8
        return num.to_bytes(length, byteorder="big")

    def quantize(embeddings: list[float]) -> list[str]:
        return ["0" if val < 0 else "1" for val in embeddings]

    session = requests.Session()

    for series in iterator:
        text_list = series.tolist()
        embeddings = []

        for i in range(0, len(text_list), BATCH_SIZE):
            batch = text_list[i : i + BATCH_SIZE]
            try:
                payload = {"model": MODEL, "input": batch}
                response = session.post(OLLAMA_URL, json=payload, timeout=60)
                response.raise_for_status()
                data = response.json()
                batch_res = data.get("embeddings", [])

                batch_embeddings = [quantize(emb) for emb in batch_res]
                embeddings.extend([list_to_bin(emb) for emb in batch_embeddings])
            except Exception as e:
                print(f"Error while generating the embeddings: {e}")
                # Inseriamo None per mantenere la cardinalità in caso di errore
                embeddings.extend([None] * len(batch))

        yield pd.Series(embeddings)


def make_fingerprints(
    df: DataFrame, input_col: str, output_col: str, fingerprint_size: int
) -> DataFrame:
    """_summary_

    Args:
        df (DataFrame): _description_
        input_col (str): _description_
        output_col (str): _description_
        fingerprint_size (int): _description_

    Returns:
        DataFrame: _description_
    """
    assert input_col in df.columns

    df = df.withColumn(
        output_col,
        F.substring(get_embeddings(F.col(input_col)), 0, int(fingerprint_size / 8)),  # type: ignore
    )  # type: ignore

    return df
