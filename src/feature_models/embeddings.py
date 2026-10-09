from typing import Iterator

import numpy as np
import pandas as pd
import pyspark.sql.functions as F
import requests
from pyspark.sql import DataFrame
from pyspark.sql.types import BinaryType

from src.fingerprint import FINGERPRINT_SIZES

# These parameters are meant for Ollama, may need to be changed on other devices.
OLLAMA_URL = "http://localhost:11434/api/embed"
MODEL = "Qwen3-embedding:0.6b"
BATCH_SIZE = 16
CTX_SIZE = 1500
PROMPT = "Instruct: Retrieve semantically equivalent texts.\nQuery: "


@F.pandas_udf(BinaryType())  # type: ignore # noqa: F821
def get_embeddings(iterator: Iterator[pd.Series]) -> Iterator[pd.Series]:
    """Pandas UDF function that transforms batches of rows into fingerprints, using an embedding model.

    Args:
        iterator (Iterator[pd.Series]): input batch

    Yields:
        Iterator[pd.Series]: output batch
    """

    def quantize(embedding: list[float], threshold: float) -> bytes:
        if not embedding:
            raise ValueError("[get_embeddings, quantize] The input embedding weight list is empty.")

        arr = np.array(embedding, dtype=np.float32)
        return np.packbits(arr >= threshold).tobytes()

    # Ollama provides a handy Python API for queries, but it's not usable
    # in a Conda environment, because the embedding model is hosted in the base environment.
    # As a workaround, the HTTP API are used here.
    session = requests.Session()

    for series in iterator:
        text_list = series.tolist()
        acc = np.zeros((len(text_list), 1024), dtype=np.float32)

        # Iterate over each batch
        for i in range(0, len(text_list), BATCH_SIZE):
            batch = text_list[i : i + BATCH_SIZE]
            try:
                # Add prompt instruction to all texts
                prompts = [PROMPT + chunk for chunk in batch]
                payload = {"model": MODEL, "input": prompts}
                print(payload)
                response = session.post(OLLAMA_URL, json=payload, timeout=20)
                response.raise_for_status()
                data = response.json()
                batch_res = data.get("embeddings", [])
                for offset, emb in enumerate(batch_res):
                    acc[i + offset] = emb

            except Exception as e:
                raise RuntimeError(
                    "[get_embeddings] Error generating embeddings. Ollama might be down, or not working properly.",
                    e,
                )

        quantized_embeddings = [
            quantize(acc[j].tolist(), float(np.median(acc[j]))) for j in range(len(text_list))
        ]

        yield pd.Series(quantized_embeddings)


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
    assert (
        input_col in df.columns
    ), f"[make_fingerprints (embeddings)] The specified input column ({input_col}) is not present in the DataFrame schema ({df.columns})"

    assert (
        dict(df.dtypes).get(input_col) == "string"
    ), f"[make_fingerprints (embeddings)] The type of the input column ({input_col}) is ({dict(df.dtypes).get(input_col)}) instead of (string)"

    assert (
        output_col not in df.columns
    ), f"[make_fingerprints (embeddings)] The specified output column ({output_col}) is already present in the DataFrame schema ({df.columns})"

    assert (
        fingerprint_size in FINGERPRINT_SIZES
    ), f"[make_fingerprints (embeddings)] The specified fingerprint size ({fingerprint_size}) is invalid; its possible values are ({FINGERPRINT_SIZES})"

    df = df.withColumn(
        output_col,
        F.substring(get_embeddings(F.col(input_col)), 0, int(fingerprint_size / 8)),  # type: ignore
    )  # type: ignore

    return df


@F.pandas_udf(BinaryType())  # type: ignore # noqa: F821
def get_embeddings_experimental(iterator: Iterator[pd.Series]) -> Iterator[pd.Series]:
    raise NotImplementedError(
        "[get_embeddings_experimental_v2] This method is experimental.\n"
        "Due to memory limits, calling get_embeddings(...) on large texts would cause OOM issues on 8GB machines.\n"
        "This method tries to circumvent this by splitting the texts into chunks and assemble the weights together.\n"
        "Unfortunately, this still causes OOM issues (and lowers benchmark scores).\n"
        "It's recommended to just run get_embeddings(...) with 16GB+ memory machines.\n"
        "It's also possible to lower the number of PySpark workers and batch_size."
    )

    def tokenize(text: str, limit: int) -> list[str]:
        return [text[i : i + limit] for i in range(0, len(text), limit)]

    def quantize(embedding: list[float], threshold: float) -> bytes:
        if not embedding:
            return b""
        arr = np.array(embedding, dtype=np.float32)
        return np.packbits(arr >= threshold).tobytes()

    # Ollama provides a handy Python API for queries, but it's not usable
    # in a Conda environment, because the embedding model is hosted in the base environment.
    # As a workaround, the HTTP API are used here.
    session = requests.Session()

    for series in iterator:
        text_list = series.tolist()
        chunks: list[str] = []
        owners: list[int] = []
        for i, text in enumerate(text_list):
            for chunk in tokenize(text, CTX_SIZE // 2):
                chunks.append(chunk)
                owners.append(i)
        acc = np.zeros((len(text_list), 1024), dtype=np.float32)

        # Iterate over each batch
        for i in range(0, len(chunks), BATCH_SIZE):
            batch = chunks[i : i + BATCH_SIZE]
            batch_owners = owners[i : i + BATCH_SIZE]
            try:
                prompts = [PROMPT + chunk for chunk in batch]
                payload = {"model": MODEL, "input": prompts}
                response = session.post(OLLAMA_URL, json=payload, timeout=60)
                response.raise_for_status()
                data = response.json()
                batch_res = data.get("embeddings", [])
                for o, emb in zip(batch_owners, batch_res):
                    acc[o] += np.asarray(emb, dtype=np.float32)

            except Exception as e:
                raise RuntimeError(
                    "[get_embeddings] Error generating embeddings. Ollama might be down, or not working properly.",
                    e,
                )
        # Qwen3-embeddings isn't a 0-centered model.
        # In order to provide a better quantization, the threshold to distinguish 0/1 is the median of all weights.
        # Mean: add.mean()
        # Median: float(np.median(acc)))
        quantized_embeddings = [
            quantize(acc[j].tolist(), float(np.median(j))) for j in range(len(text_list))
        ]
        yield pd.Series(quantized_embeddings)


# TODO: experimental. Works on larger documents, for some reason stops working after 16 fingerprints (a batch)
@F.pandas_udf(BinaryType())  # type: ignore # noqa: F821
def get_embeddings_experimental_v2(iterator: Iterator[pd.Series]) -> Iterator[pd.Series]:
    raise NotImplementedError(
        "[get_embeddings_experimental_v2] This method is experimental.\n"
        "Due to memory limits, calling get_embeddings(...) on large texts would cause OOM issues on 8GB machines.\n"
        "This method tries to circumvent this by splitting the texts into chunks and assemble the weights together.\n"
        "Unfortunately, this still causes OOM issues (and lowers benchmark scores).\n"
        "It's recommended to just run get_embeddings(...) with 16GB+ memory machines.\n"
        "It's also possible to lower the number of PySpark workers and batch_size."
    )

    def tokenize(text: str, limit: int) -> list[str]:
        return [text[i : i + limit] for i in range(0, len(text), limit)]

    def quantize(embedding: list[float], threshold: float) -> bytes:
        if not embedding:
            return b""
        arr = np.array(embedding, dtype=np.float32)
        return np.packbits(arr >= threshold).tobytes()

    # Ollama provides a handy Python API for queries, but it's not usable
    # in a Conda environment, because the embedding model is hosted in the base environment.
    # As a workaround, the HTTP API are used here.
    session = requests.Session()

    for series in iterator:
        text_list = series.tolist()
        chunks: list[str] = []
        chunk_owners: list[int] = []
        chunk_counts = [0] * len(text_list)

        for text_idx, text in enumerate(text_list):
            toks = tokenize(text, CTX_SIZE // 2) if text else []
            for chunk in toks:
                chunks.append(chunk)
                chunk_owners.append(text_idx)
            chunk_counts[text_idx] = len(toks)

        acc = np.zeros((len(text_list), 1024), dtype=np.float32)

        # Batch inference over Ollama
        for batch_start in range(0, len(chunks), BATCH_SIZE):
            batch_chunks = chunks[batch_start : batch_start + BATCH_SIZE]
            batch_owners = chunk_owners[batch_start : batch_start + BATCH_SIZE]
            try:
                prompts = [PROMPT + chunk for chunk in batch_chunks]
                payload = {"model": MODEL, "input": prompts}
                response = session.post(OLLAMA_URL, json=payload, timeout=60)
                response.raise_for_status()
                data = response.json()
                batch_res = data.get("embeddings", [])
                for o, emb in zip(batch_owners, batch_res):
                    acc[o] += np.asarray(emb, dtype=np.float32)

            except Exception as e:
                raise RuntimeError(
                    "[get_embeddings] Error generating embeddings. Ollama might be down, or not working properly.",
                    e,
                )

        # Normalize rows by number of chunks so long texts don't overpower
        for text_idx, count in enumerate(chunk_counts):
            if count > 1:
                acc[text_idx] /= count

        # Quantize row by row using each vector's own median
        quantized_embeddings = []
        for text_idx in range(len(text_list)):
            if chunk_counts[text_idx] == 0:
                # Handle empty text
                quantized_embeddings.append(b"")
            else:
                row = acc[text_idx]
                row_median = float(np.median(row))
                quantized_embeddings.append(quantize(row.tolist(), row_median))

        yield pd.Series(quantized_embeddings)
