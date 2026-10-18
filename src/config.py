import os
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")

DATA_DIR = Path(os.getenv("DATA_DIR", PROJECT_ROOT / "data"))
if not DATA_DIR.is_absolute():
    DATA_DIR = (PROJECT_ROOT / DATA_DIR).resolve()

OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", PROJECT_ROOT / "output"))
if not OUTPUT_DIR.is_absolute():
    OUTPUT_DIR = (PROJECT_ROOT / OUTPUT_DIR).resolve()

# --- Spark Configuration (for laptop) ---
SPARK_APP_NAME = os.getenv("SPARK_APP_NAME", "pyspark-dev")
SPARK_MASTER = os.getenv("SPARK_MASTER", "local[*]")
SPARK_DRIVER_MEMORY = os.getenv("SPARK_DRIVER_MEMORY", "4g")
SPARK_SHUFFLE_PARTITIONS = os.getenv("SPARK_SHUFFLE_PARTITIONS", "8")

FINGERPRINT_SIZES = (32, 64, 128, 256, 512)
FEATURE_MODELS = ("bag_of_words", "shingles", "tf-idf", "embeddings")


def data_path(*parts: str) -> Path:
    """
    Returns an absolute path in the following form - DATA_DIR/parts[0]/parts[1]...

    Usage:
        df = spark.read.csv(str(data_path("sells", "2024.csv")), header=True)
    """
    return DATA_DIR.joinpath(*parts)


def output_path(*parts: str) -> Path:
    """
    Returns an absolute path in the following form - OUTPUT_DIR/parts[0]/parts[1]...

    Usage:
        df = spark.read.csv(str(data_path("sells", "analysis_results.csv")), header=True)
    """
    return OUTPUT_DIR.joinpath(*parts)
