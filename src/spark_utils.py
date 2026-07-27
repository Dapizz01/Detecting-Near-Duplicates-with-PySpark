from pyspark.sql import SparkSession

from .config import (
    SPARK_APP_NAME,
    SPARK_DRIVER_MEMORY,
    SPARK_MASTER,
    SPARK_SHUFFLE_PARTITIONS,
)


def get_spark() -> SparkSession:
    """
    Creates a local SparkSession, using the parameters in the configuration file.
    Returns the instance of SparkSession if Spark is already initialized.

    Usage:
        from src.spark_utils import get_spark
        spark = get_spark()
    """
    builder = (
        SparkSession.builder.appName(SPARK_APP_NAME)
        .master(SPARK_MASTER)
        .config("spark.driver.memory", SPARK_DRIVER_MEMORY)
        .config("spark.sql.shuffle.partitions", SPARK_SHUFFLE_PARTITIONS)
        .config("spark.sql.warehouse.dir", "/tmp/spark-warehouse")
    )
    return builder.getOrCreate()


def stop_spark(spark: SparkSession) -> None:
    """Stops the current SparkSession."""
    spark.stop()
