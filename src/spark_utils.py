from pyspark.sql import SparkSession

from .config import (
    SPARK_APP_NAME,
    SPARK_DRIVER_MEMORY,
    SPARK_MASTER,
    SPARK_SHUFFLE_PARTITIONS,
)


def get_spark() -> SparkSession:
    """Creates a local SparkSession, using the parameters in the configuration file.
    Returns the instance of SparkSession if Spark is already initialized.
    """
    # Checks if session is already active
    active_session = SparkSession.getActiveSession()
    if active_session is not None:
        return active_session

    # Generates a new session and returns it
    builder = (
        SparkSession.builder.appName(SPARK_APP_NAME)
        .master(SPARK_MASTER)
        .config("spark.driver.memory", SPARK_DRIVER_MEMORY)
        .config("spark.sql.shuffle.partitions", SPARK_SHUFFLE_PARTITIONS)
        .config("spark.sql.warehouse.dir", "/tmp/spark-warehouse")
        # .config("spark.driver.bindAddress", "127.0.0.1")
        # .config("spark.python.profile", "true")
        # .config("spark.python.profile.memory", "true")
    )
    return builder.getOrCreate()


def stop_spark(spark: SparkSession) -> None:
    """Stops the current SparkSession.

    Args:
        spark (SparkSession): the Spark session to close.
    """
    spark.stop()
