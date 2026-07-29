from pyspark.sql import Column
from pyspark.sql import functions as F


def simhash(colname: str) -> Column:
    return F.array()
