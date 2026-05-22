# Пример PySpark + HDFS (после docker compose up)
import os

from pyspark.sql import SparkSession

HDFS_URL = os.environ.get("HDFS_URL", "hdfs://namenode:9000")

spark = (
    SparkSession.builder
    .appName("hdfs-example")
    .master("local[*]")
    .config("spark.hadoop.fs.defaultFS", HDFS_URL)
    .getOrCreate()
)

data = [("Огненный шар", 4, 6), ("Молния", 1, 6)]
sdf = spark.createDataFrame(data, ["name", "mana_cost", "damage"])

path = f"{HDFS_URL}/tmp/spells_demo"
sdf.write.mode("overwrite").parquet(path)
print("Записано в", path)

spark.read.parquet(path).show()
spark.stop()
