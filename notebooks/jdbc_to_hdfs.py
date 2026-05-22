"""
PostgreSQL (как в 43_spell_classifiers) → Parquet в HDFS.
Запуск: docker compose up -d, затем выполнить в Jupyter / DataSpell.
"""
import os

from pyspark.sql import SparkSession

POSTGRES_JAR = os.environ.get(
    "POSTGRES_JDBC_JAR", "/opt/spark-jars/postgresql-42.7.4.jar"
)
HDFS_URL = os.environ.get("HDFS_URL", "hdfs://namenode:9000")
HDFS_PATH = f"{HDFS_URL}/data/spells"

active = SparkSession.getActiveSession()
if active is not None:
    active.stop()

spark = (
    SparkSession.builder
    .appName("jdbc-to-hdfs")
    .master("local[*]")
    .config("spark.jars", POSTGRES_JAR)
    .config("spark.hadoop.fs.defaultFS", HDFS_URL)
    .getOrCreate()
)

print("=== Чтение из PostgreSQL ===")
df = (
    spark.read.format("jdbc")
    .option("url", "jdbc:postgresql://host.docker.internal:5432/postgres")
    .option("dbtable", "spells")
    .option("user", "postgres")
    .option("password", "password")
    .option("driver", "org.postgresql.Driver")
    .load()
)

df.show()

print(f"=== Запись в HDFS: {HDFS_PATH} ===")
df.write.mode("overwrite").parquet(HDFS_PATH)

print("=== Проверка чтения из HDFS ===")
spark.read.parquet(HDFS_PATH).show()

print("Готово. В NameNode UI: Browse → /data/spells")
# spark.stop()
