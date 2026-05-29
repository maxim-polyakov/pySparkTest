"""
PostgreSQL (movies, ratings) → Parquet в HDFS.

Запуск (в spark-jupyter):
  python3 /home/jovyan/work/jdbc_movielens_to_hdfs.py

Переменные окружения:
  POSTGRES_JDBC_URL, POSTGRES_USER, POSTGRES_PASSWORD
  POSTGRES_MOVIES_TABLE (default movies)
  POSTGRES_RATINGS_TABLE (default ratings)
  MOVIES_HDFS_PATH (default hdfs://namenode:9000/data/movies)
  RATINGS_HDFS_PATH (default hdfs://namenode:9000/data/ratings)
"""
from __future__ import annotations

import os

from pyspark.sql import functions as F

from hdfs_utils import write_parquet_hdfs
from spark_utils import create_spark_session, enable_hdfs

POSTGRES_JAR = os.environ.get(
    "POSTGRES_JDBC_JAR", "/opt/spark-jars/postgresql-42.7.4.jar"
)
HDFS_URL = os.environ.get("HDFS_URL", "hdfs://namenode:9000")
POSTGRES_JDBC_URL = os.environ.get(
    "POSTGRES_JDBC_URL",
    "jdbc:postgresql://host.docker.internal:5432/postgres",
)
JDBC_USER = os.environ.get("POSTGRES_USER", "postgres")
JDBC_PASSWORD = os.environ.get("POSTGRES_PASSWORD", "password")
MOVIES_TABLE = os.environ.get("POSTGRES_MOVIES_TABLE", "movies")
RATINGS_TABLE = os.environ.get("POSTGRES_RATINGS_TABLE", "ratings")
MOVIES_HDFS_PATH = os.environ.get("MOVIES_HDFS_PATH", f"{HDFS_URL}/data/movies")
RATINGS_HDFS_PATH = os.environ.get("RATINGS_HDFS_PATH", f"{HDFS_URL}/data/ratings")

JDBC_OPTS = {
    "url": POSTGRES_JDBC_URL,
    "user": JDBC_USER,
    "password": JDBC_PASSWORD,
    "driver": "org.postgresql.Driver",
    "connectTimeout": "10",
    "socketTimeout": "120",
}


def read_jdbc_table(spark, table: str, *, num_partitions: str = "1"):
    return (
        spark.read.format("jdbc")
        .option("dbtable", table)
        .option("numPartitions", num_partitions)
        .options(**JDBC_OPTS)
        .load()
    )


def export_table(
    spark, table: str, hdfs_path: str, *, num_partitions: str = "1"
) -> None:
    print(f"=== Чтение из PostgreSQL: {table} ===")
    df = read_jdbc_table(spark, table, num_partitions=num_partitions)
    if table.lower() == "ratings" or "rating" in df.columns:
        df = df.withColumn("rating", F.col("rating").cast("double"))
    df.printSchema()
    df.show(3, truncate=80)
    print(f"=== Запись в HDFS: {hdfs_path} ===")
    write_parquet_hdfs(df, hdfs_path, spark, HDFS_URL)
    print(f"=== Проверка HDFS: {hdfs_path} ===")
    spark.read.parquet(hdfs_path).show(3, truncate=80)


def main() -> None:
    spark = create_spark_session(
        app_name="jdbc-movielens-to-hdfs",
        hdfs_url=HDFS_URL,
        postgres_jar=POSTGRES_JAR,
        driver_memory=os.environ.get("SPARK_DRIVER_MEMORY", "1g"),
    )
    enable_hdfs(spark, HDFS_URL)

    export_table(spark, MOVIES_TABLE, MOVIES_HDFS_PATH)
    export_table(spark, RATINGS_TABLE, RATINGS_HDFS_PATH, num_partitions="2")

    print(
        f"Готово. Hive: pysparktest.movies → {MOVIES_HDFS_PATH}; "
        f"pysparktest.ratings → {RATINGS_HDFS_PATH}"
    )
    spark.stop()


if __name__ == "__main__":
    main()
