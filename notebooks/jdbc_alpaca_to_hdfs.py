"""
PostgreSQL (таблица alpaca) → Parquet в HDFS.
Запуск: docker compose up -d, таблица alpaca создана в Postgres на host:5432.
"""
import os

from hdfs_utils import write_parquet_hdfs
from spark_utils import create_spark_session, enable_hdfs

POSTGRES_JAR = os.environ.get(
    "POSTGRES_JDBC_JAR", "/opt/spark-jars/postgresql-42.7.4.jar"
)
HDFS_URL = os.environ.get("HDFS_URL", "hdfs://namenode:9000")
HDFS_PATH = os.environ.get("ALPACA_HDFS_PATH", f"{HDFS_URL}/data/alpaca")
POSTGRES_JDBC_URL = os.environ.get(
    "POSTGRES_JDBC_URL",
    "jdbc:postgresql://host.docker.internal:5432/postgres",
)
POSTGRES_TABLE = os.environ.get("POSTGRES_ALPACA_TABLE", "alpaca")
JDBC_USER = os.environ.get("POSTGRES_USER", "postgres")
JDBC_PASSWORD = os.environ.get("POSTGRES_PASSWORD", "password")

spark = create_spark_session(
    app_name="jdbc-alpaca-to-hdfs",
    hdfs_url=HDFS_URL,
    postgres_jar=POSTGRES_JAR,
    driver_memory=os.environ.get("SPARK_DRIVER_MEMORY", "1g"),
)

print(f"=== Чтение из PostgreSQL: {POSTGRES_TABLE} ===")
df = (
    spark.read.format("jdbc")
    .option("url", POSTGRES_JDBC_URL)
    .option("dbtable", POSTGRES_TABLE)
    .option("user", JDBC_USER)
    .option("password", JDBC_PASSWORD)
    .option("driver", "org.postgresql.Driver")
    .option("numPartitions", "1")
    .option("connectTimeout", "10")
    .option("socketTimeout", "60")
    .load()
)

df.printSchema()
df.show(3, truncate=80)

enable_hdfs(spark, HDFS_URL)
print(f"=== Запись в HDFS: {HDFS_PATH} ===")
write_parquet_hdfs(df, HDFS_PATH, spark, HDFS_URL)

print("=== Проверка чтения из HDFS ===")
spark.read.parquet(HDFS_PATH).show(3, truncate=80)

print(f"Готово. Hive: pysparktest.alpaca → {HDFS_PATH}")
