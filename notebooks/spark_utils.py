"""Одна SparkSession на весь ноутбук."""
from __future__ import annotations

import os

from pyspark.sql import SparkSession


def _clean_hadoop_env() -> None:
    # HADOOP_HOME/HIVE_HOME из образа ломают JDBC; SPARK_HOME не трогаем
    for _var in ("HADOOP_HOME", "HADOOP_PREFIX", "HIVE_HOME"):
        os.environ.pop(_var, None)
    os.environ.setdefault("HADOOP_CONF_DIR", "/etc/hadoop")


def create_spark_session(
    *,
    app_name: str = "pysparktest",
    hdfs_url: str,
    postgres_jar: str,
    driver_memory: str = "1g",
) -> SparkSession:
    """Сессия для JDBC: без HDFS-конфига в JVM до записи в HDFS."""
    _clean_hadoop_env()

    spark = (
        SparkSession.builder
        .appName(app_name)
        .master("local[1]")
        .config("spark.jars", postgres_jar)
        .config("spark.driver.memory", driver_memory)
        .config("spark.ui.enabled", "false")
        .config("spark.sql.shuffle.partitions", "2")
        .config("spark.python.worker.memory", "128m")
        .config(
            "spark.driver.extraJavaOptions",
            "-XX:+UseG1GC -XX:MaxMetaspaceSize=256m",
        )
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")
    return spark


def enable_hdfs(spark: SparkSession, hdfs_url: str) -> None:
    """Перед записью в HDFS — подключаем Hadoop FS (после JDBC)."""
    spark.conf.set("spark.hadoop.fs.defaultFS", hdfs_url)
    spark.conf.set("spark.hadoop.dfs.replication", "1")


def require_spark(spark=None) -> SparkSession:
    if spark is not None:
        return spark
    raise RuntimeError(
        "Нет SparkSession. Сначала выполните ячейку загрузки данных (переменная spark)."
    )
