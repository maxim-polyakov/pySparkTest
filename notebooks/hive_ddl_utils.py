"""Hive: проверка портов и чтение таблицы через Spark SQL (metastore)."""
from __future__ import annotations

import os
import socket
import time

from pyspark.sql import DataFrame, SparkSession


def build_hive_spark(*_args, **_kwargs):
    raise RuntimeError(
        "build_hive_spark удалён. Используйте create_hive_spark_session / read_spells_hive."
    )


def wait_hive_ports(max_wait_sec: int = 45) -> None:
    checks = [
        ("metastore", "hive-metastore", 9083),
        (
            "hiveserver2",
            os.environ.get("HIVE_SERVER_HOST", "hive-server"),
            int(os.environ.get("HIVE_SERVER_PORT", "10000")),
        ),
    ]
    for name, host, port in checks:
        print(f"Ожидание {name} {host}:{port} ...")
        ok = False
        for _ in range(0, max_wait_sec, 2):
            try:
                s = socket.create_connection((host, port), timeout=2)
                s.close()
                print(f"  {name} доступен.")
                ok = True
                break
            except OSError:
                time.sleep(2)
        if not ok:
            raise RuntimeError(f"{name} недоступен. docker compose ps")


wait_hive_server = wait_hive_ports


def create_hive_spark_session(
    *,
    app_name: str = "pysparktest-hive",
    hdfs_url: str | None = None,
    metastore_uri: str | None = None,
    driver_memory: str | None = None,
) -> SparkSession:
    """SparkSession с Hive catalog (таблица зарегистрирована Airflow / register-spells-table.sh)."""
    hdfs_url = hdfs_url or os.environ.get("HDFS_URL", "hdfs://namenode:9000")
    metastore_uri = metastore_uri or os.environ.get(
        "HIVE_METASTORE_URI", "thrift://hive-metastore:9083"
    )
    driver_memory = driver_memory or os.environ.get("SPARK_DRIVER_MEMORY", "1g")
    warehouse = f"{hdfs_url}/user/hive/warehouse"

    os.environ.setdefault("HADOOP_CONF_DIR", "/etc/hadoop")
    for _var in ("HADOOP_HOME", "HADOOP_PREFIX", "HIVE_HOME"):
        os.environ.pop(_var, None)

    spark = (
        SparkSession.builder.appName(app_name)
        .master("local[1]")
        .config("spark.driver.memory", driver_memory)
        .config("spark.ui.enabled", "false")
        .config("spark.sql.shuffle.partitions", "2")
        .config("spark.python.worker.memory", "128m")
        .config("spark.hadoop.fs.defaultFS", hdfs_url)
        .config("spark.sql.catalogImplementation", "hive")
        .config("hive.metastore.uris", metastore_uri)
        .config("spark.hadoop.hive.metastore.uris", metastore_uri)
        .config("spark.sql.warehouse.dir", warehouse)
        .config(
            "spark.driver.extraJavaOptions",
            "-XX:+UseG1GC -XX:MaxMetaspaceSize=256m",
        )
        .enableHiveSupport()
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")
    return spark


def read_spells_hive(
    spark: SparkSession,
    hive_db: str = "pysparktest",
    hive_table: str = "spells",
) -> DataFrame:
    from spark_utils import require_spark

    spark = require_spark(spark)
    full_name = f"{hive_db}.{hive_table}"
    print(f"=== SELECT * FROM {full_name} (Hive metastore) ===")
    return spark.sql(f"SELECT * FROM {full_name}")


def read_spells_parquet(spark, hdfs_path: str) -> DataFrame:
    """Запасной вариант: Parquet по LOCATION таблицы (без SQL)."""
    from spark_utils import require_spark

    return require_spark(spark).read.parquet(hdfs_path)
