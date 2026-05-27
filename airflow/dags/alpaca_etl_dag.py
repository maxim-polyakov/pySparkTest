"""
DAG: PostgreSQL (alpaca) → HDFS (Parquet) → EXTERNAL TABLE в Hive.

Требования:
  - Postgres на host:5432, таблица alpaca (instruction, input, output, text)
  - docker compose up -d (spark-jupyter, hive-server, hdfs, ...)
"""
from __future__ import annotations

from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.bash import BashOperator

default_args = {
    "owner": "pysparktest",
    "depends_on_past": False,
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=2),
}

with DAG(
    dag_id="alpaca_postgres_to_hive",
    default_args=default_args,
    description="Alpaca: Postgres JDBC → HDFS Parquet → Hive EXTERNAL TABLE",
    schedule=None,
    start_date=datetime(2025, 1, 1),
    catchup=False,
    tags=["alpaca", "spark", "hive", "hdfs"],
) as dag:
    postgres_to_hdfs = BashOperator(
        task_id="postgres_to_hdfs",
        bash_command=(
            "docker exec spark_dataspell_container "
            "python3 /home/jovyan/work/jdbc_alpaca_to_hdfs.py && echo ok"
        ),
    )

    register_hive_table = BashOperator(
        task_id="register_hive_table",
        bash_command=(
            "docker exec hive-server "
            "bash /scripts/register-alpaca-table.sh && echo ok"
        ),
    )

    postgres_to_hdfs >> register_hive_table
