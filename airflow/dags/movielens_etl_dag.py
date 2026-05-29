"""
DAG: PostgreSQL (movies, ratings) → HDFS (Parquet) → EXTERNAL TABLE в Hive.

Требования:
  - Postgres на host:5432, таблицы movies и ratings (см. notebooks/movielens_schema.sql)
  - docker compose up -d (spark-jupyter, hive-server, hdfs, airflow-scheduler, ...)
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
    dag_id="movielens_postgres_to_hive",
    default_args=default_args,
    description="MovieLens: Postgres movies+ratings → HDFS Parquet → Hive",
    schedule=None,
    start_date=datetime(2025, 1, 1),
    catchup=False,
    tags=["movielens", "recommend", "spark", "hive", "hdfs"],
) as dag:
    postgres_to_hdfs = BashOperator(
        task_id="postgres_to_hdfs",
        bash_command=(
            "docker exec spark_dataspell_container "
            "python3 /home/jovyan/work/jdbc_movielens_to_hdfs.py && echo ok"
        ),
        execution_timeout=timedelta(minutes=30),
    )

    register_hive_tables = BashOperator(
        task_id="register_hive_tables",
        bash_command=(
            "docker exec hive-server "
            "bash /scripts/register-movielens-tables.sh && echo ok"
        ),
        execution_timeout=timedelta(minutes=5),
    )

    postgres_to_hdfs >> register_hive_tables
