"""
DAG: PostgreSQL → HDFS (Parquet) → регистрация EXTERNAL TABLE в Hive.

Использует уже запущенные контейнеры spark-jupyter и hive-server.
Перед запуском DAG: docker compose up -d && docker compose build spark-jupyter
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
    dag_id="spells_postgres_to_hive",
    default_args=default_args,
    description="Spells: Postgres JDBC → HDFS Parquet → Hive EXTERNAL TABLE",
    schedule=None,
    start_date=datetime(2025, 1, 1),
    catchup=False,
    tags=["spells", "spark", "hive", "hdfs"],
) as dag:
    # Команда не должна заканчиваться на .sh/.bash — иначе BashOperator
    # ищет файл-шаблон в /opt/airflow/dags (TemplateNotFound).
    postgres_to_hdfs = BashOperator(
        task_id="postgres_to_hdfs",
        bash_command=(
            "docker exec spark_dataspell_container "
            "python3 /home/jovyan/work/jdbc_to_hdfs.py && echo ok"
        ),
    )

    register_hive_table = BashOperator(
        task_id="register_hive_table",
        bash_command=(
            "docker exec hive-server "
            "bash /scripts/register-spells-table.sh && echo ok"
        ),
    )

    postgres_to_hdfs >> register_hive_table
