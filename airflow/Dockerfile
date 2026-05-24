FROM apache/airflow:2.10.5-python3.11

# docker exec из задач DAG (spark-jupyter, hive-server)
USER root
RUN apt-get update \
    && apt-get install -y --no-install-recommends docker.io \
    && rm -rf /var/lib/apt/lists/*
USER airflow
