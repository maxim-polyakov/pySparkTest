#!/usr/bin/env bash
# Регистрация EXTERNAL TABLE alpaca (внутри hive-server)
set -euo pipefail

HIVE_DB="${HIVE_DB:-pysparktest}"
HIVE_TABLE="${HIVE_TABLE:-alpaca}"
HDFS_PATH="${HDFS_PATH:-hdfs://namenode:9000/data/alpaca}"
export HIVE_DB HIVE_TABLE HDFS_PATH

hive -e "
CREATE DATABASE IF NOT EXISTS ${HIVE_DB};
DROP TABLE IF EXISTS ${HIVE_DB}.${HIVE_TABLE};
CREATE EXTERNAL TABLE ${HIVE_DB}.${HIVE_TABLE} (
    instruction STRING,
    input STRING,
    output STRING,
    text STRING
)
STORED AS PARQUET
LOCATION '${HDFS_PATH}';
SHOW TABLES IN ${HIVE_DB};
DESCRIBE ${HIVE_DB}.${HIVE_TABLE};
"

echo "Hive table ${HIVE_DB}.${HIVE_TABLE} -> ${HDFS_PATH}"
