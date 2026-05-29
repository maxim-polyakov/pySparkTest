#!/usr/bin/env bash
# MovieLens: EXTERNAL TABLE movies + ratings (внутри hive-server)
set -euo pipefail

HIVE_DB="${HIVE_DB:-pysparktest}"
MOVIES_TABLE="${MOVIES_TABLE:-movies}"
RATINGS_TABLE="${RATINGS_TABLE:-ratings}"
MOVIES_HDFS_PATH="${MOVIES_HDFS_PATH:-hdfs://namenode:9000/data/movies}"
RATINGS_HDFS_PATH="${RATINGS_HDFS_PATH:-hdfs://namenode:9000/data/ratings}"
export HIVE_DB MOVIES_TABLE RATINGS_TABLE MOVIES_HDFS_PATH RATINGS_HDFS_PATH

hive -e "
CREATE DATABASE IF NOT EXISTS ${HIVE_DB};

DROP TABLE IF EXISTS ${HIVE_DB}.${RATINGS_TABLE};
DROP TABLE IF EXISTS ${HIVE_DB}.${MOVIES_TABLE};

CREATE EXTERNAL TABLE ${HIVE_DB}.${MOVIES_TABLE} (
    movieId INT,
    title STRING,
    genres STRING
)
STORED AS PARQUET
LOCATION '${MOVIES_HDFS_PATH}';

CREATE EXTERNAL TABLE ${HIVE_DB}.${RATINGS_TABLE} (
    userId INT,
    movieId INT,
    rating DOUBLE,
    \`timestamp\` BIGINT
)
STORED AS PARQUET
LOCATION '${RATINGS_HDFS_PATH}';

SHOW TABLES IN ${HIVE_DB};
DESCRIBE ${HIVE_DB}.${MOVIES_TABLE};
DESCRIBE ${HIVE_DB}.${RATINGS_TABLE};
"
# Не делать SELECT COUNT(*) здесь: Hive-on-MR в Docker часто «висит» часами.
# Проверка в Jupyter: spark.sql('SELECT COUNT(*) FROM pysparktest.ratings').show()

echo "Hive: ${HIVE_DB}.${MOVIES_TABLE} -> ${MOVIES_HDFS_PATH}"
echo "Hive: ${HIVE_DB}.${RATINGS_TABLE} -> ${RATINGS_HDFS_PATH}"
