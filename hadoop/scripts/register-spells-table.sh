#!/usr/bin/env bash
# Регистрация EXTERNAL TABLE (внутри hive-server: docker exec hive-server bash /scripts/register-spells-table.sh)
set -euo pipefail

HIVE_DB="${HIVE_DB:-pysparktest}"
HIVE_TABLE="${HIVE_TABLE:-spells}"
HDFS_PATH="${HDFS_PATH:-hdfs://namenode:9000/data/spells}"
HIVE_BIN="${HIVE_BIN:-/opt/hive/bin/hive}"
export HIVE_DB HIVE_TABLE HDFS_PATH

"${HIVE_BIN}" -e "
CREATE DATABASE IF NOT EXISTS ${HIVE_DB};
DROP TABLE IF EXISTS ${HIVE_DB}.${HIVE_TABLE};
CREATE EXTERNAL TABLE ${HIVE_DB}.${HIVE_TABLE} (
    id INT,
    description STRING,
    image_url STRING,
    mana_cost INT,
    name STRING,
    sound_url STRING,
    animation_url STRING,
    play_effect_url STRING,
    damage INT
)
STORED AS PARQUET
LOCATION '${HDFS_PATH}';
SHOW TABLES IN ${HIVE_DB};
"

echo "Hive table ${HIVE_DB}.${HIVE_TABLE} -> ${HDFS_PATH}"
