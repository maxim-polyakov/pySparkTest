FROM bde2020/hive:2.3.2-postgresql-metastore AS hive-dist
FROM bde2020/hadoop-namenode:2.0.0-hadoop3.2.1-java8 AS hadoop-dist
FROM eclipse-temurin:8-jre-jammy AS java8
FROM eclipse-temurin:17-jre-jammy

ARG MAVEN=https://repo1.maven.org/maven2
ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cpu

ENV PIP_NO_INPUT=1
ENV PIP_DISABLE_PIP_VERSION_CHECK=1

# Java 17 — PySpark/JDBC; Java 8 — только hive CLI (JAVA_HOME=/opt/java-8 в ячейке)
COPY --from=java8 /opt/java/openjdk /opt/java-8
COPY --from=hive-dist /opt/hive /opt/hive
COPY --from=hadoop-dist /opt/hadoop-3.2.1 /opt/hadoop-3.2.1
COPY hadoop/scripts/hive-java8.sh /opt/hive/bin/hive-java8
RUN chmod +x /opt/hive/bin/hive-java8

RUN mkdir -p /opt/spark-jars \
    && apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip curl \
    && curl -fsSL -o /opt/spark-jars/postgresql-42.7.4.jar \
        ${MAVEN}/org/postgresql/postgresql/42.7.4/postgresql-42.7.4.jar \
    && apt-get purge -y curl \
    && apt-get autoremove -y \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install --no-cache-dir --ignore-installed \
    pyspark==3.5.4 \
    jupyterlab==4.3.4 \
    pandas \
    matplotlib \
    scikit-learn \
    mlflow==2.18.0 \
    "numpy>=1.26,<2" \
    "protobuf>=4.21,<5" \
    "pyarrow>=18,<19" \
    "transformers>=4.36,<5" \
    "accelerate>=1.1.0" \
    "peft>=0.11.0" \
    sentencepiece \
    rouge-score \
    sacrebleu \
    && python3 -m pip install --no-cache-dir --ignore-installed \
    torch --index-url ${TORCH_INDEX_URL}

RUN python3 -c "import pyspark, os; os.symlink(os.path.dirname(pyspark.__file__), '/opt/spark', target_is_directory=True)"

ENV SPARK_HOME=/opt/spark
ENV JAVA_HOME=/opt/java/openjdk
ENV JAVA8_HOME=/opt/java-8
ENV HIVE_HOME=/opt/hive
ENV HADOOP_HOME=/opt/hadoop-3.2.1
ENV PYSPARK_PYTHON=python3
ENV PYSPARK_DRIVER_PYTHON=python3
ENV JUPYTER_ENABLE_LAB=yes
ENV POSTGRES_JDBC_JAR=/opt/spark-jars/postgresql-42.7.4.jar
ENV PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
ENV USE_TF=0

RUN useradd -m -s /bin/bash -u 1000 jovyan \
    && chown -R jovyan:jovyan /opt/spark-jars /opt/hive /opt/hadoop-3.2.1 /opt/java-8
USER jovyan
WORKDIR /home/jovyan/work

EXPOSE 8888 4040

CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--ServerApp.token=my_secret_token_123", "--ServerApp.allow_origin=*"]
