FROM eclipse-temurin:17-jre-jammy

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

RUN pip3 install --no-cache-dir \
    pyspark==3.5.4 \
    jupyterlab==4.3.4 \
    pandas

ENV JAVA_HOME=/opt/java/openjdk
ENV PYSPARK_PYTHON=python3
ENV PYSPARK_DRIVER_PYTHON=python3
ENV JUPYTER_ENABLE_LAB=yes

RUN useradd -m -s /bin/bash -u 1000 jovyan
USER jovyan
WORKDIR /home/jovyan/work

EXPOSE 8888 4040

CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser"]
