from pyspark.sql import SparkSession

spark = SparkSession.builder.appName("pySparkTest").getOrCreate()

df = spark.createDataFrame([(1, "a"), (2, "b")], ["id", "value"])
df.show()

spark.stop()
