from pyspark.sql import SparkSession
import sklearn

# spark = SparkSession.builder.appName("OptimalBreeding").getOrCreate()
# bulls_spark = spark.read.csv("bulls.csv", header=True, inferSchema=True)
# cows_spark = spark.read.csv("cows.csv", header=True, inferSchema=True)
#
# # Кросс-джойн для создания пар
# all_pairs_spark = bulls_spark.crossJoin(cows_spark)