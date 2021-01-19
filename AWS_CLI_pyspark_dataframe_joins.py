from pyspark.sql import SparkSession
from pyspark import SparkContext, SparkConf
from pyspark.ml.linalg import Vectors
from pyspark.ml.feature import VectorAssembler
import pyspark.ml
from pyspark.ml.regression import LinearRegression
from pyspark.ml.regression import GeneralizedLinearRegression
from pyspark.sql.functions import col

## This file uses the AWS CLI to get data on the number of properties in various zip codes,
## and then uses a file from the state of Maryland which connects zip codes to counties
## to join with and get a total for the number of properties in each zip code in Maryland.

## Set up context.
conf=SparkConf()
sc = SparkContext(conf=conf)
sc.setLogLevel("WARN")
spark = SparkSession(sc)

## Used the AWS CLI to download this file to the local directory.
## aws s3 cp s3://first-street-climate-risk-statistics-for-noncommercial-use/v1.1/Zip_level_risk_FEMA_FSF_v1.1.csv . --no-sign-request
## This file is made by First Street Foundation to catalogue flood risk.
## But here I just use it to get a total of the number of properties per zip code in the USA.
## Put the first two columns, "zip" and "count_property" into the dataframe fdf,
## casting string to int in the latter case.

fdftemp = spark.read.format("csv").option("header","true").load("Zip_level_risk_FEMA_FSF_v1.1.csv")
fdf = fdftemp.select(col("zcta5ce").alias("zip"),col("count_property").cast("int"))
fdf.printSchema()

## Get a table where each zip code in Maryland is connected to its city and county, copy to local directory.
## https://opendata.maryland.gov/api/views/ryxx-aeaf/rows.csv?accessType=DOWNLOAD
## Use only county data. Put zip and County into a dataframe mcdf.
mcdftemp = spark.read.format("csv").option("header","true").load("Zip_Code_Lookup_Table.csv")
mcdf = mcdftemp.select(col("Zip Code").alias("zip"),"County")
mcdf.printSchema()

## Join the two dataframes together on the zip code.
## Select only County and count_property.
md_prop_county = mcdf.join(fdf, mcdf.zip==fdf.zip).select("County",fdf.count_property)

## Groupby and sum on County to get total properties in each county.
md_prop_county.groupby("County").sum("count_property").sort("County").show()






