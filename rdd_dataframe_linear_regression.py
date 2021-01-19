from pyspark.sql import SparkSession
from pyspark import SparkContext, SparkConf
from pyspark.ml.linalg import Vectors
from pyspark.ml.feature import VectorAssembler
import pyspark.ml
from pyspark.ml.regression import LinearRegression
from pyspark.ml.regression import GeneralizedLinearRegression

## This script works through very basic data cleaning in an RDD then
## changes to a dataframe to use Spark's ML libraries and compute 
## a linear regression and prediction.
## This makes predictions on the kaggle.com Titanic data set.
## https://www.kaggle.com/c/titanic/overview
## Multiple linear regression is not the best algorithm for this classification task,
## but the purpose is to show the use of pyspark.


## Set up context.
conf=SparkConf()
sc = SparkContext(conf=conf)
sc.setLogLevel("WARN")
spark = SparkSession(sc)

## Process the training set, which has been downloaded.
train_rdd = sc.textFile("kaggle/train.csv")                                 # read in the file
header = train_rdd.first()                                                  # header is first row
train_rdd = train_rdd.filter(lambda r: r!=header)                           # remove the header
train_rdd = train_rdd.map(lambda s: s.replace('""',''))                     # get rid of double quotes
train_rdd = train_rdd.map(lambda s: s.replace('female','1'))                # encode female as 1 for regression
train_rdd = train_rdd.map(lambda s: s.replace('male','0'))                  # encode male as 0 for regression
train_rdd = train_rdd.map(lambda x: x.split("\""))                          # split based on quotes
train_rdd = train_rdd.map(lambda x: [x[i] for i in [0,2]])                  # ignore the person's name
train_rdd = train_rdd.map(lambda x: str(x[0])+str(x[1]))                    # make it one string
train_rdd = train_rdd.map(lambda s: s.replace(',,',',0,'))                  # put 0 when text is empty
train_rdd = train_rdd.map(lambda x: x.split(","))                           # split based on comma
train_rdd = train_rdd.map(lambda x: [float(x[i])*1.0 for i in [1,4,5,9]])   # 1=survived(target), 4=sex, 5=age, 9=fare
train_df = train_rdd.toDF()                                                 # make it a dataframe 
#train_df.show()

## Prepare a vector assembler to put together the feature vector.
assembler = VectorAssembler(inputCols=["_2","_3","_4"], outputCol="features")
output = assembler.transform(train_df)                                      # output is now the input to train the model
#output.show()

## Train the model.
lin_reg = LinearRegression(featuresCol='features',labelCol='_1',predictionCol='prediction')
lin_reg_model = lin_reg.fit(output)


## Now process the test data in the same way. 
## It's the same format but without the target column.
test_rdd = sc.textFile("kaggle/test.csv")                                    # read in the file
header = test_rdd.first()                                                    # header is first row
test_rdd = test_rdd.filter(lambda r: r!=header)                              # remove header
test_rdd = test_rdd.map(lambda s: s.replace('""',''))                        # get rid of double quotes
test_rdd = test_rdd.map(lambda s: s.replace('female','1'))                   # encode female as 1 for regression
test_rdd = test_rdd.map(lambda s: s.replace('male','0'))                     # encode male as 0 for regression
test_rdd = test_rdd.map(lambda s: s.replace(',,',',0,'))                     # put 0 when value missing
test_rdd = test_rdd.map(lambda x: x.split("\""))                             # split based on quotes
test_rdd = test_rdd.map(lambda x: [x[i] for i in [2]])                       # ignore the person's name
test_rdd = test_rdd.map(lambda x: x[0])                                      # make it a string
test_rdd = test_rdd.map(lambda x: x.split(","))                              # split on comma
test_rdd = test_rdd.map(lambda x: [float(x[i])*1.0 for i in [1,2,6]])        # take sex, age, and fare
test_df = test_rdd.toDF()                                                    # make it a dataframe
#test_df.show()

## Put together the feature vectors to get predictions. 
assembler2 = VectorAssembler(inputCols=["_1","_2","_3"], outputCol="features")
testin = assembler2.transform(test_df)
result = lin_reg_model.evaluate(testin)

predictions = lin_reg_model.transform(testin.select('features'))
#predictions.show()

## This is the output with the predictions.
predictions.toPandas().to_csv('preds.csv')

## These predictions got a score of 0.76, which is the same that I got when I did this in R.


