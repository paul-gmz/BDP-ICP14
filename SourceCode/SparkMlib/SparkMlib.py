import os

from pyspark.ml.classification import DecisionTreeClassifier
from pyspark.ml.classification import NaiveBayes
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.clustering import KMeans
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import LinearRegression
from pyspark.sql import *
from pyspark.sql.functions import col, when
from pyspark.ml.classification import LogisticRegression
from pyspark.sql.types import IntegerType

os.environ["SPARK_HOME"] = "C:\Installations\spark-2.4.5-bin-hadoop2.7"
os.environ["HADOOP_HOME"] = "C:\Installations\Hadoop"
os.environ["PYSPARK_SUBMIT_ARGS"]  = ("--packages  graphframes:graphframes:0.5.0-spark2.0-s_2.11 pyspark-shell")

spark = SparkSession \
    .builder \
    .appName("SparkMlib") \
    .getOrCreate()

# loading adult data
adult_df = spark.read.load(r"adult.csv", format="csv", header=True, delimiter=",")

# Changing datatypes to int
adult_df = adult_df.withColumn("age", adult_df["age"].cast(IntegerType()))
adult_df = adult_df.withColumn("fnlwgt", adult_df["fnlwgt"].cast("integer"))
adult_df = adult_df.withColumn("educational-num", adult_df["educational-num"].cast(IntegerType()))
adult_df = adult_df.withColumn("capital-gain", adult_df["capital-gain"].cast(IntegerType()))
adult_df = adult_df.withColumn("capital-loss", adult_df["capital-loss"].cast(IntegerType()))
adult_df = adult_df.withColumn("hours-per-week", adult_df["hours-per-week"].cast(IntegerType()))
adult_df = adult_df.withColumn("label", adult_df['hours-per-week'] - 0)

# creates vector assembler to transform the dataset
assembler = VectorAssembler(inputCols=adult_df.columns[10:13], outputCol='features')
data = assembler.transform(adult_df)
data.show(5)

# creates train & test datasets
train, test = data.randomSplit([0.6, 0.4], 1234)

# Naive bayes
nb = NaiveBayes(smoothing=1.0, modelType="multinomial")

# trains the model on train data set
model_nb = nb.fit(train)

predictions = model_nb.transform(test)
predictions.show(3)
evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction",
                                              metricName="accuracy")
accuracy = evaluator.evaluate(predictions)
print("Test set accuracy: " + str(accuracy))


# decision tree
dt = DecisionTreeClassifier(labelCol="label", featuresCol="features")

# trains the model on train data set
model_dt = dt.fit(train)

predictions = model_dt.transform(test)
predictions.show(3)

# computes accuracy on test dataset
evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction",
                                              metricName="accuracy")
accuracy = evaluator.evaluate(predictions)
print("Test set accuracy = " + str(accuracy))


# Random forest

rf = RandomForestClassifier(labelCol="label", featuresCol="features", numTrees=10)

# trains the model
model_rf = rf.fit(train)

predictions = model_rf.transform(test)
predictions.show(3)

# computes accuracy on test dataset
evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction",
                                              metricName="accuracy")
accuracy = evaluator.evaluate(predictions)
print("Test set accuracy = " + str(accuracy))


# k-means clustering

# Loads data and selects feature
diabetic_df = spark.read.format("csv").option("header", True).option("inferSchema", True).option("delimiter", ",").load("diabetic_data.csv")
diabetic_df = diabetic_df.select("admission_type_id", "discharge_disposition_id", "admission_source_id", "time_in_hospital", "num_lab_procedures")

# creates vector assembler to transform the dataset
assembler = VectorAssembler(inputCols=diabetic_df.columns, outputCol="features")
data = assembler.transform(diabetic_df)

# Trains a k-means model.
kmeans = KMeans().setK(2).setSeed(1)

model_kmeans = kmeans.fit(data)

# Make predictions
predictions = model_kmeans.transform(data)

# Shows the result.
centers = model_kmeans.clusterCenters()
print("Cluster Centers: ")
for center in centers:
    print(center)





# Loads data and selects feature
auto_df = spark.read.format("csv").option("header", True).option("inferSchema", True).option("delimiter", ",").load("imports-85.csv")

# Linear Regression
linear_df = auto_df.withColumnRenamed("wheel-base", "label").select("label", "length", "width", "height")

assembler = VectorAssembler(inputCols=linear_df.columns[1:], outputCol="features")
data = assembler.transform(linear_df)

lr = LinearRegression(maxIter=10, regParam=0.3, elasticNetParam=0.8)
model_lr = lr.fit(data)

print("Coefficients: %s" % str(model_lr.coefficients))
print("Intercept: %s" % str(model_lr.intercept))

# summarization on training set and some metrics
trainingSummary = model_lr.summary
print("numIterations: %d" % trainingSummary.totalIterations)
print("objectiveHistory: %s" % str(trainingSummary.objectiveHistory))
trainingSummary.residuals.show()
print("RMSE: %f" % trainingSummary.rootMeanSquaredError)
print("r2: %f" % trainingSummary.r2)

# Logistic regression

logistic_df = auto_df.withColumn("label", when(col("num-of-doors") == "four", 1).otherwise(0)).select("label", "length", "width", "height")

assembler = VectorAssembler(inputCols=logistic_df.columns[1:], outputCol="features")
data = assembler.transform(logistic_df)

logR = LogisticRegression(maxIter=10, regParam=0.3, elasticNetParam=0.8)

# Fits the model
model_logR = logR.fit(data)

print("Coefficients: " + str(model_logR.coefficients))
print("Intercept: " + str(model_logR.intercept))

mLogR = LogisticRegression(maxIter=10, regParam=0.3, elasticNetParam=0.8, family="multinomial")

# Fits the multinomial model
model_mLogR = mLogR.fit(data)

print("Multinomial coefficients: " + str(model_mLogR.coefficientMatrix))
print("Multinomial intercepts: " + str(model_mLogR.interceptVector))
