from pyspark.sql import SparkSession
from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.classification import LogisticRegression

# Create a SparkSession
spark = SparkSession.builder.appName("SparkEDA").getOrCreate()

# Read in data
df = spark.read.csv("path/to/data.csv", header=True, inferSchema=True)

# Perform EDA
print("Data shape:", (df.count(), len(df.columns)))
df.describe().show()

# Feature Selection and Transformation
# Assemble features into a single vector column
assembler = VectorAssembler(inputCols=["feature1", "feature2", "feature3"], outputCol="features")
df = assembler.transform(df)

# Scale the features
scaler = StandardScaler(inputCol="features", outputCol="scaled_features")
scaler_model = scaler.fit(df)
df = scaler_model.transform(df)

# Split the data into training and test sets
train, test = df.randomSplit([0.8, 0.2], seed=42)

# Define the logistic regression model and set the parameters
lr = LogisticRegression(labelCol="response_variable", featuresCol="scaled_features")

# Build a pipeline
pipeline = Pipeline(stages=[lr])

# Train the model
model = pipeline.fit(train)

# Test the model on the test set
results = model.transform(test)

# Evaluate the model
from pyspark.ml.evaluation import BinaryClassificationEvaluator

evaluator = BinaryClassificationEvaluator(labelCol="response_variable")
print("Area under ROC:", evaluator.evaluate(results, {evaluator.metricName: "areaUnderROC"}))

# Stop the SparkSession
spark.stop()
