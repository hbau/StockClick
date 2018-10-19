from sklearn.model_selection import train_test_split
import spark_process

# split cleaned dataframe into training and test sets
training, test = spark_process.randomSplit([0.70, 0.30], seed=1)

### Spark ML training and testing ###
from pyspark.ml.feature import HashingTF, IDF, Tokenizer
from pyspark.ml import Pipeline
from pyspark.ml.classification import NaiveBayes
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml import Pipeline
from pyspark.ml import PipelineModel

# Configure an ML pipeline, which consists of three stages: tokenizer, hashingTF, and nb.
tokenizer = Tokenizer(inputCol="text", outputCol="words")
hashingTF = HashingTF(inputCol=tokenizer.getOutputCol(), outputCol="features")
idf = IDF(inputCol="features", outputCol="idf")
nb = NaiveBayes(smoothing=1.0, modelType="multinomial")
pipeline = Pipeline(stages=[tokenizer, hashingTF, idf, nb])

paramGrid = ParamGridBuilder().addGrid(nb.smoothing, [0.0, 1.0]).addGrid(hashingTF.numFeatures, [100, 1000, 10000]).build()

cv = CrossValidator(estimator=pipeline, 
                    estimatorParamMaps=paramGrid, 
                    evaluator=MulticlassClassificationEvaluator(), 
                    numFolds=4)

cvModel = cv.fit(training).bestModel
cvModel.save('trained_model')

saved_cvModel = PipelineModel.load('trained_model')

# evaluation model with test set
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
result = saved_cvModel.transform(test)
prediction_df_test = result.select("Date", "text", "label", "probability", "prediction")
evaluator = MulticlassClassificationEvaluator(predictionCol="prediction")
print("Test Set Accuracy = %g" % evaluator.evaluate(result, {evaluator.metricName: "accuracy"}))
print("Test Set F1 Score = %g" % evaluator.evaluate(result, {evaluator.metricName: "f1"}))
print("Test Set weightedPrecision = %g" % evaluator.evaluate(result, {evaluator.metricName: "weightedPrecision"}))
print("Test Set weightedRecall = %g" % evaluator.evaluate(result, {evaluator.metricName: "weightedRecall"}))


