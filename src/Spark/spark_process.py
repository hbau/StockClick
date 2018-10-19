from pyspark import SparkContext
from pyspark.sql import SQLContext
from pyspark.sql.types import DoubleType, StringType
from pyspark.sql.functions import udf
from pyspark.sql import functions as F
from pyspark.sql import DataFrame
from pyspark.conf import SparkConf
import preprocess as pp

def sparkProcess():

    # define pyspark functions to do NLP 
    remove_stops_udf = udf(pp.remove_stops, StringType())
    remove_features_udf = udf(pp.remove_features, StringType())
    tag_and_remove_udf = udf(pp.tag_and_remove, StringType())
    lemmatize_udf = udf(pp.lemmatize, StringType())

    # create pyspark objects
    conf = SparkConf().setAppName("stocktwits")
    sc = SparkContext(conf=conf)
    sqlContext = SQLContext(sc)

    # load stock files from s3
    stocks = ["AAPL", "AMD", "AMZN", "BAC", "BBY", "BIDU", "CRM", "EBAY", "FB", "GOOGL", "TSLA", "GS", "IBM", "JPM", "MSFT", "NFLX", "SINA", "TSLA"]
    for i in range(len(stocks)):
        stocks[i] = sqlContext.read.load("s3n://stocktwits/%s.csv" % i), 
                                format='com.databricks.spark.csv', 
                                header='true', 
                                inferSchema='true')

    # concat all dataframes from different stock symbols
    def unionAll(*dfs):
        return reduce(DataFrame.unionAll, dfs)
    for i in stocks:
        data_df = data_df.unionAll(i)

    # "bearish" = 0, "bullish" = 1, no label = 5
    data_df = data_df.withColumn('label', F.when(data_df.Sentiment == "Bearish", 0)\
                            .when(data_df.Sentiment == "Bullish", 1)\
                            .otherwise(5))

    # filter labeled records 
    data_df = data_df.where((data_df.label == 0.0) | (data_df.label == 1.0))

    # remove stop words to reduce dimensionality
    rm_stops_df = data_df.withColumn("stop_text", remove_stops_udf(data_df["Body"]))

    # remove other non essential words, think of it as my personal stop word list
    rm_features_df = rm_stops_df.withColumn("feat_text", remove_features_udf(rm_stops_df["stop_text"]))

    # tag the words remaining and keep only Nouns, Verbs and Adjectives
    tagged_df = rm_features_df.withColumn("tagged_text", tag_and_remove_udf(rm_features_df["feat_text"]))

    # lemmatization of remaining words to reduce dimensionality & boost measures
    lemm_df = tagged_df.withColumn("text", lemmatize_udf(tagged_df["tagged_text"]))

    # dedupe important since alot of the tweets only differed by url's and RT mentions
    dedup_df = lemm_df.dropDuplicates(['Body', 'label'])

    # select only the columns we care about
    cleanData_df = dedup_df.select(dedup_df['ID'], dedup_df['Date'], dedup_df['Symbol'], dedup_df['text'], dedup_df['label'])

    bearishCount = cleanData_df.filter(cleanData_df.label == 0.0).count()
    bullishCount = cleanData_df.filter(cleanData_df.label == 1.0).count()
    print("Total Bearish Tags = %g" % bearishCount)
    print("Total Bullish Tags = %g" % bullishCount)

    return cleanData_df

