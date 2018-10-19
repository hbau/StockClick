### predict and aggregate new daily stock sentiment scores ###

import boto3
s3 = boto3.resource('s3')
today_str=datetime.today().strftime('%Y%m%d')
seq=datetime.today().strftime('%H%M%S')
pred_df = []
stocks = ["AAPL", "AMD", "AMZN", "BAC", "BBY", "BIDU", "CRM", "EBAY", "FB", "GOOGL", "TSLA", "GS", "IBM", "JPM", "MSFT", "NFLX", "SINA", "TSLA"]

def spark_prediction():
    
    for stock in stocks:
        file = stock+"."+today_str+"."+seq+".csv"
        pred_df = pred_df.append(s3.Bucket('stocktwitsdata').download_file("../../Data/StockTwits/"+file, file))

    # preprocess new daily stock data
    pred_df = pred_df.withColumn('label', F.when(pred_df.Sentiment == "Bearish", 0)\
                            .when(pred_df.Sentiment == "Bullish", 1)\
                    .otherwise(5))

    # remove stop words to reduce dimensionality
    rm_stops_df = pred_df.withColumn("stop_text", remove_stops_udf(pred_df["Body"]))

    # remove other non essential words, think of it as my personal stop word list
    rm_features_df = rm_stops_df.withColumn("feat_text", remove_features_udf(rm_stops_df["stop_text"]))

    # tag the words remaining and keep only Nouns, Verbs and Adjectives
    tagged_df = rm_features_df.withColumn("tagged_text", tag_and_remove_udf(rm_features_df["feat_text"]))

    # lemmatization of remaining words to reduce dimensionality & boost measures
    lemm_df = tagged_df.withColumn("text", lemmatize_udf(tagged_df["tagged_text"]))

    # dedupe important since alot of the tweets only differed by url's and RT mentions
    dedup_df = lemm_df.dropDuplicates(['Body', 'label'])

    # select only the columns we care about
    cleanData_df = dedup_df.select(dedup_df['CreateTime'], dedup_df['Date'], dedup_df['Symbol'], dedup_df['text'], dedup_df['label'])

    # Predict on full data set using the trained cross-validated best NB model
    result = saved_cvModel.transform(cleanData_df)

    prediction_df = result.select("CreateTime", "Symbol", "text", "label", "probability", "prediction")

    # separate nb model classification probability from one dense vector to 2 diff. features [bearish_probability, bullish_probability ]
    from pyspark.sql.functions import udf
    from pyspark.sql.types import FloatType

    bearish = udf(lambda v:float(v[0]),FloatType())
    bullish = udf(lambda v:float(v[1]),FloatType())
    bearish_probability = prediction_df.select(bearish('probability'))
    bullish_probability = prediction_df.select(bullish('probability'))

    # merge separated spark dataframes into 1 and drop the original 'probability' dense vector
    from pyspark.sql.functions import monotonically_increasing_id

    bearish_probability = bearish_probability.withColumn("id", monotonically_increasing_id()).withColumnRenamed("<lambda>(probability)", "bearish_probability")
    bullish_probability = bullish_probability.withColumn("id", monotonically_increasing_id()).withColumnRenamed("<lambda>(probability)", "bullish_probability")
    joined = bearish_probability.join(bullish_probability, ["id"], 'outer').sort("id")

    prediction_df = prediction_df.withColumn("id", monotonically_increasing_id()).join(joined, ["id"], 'outer').sort("id")
    prediction_df = prediction_df.drop('probability')

    # calculate sentiment score based on bearish/bullish probabilities
    prediction_df = prediction_df.withColumn('score', F.when(prediction_df.bullish_probability <= 0.3, prediction_df.bullish_probability -1)\
                                            .when(prediction_df.bullish_probability <= 0.5, prediction_df.bullish_probability - 0.5)\
                                            .when(prediction_df.bullish_probability <= 0.7, prediction_df.bullish_probability - 0.5)\
                                            .otherwise(prediction_df.bullish_probability))

    # Write data to csv for post-analysis
    prediction_pd = prediction_df.toPandas()

return prediction_pd
