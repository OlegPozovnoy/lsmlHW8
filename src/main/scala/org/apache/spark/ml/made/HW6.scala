package org.apache.spark.ml.made

import breeze.linalg.normalize
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.feature.{HashingTF, IDF, RegexTokenizer}
import org.apache.spark.ml.linalg.{BLAS, Vector, Vectors}
import org.apache.spark.sql.{SparkSession, functions}
import org.apache.spark.sql.functions.{col, lit, monotonically_increasing_id}

object HW6 extends App {
  override def main(args: Array[String]): Unit = {

    val spark = SparkSession.builder()
      .master("local[2]")
      .appName("made-demo")
      .getOrCreate()

    import spark.implicits._

    val df = spark.read
      .option("inferSchema","true")
      .option("header","true")
      .csv("./tripadvisor_hotel_reviews.csv").sample(0.1)


    df.show()

    val preprocessingPipe = new Pipeline()
      .setStages(Array(
        new RegexTokenizer()
          .setInputCol("Review")
          .setOutputCol("tokenized")
          .setPattern("\\W+"),
        new HashingTF()
          .setInputCol("tokenized")
          .setOutputCol("tf")
          .setBinary(true)
          .setNumFeatures(10000),
        new HashingTF()
          .setInputCol("tokenized")
          .setOutputCol("tf2")
          .setNumFeatures(10000),
        new IDF()
          .setInputCol("tf2")
          .setOutputCol("tfidf")
      ))

    val Array(train, test) = df.randomSplit(Array(0.8,0.2))

    val pipe = preprocessingPipe.fit(train)

    val trainFeatures = pipe.transform(train)
    val testFeatures = pipe.transform(test)

    trainFeatures.printSchema
    print(trainFeatures.show(20))

    val testFeaturesWithIndex = testFeatures.withColumn("id", monotonically_increasing_id()).cache()

    testFeaturesWithIndex.show

    import org.apache.spark.ml.evaluation.RegressionEvaluator

    val metrics = new RegressionEvaluator()
      .setLabelCol("Rating")
      .setPredictionCol("predict")
      .setMetricName("rmse")

    val Cosine = new CosineLSH()
      .setInputCol("tfidf")
      .setOutputCol("brpBuckets")
      .setNumHashTables(5)

    val mhModel3 = Cosine.fit(trainFeatures)

    trainFeatures.show

    val eqlidNeigh3 = mhModel3.approxSimilarityJoin(trainFeatures, testFeaturesWithIndex,0.7)

    eqlidNeigh3.count()
    eqlidNeigh3.show()


    val predictions3 = eqlidNeigh3
      .withColumn("similarity", lit(1) / (col("distCol") + lit(0.000001)))
      .groupBy("datasetB.id")
      .agg((functions.sum(col("similarity") *col("datasetA.Rating"))/ functions.sum(col("similarity"))).as("predict"))

    val forMetric3 = testFeaturesWithIndex.join(predictions3, Seq("id"))

    println(metrics.evaluate(forMetric3))

  }
}
