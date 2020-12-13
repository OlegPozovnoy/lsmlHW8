package org.apache.spark.ml.made

import scala.util.Random
import breeze.linalg.normalize
import org.apache.hadoop.fs.Path
import org.apache.spark.annotation.Since
import org.apache.spark.ml.feature.{LSH, LSHModel}
import org.apache.spark.ml.linalg._
import org.apache.spark.ml.param._
import org.apache.spark.ml.param.shared.HasSeed
import org.apache.spark.ml.util._
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.sql.Row
import org.apache.spark.sql.types.StructType

/// Я так и не понял, как мы можем генерировать плоскость векторами из 1 и -1, плоскость задается вектором нормали
// то по какую сторону вектор от плоскости определяется знаком скалярного произведения нормали и вектора
// вот знаки это действительно -1 и 1


@Since("2.1.0")
class CosineLSHModel private[ml](
                                                    override val uid: String,
                                                    private[ml] val randUnitVectors: Array[Vector])
  extends LSHModel[CosineLSHModel] {

  /** @group setParam */
  @Since("2.4.0")
  override def setInputCol(value: String): this.type = super.set(inputCol, value)

  /** @group setParam */
  @Since("2.4.0")
  override def setOutputCol(value: String): this.type = super.set(outputCol, value)

  @Since("2.1.0")
  override protected[ml] def hashFunction(elems: Vector): Array[Vector] = {
    val hashValues = randUnitVectors.map(
      randUnitVector => Math.signum(BLAS.dot(elems, randUnitVector)/ Vectors.norm(elems,2) )
    )
    // TODO: Output vectors of dimension numHashFunctions in SPARK-18450
    hashValues.map(Vectors.dense(_))
  }

  @Since("2.1.0")
  override protected[ml] def keyDistance(x: Vector, y: Vector): Double = {
    Math.acos(BLAS.dot(x, y)/ Vectors.norm(x,2) * Vectors.norm(y,2))/Math.PI
  }

  @Since("2.1.0")
  override protected[ml] def hashDistance(x: Seq[Vector], y: Seq[Vector]): Double = {
    // Since it's generated by hashing, it will be a pair of dense vectors.
    val tmp = x.zip(y).map(vectorPair =>
      vectorPair._1.toArray.zip(vectorPair._2.toArray).count(pair => pair._1 != pair._2))

    tmp.sum.toDouble/tmp.length
    //x.zip(y).map(vectorPair => Vectors.sqdist(vectorPair._1, vectorPair._2)).min
  }

  @Since("2.1.0")
  override def copy(extra: ParamMap): CosineLSHModel = {
    val copied = new CosineLSHModel(uid, randUnitVectors).setParent(parent)
    copyValues(copied, extra)
  }

  @Since("2.1.0")
  override def write: MLWriter = {
    new CosineLSHModel.CosineLSHModelWriter(this)
  }

  @Since("3.0.0")
  override def toString: String = {
    s"CosineLSHModel: uid=$uid, numHashTables=${$(numHashTables)}"
  }
}

/////////////////////////////////////////////////////////////////////////////////////////////
@Since("2.1.0")
class CosineLSH(override val uid: String)
  extends LSH[CosineLSHModel]
     with HasSeed {

  @Since("2.1.0")
  override def setInputCol(value: String): this.type = super.setInputCol(value)

  @Since("2.1.0")
  override def setOutputCol(value: String): this.type = super.setOutputCol(value)

  @Since("2.1.0")
  override def setNumHashTables(value: Int): this.type = super.setNumHashTables(value)

  @Since("2.1.0")
  def this() = {
    this(Identifiable.randomUID("brp-lsh"))
  }

  /** @group setParam */
  @Since("2.1.0")
  def setSeed(value: Long): this.type = set(seed, value)

  @Since("2.1.0")
  override protected[this] def createRawLSHModel(
                                                  inputDim: Int): CosineLSHModel = {
    val rand = new Random($(seed))
    val randUnitVectors: Array[Vector] = {
      Array.fill($(numHashTables)) {
        val randArray = Array.fill(inputDim)(rand.nextGaussian())
        Vectors.fromBreeze(normalize(breeze.linalg.Vector(randArray)))
      }
    }
    new CosineLSHModel(uid, randUnitVectors)
  }

  @Since("2.1.0")
  override def transformSchema(schema: StructType): StructType = {
    SchemaUtils.checkColumnType(schema, $(inputCol), new VectorUDT)
    validateAndTransformSchema(schema)
  }

  @Since("2.1.0")
  override def copy(extra: ParamMap): this.type = defaultCopy(extra)
}

@Since("2.1.0")
object CosineLSH extends DefaultParamsReadable[CosineLSH] {

  @Since("2.1.0")
  override def load(path: String): CosineLSH = super.load(path)
}

/////////////////////////////////////////////////////////////////////////////////////////////

@Since("2.1.0")
object CosineLSHModel extends MLReadable[CosineLSHModel] {

  @Since("2.1.0")
  override def read: MLReader[CosineLSHModel] = {
    new CosineLSHModelReader
  }

  @Since("2.1.0")
  override def load(path: String): CosineLSHModel = super.load(path)

  private[CosineLSHModel] class CosineLSHModelWriter(
                                                                                          instance: CosineLSHModel) extends MLWriter {

    // TODO: Save using the existing format of Array[Vector] once SPARK-12878 is resolved.
    private case class Data(randUnitVectors: Matrix)

    override protected def saveImpl(path: String): Unit = {
      DefaultParamsWriter.saveMetadata(instance, path, sc)
      val numRows = instance.randUnitVectors.length
      require(numRows > 0)
      val numCols = instance.randUnitVectors.head.size
      val values = instance.randUnitVectors.map(_.toArray).reduce(Array.concat(_, _))
      val randMatrix = Matrices.dense(numRows, numCols, values)
      val data = Data(randMatrix)
      val dataPath = new Path(path, "data").toString
      sparkSession.createDataFrame(Seq(data)).repartition(1).write.parquet(dataPath)
    }
  }

  private class CosineLSHModelReader
    extends MLReader[CosineLSHModel] {

    /** Checked against metadata when loading model */
    private val className = classOf[CosineLSHModel].getName

    override def load(path: String): CosineLSHModel = {
      val metadata = DefaultParamsReader.loadMetadata(path, sc, className)

      val dataPath = new Path(path, "data").toString
      val data = sparkSession.read.parquet(dataPath)
      val Row(randUnitVectors: Matrix) = MLUtils.convertMatrixColumnsToML(data, "randUnitVectors")
        .select("randUnitVectors")
        .head()
      val model = new CosineLSHModel(metadata.uid,
        randUnitVectors.rowIter.toArray)

      metadata.getAndSetParams(model)
      model
    }
  }
}