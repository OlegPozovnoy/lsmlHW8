package org.apache.spark.ml.made

import breeze.linalg.normalize
import org.apache.spark.ml.linalg.{BLAS, Vector, Vectors}

import scala.util.Random

object HW6 extends App {
  override def main(args: Array[String]): Unit = {

    val rand = new Random(0)
    /////////////////////////////////////////////////////////////////////////////////////////
    var elems1: Vector = Vectors.dense(1, 0, 1)
    var elems2: Vector = Vectors.dense(1, 1, 0)

    val randUnitVectors: Array[Vector] = {
      Array.fill(2000) {
        val randArray = Array.fill(3)(rand.nextGaussian())
        Vectors.fromBreeze(normalize(breeze.linalg.Vector(randArray)))
      }
    }
    println("Randunitvecc")
    println(randUnitVectors.deep.mkString("\n"))

    var hashValues1 = randUnitVectors.map(
      randUnitVector => Math.signum(BLAS.dot(elems1, randUnitVector)/ Vectors.norm(elems1,2) )
    )

    var hashValues2 = randUnitVectors.map(
      randUnitVector =>Math.signum(BLAS.dot(elems2, randUnitVector)/ Vectors.norm(elems2,2) )
    )


    var x1: Seq[Vector] = (hashValues1.map(Vectors.dense(_)))
    var y1: Seq[Vector] = (hashValues2.map(Vectors.dense(_)))

    println("vectors")
    println(x1)
    println(y1)

    val distance  =  x1.zip(y1).map(vectorPair =>
      vectorPair._1.toArray.zip(vectorPair._2.toArray).count(pair => pair._1 != pair._2))
    println("distance")
    println(distance)


    println(distance.sum.toDouble/distance.length)

  }
}
