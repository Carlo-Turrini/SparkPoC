package udaf_demo

import org.apache.spark.sql.Row
import org.apache.spark.sql.expressions.{MutableAggregationBuffer, UserDefinedAggregateFunction}
import org.apache.spark.sql.types.{ArrayType, DataType, DoubleType, StructType}

import scala.collection.mutable
import scala.collection.mutable.ArrayBuffer

object GroupArray extends UserDefinedAggregateFunction {
  override def inputSchema: StructType = new StructType().add("scaled_features", ArrayType(DoubleType))

  override def bufferSchema: StructType = new StructType().add("buff", ArrayType(ArrayType(DoubleType)))

  override def dataType: DataType = ArrayType(ArrayType(DoubleType))

  override def deterministic: Boolean = true

  override def initialize(buffer: MutableAggregationBuffer): Unit = {
    buffer.update(0, ArrayBuffer.empty[Array[Double]])
  }

  override def update(buffer: MutableAggregationBuffer, input: Row): Unit = {
    if(!input.isNullAt(0))
      buffer.update(0, buffer.getSeq[Array[Double]](0) :+ input.getSeq[Double](0))
  }

  override def merge(buffer1: MutableAggregationBuffer, buffer2: Row): Unit = {
    buffer1.update(0, buffer1.getSeq[Array[Double]](0) ++ buffer2.getSeq[Array[Double]](0))
  }

  private def sortByTimestamp(arr1: mutable.WrappedArray[Double], arr2: mutable.WrappedArray[Double]) = {
    arr1(0) < arr2(0)
  }

  private def mapToFeatures(arr: mutable.WrappedArray[Double]) = {
    arr.drop(1)
  }

  override def evaluate(buffer: Row): Any = {
    buffer.getSeq[mutable.WrappedArray[Double]](0).sortWith(sortByTimestamp).map(mapToFeatures)
  }
}
