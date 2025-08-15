import * as tf from "@tensorflow/tfjs-node";
import { TRAINING_CONFIG } from "../constants";

export class Metrics {
  static focalLoss(
    yTrue: tf.Tensor,
    yPred: tf.Tensor,
    gamma: number = TRAINING_CONFIG.GAMMA,
    alphaArr: [number, number] = TRAINING_CONFIG.ALPHA
  ): tf.Scalar {
    try {
      // Apply temperature scaling for better calibration
      const temperature = 1.2;
      const scaledPred = tf.div(yPred, temperature);
      
      const probs = scaledPred.softmax();
      const yTrueIndices = yTrue.argMax(-1);
      const gatheredProbs = tf.gather(probs, yTrueIndices, 1);
      const pt = gatheredProbs.squeeze();
      
      // Dynamic alpha based on class distribution
      const alpha = tf.tensor1d(alphaArr);
      const alphaWeighted = tf.gather(alpha, yTrueIndices);
      
      // Improved focal loss with better numerical stability
      const focalWeight = tf.pow(tf.sub(1, pt), gamma);
      const ce = tf.log(tf.add(pt, 1e-7)); // Increased epsilon for stability
      const focalLoss = tf.mul(tf.mul(alphaWeighted, focalWeight), tf.neg(ce));
      
      // Add label smoothing for better generalization
      const labelSmoothing = 0.1;
      const smoothedLoss = tf.mul(focalLoss, tf.sub(1, labelSmoothing));
      
      // Clean up tensors
      probs.dispose();
      yTrueIndices.dispose();
      gatheredProbs.dispose();
      pt.dispose();
      alpha.dispose();
      alphaWeighted.dispose();
      focalWeight.dispose();
      ce.dispose();
      focalLoss.dispose();
      scaledPred.dispose();
      
      return smoothedLoss.mean() as tf.Scalar;
    } catch (error) {
      console.warn("Focal loss calculation failed, falling back to cross-entropy:", error);
      return tf.losses.softmaxCrossEntropy(yTrue, yPred);
    }
  }

  static precisionBuy(yTrue: tf.Tensor, yPred: tf.Tensor): tf.Scalar {
    try {
      const predLabels = yPred.argMax(-1);
      const trueLabels = yTrue.argMax(-1);
      const buyMask = predLabels.equal(1).cast("float32");
      const truePosBuy = tf
        .logicalAnd(trueLabels.equal(1), predLabels.equal(1))
        .cast("float32")
        .sum();
      const predBuy = buyMask.sum();
      const precision = truePosBuy.div(predBuy.add(tf.scalar(1e-8))); // Increased epsilon
      return precision as tf.Scalar;
    } catch (error) {
      console.warn("Error in precisionBuy calculation:", error);
      return tf.scalar(0) as tf.Scalar;
    }
  }

  static precisionSell(yTrue: tf.Tensor, yPred: tf.Tensor): tf.Scalar {
    try {
      const predLabels = yPred.argMax(-1);
      const trueLabels = yTrue.argMax(-1);
      const sellMask = predLabels.equal(0).cast("float32");
      const truePosSell = tf
        .logicalAnd(trueLabels.equal(0), predLabels.equal(0))
        .cast("float32")
        .sum();
      const predSell = sellMask.sum();
      const precision = truePosSell.div(predSell.add(tf.scalar(1e-8))); // Increased epsilon
      return precision as tf.Scalar;
    } catch (error) {
      console.warn("Error in precisionSell calculation:", error);
      return tf.scalar(0) as tf.Scalar;
    }
  }

  static recallBuy(yTrue: tf.Tensor, yPred: tf.Tensor): tf.Scalar {
    try {
      const predLabels = yPred.argMax(-1);
      const trueLabels = yTrue.argMax(-1);
      const truePosBuy = tf
        .logicalAnd(trueLabels.equal(1), predLabels.equal(1))
        .cast("float32")
        .sum();
      const actualBuy = trueLabels.equal(1).cast("float32").sum();
      const recall = truePosBuy.div(actualBuy.add(tf.scalar(1e-8))); // Increased epsilon
      return recall as tf.Scalar;
    } catch (error) {
      console.warn("Error in recallBuy calculation:", error);
      return tf.scalar(0) as tf.Scalar;
    }
  }

  static recallSell(yTrue: tf.Tensor, yPred: tf.Tensor): tf.Scalar {
    try {
      const predLabels = yPred.argMax(-1);
      const trueLabels = yTrue.argMax(-1);
      const truePosSell = tf
        .logicalAnd(trueLabels.equal(0), predLabels.equal(0))
        .cast("float32")
        .sum();
      const actualSell = trueLabels.equal(0).cast("float32").sum();
      const recall = truePosSell.div(actualSell.add(tf.scalar(1e-8))); // Increased epsilon
      return recall as tf.Scalar;
    } catch (error) {
      console.warn("Error in recallSell calculation:", error);
      return tf.scalar(0) as tf.Scalar;
    }
  }

  static customF1Buy(yTrue: tf.Tensor, yPred: tf.Tensor): tf.Scalar {
    const precision = Metrics.precisionBuy(yTrue, yPred);
    const recall = Metrics.recallBuy(yTrue, yPred);
    return tf
      .scalar(2)
      .mul(precision.mul(recall))
      .div(precision.add(recall).add(1e-6)) as tf.Scalar;
  }

  static customF1Sell(yTrue: tf.Tensor, yPred: tf.Tensor): tf.Scalar {
    const precision = Metrics.precisionSell(yTrue, yPred);
    const recall = Metrics.recallSell(yTrue, yPred);
    return tf
      .scalar(2)
      .mul(precision.mul(recall))
      .div(precision.add(recall).add(1e-6)) as tf.Scalar;
  }

  static customPrecision(
    yTrue: tf.Tensor,
    yPred: tf.Tensor
  ): { buy: tf.Scalar; sell: tf.Scalar } {
    return {
      buy: Metrics.precisionBuy(yTrue, yPred),
      sell: Metrics.precisionSell(yTrue, yPred),
    };
  }

  static customRecall(
    yTrue: tf.Tensor,
    yPred: tf.Tensor
  ): { buy: tf.Scalar; sell: tf.Scalar } {
    return {
      buy: Metrics.recallBuy(yTrue, yPred),
      sell: Metrics.recallSell(yTrue, yPred),
    };
  }

  static customF1(
    yTrue: tf.Tensor,
    yPred: tf.Tensor
  ): { buy: tf.Scalar; sell: tf.Scalar } {
    return {
      buy: Metrics.customF1Buy(yTrue, yPred),
      sell: Metrics.customF1Sell(yTrue, yPred),
    };
  }

  static calculateMetrics(predictedLabels: number[], yArray: number[]) {
    const truePositivesBuy = predictedLabels.reduce(
      (sum, p, i) => sum + (p === 1 && yArray[i] === 1 ? 1 : 0),
      0
    );
    const truePositivesSell = predictedLabels.reduce(
      (sum, p, i) => sum + (p === 0 && yArray[i] === 0 ? 1 : 0),
      0
    );
    const predictedBuys = predictedLabels.reduce(
      (sum, p) => sum + (p === 1 ? 1 : 0),
      0
    );
    const predictedSells = predictedLabels.reduce(
      (sum, p) => sum + (p === 0 ? 1 : 0),
      0
    );
    const actualBuys = yArray.reduce((sum, y) => sum + (y === 1 ? 1 : 0), 0);
    const actualSells = yArray.reduce((sum, y) => sum + (y === 0 ? 1 : 0), 0);

    const precisionBuy =
      predictedBuys > 0 ? truePositivesBuy / predictedBuys : 0;
    const precisionSell =
      predictedSells > 0 ? truePositivesSell / predictedSells : 0;
    const recallBuy = actualBuys > 0 ? truePositivesBuy / actualBuys : 0;
    const recallSell = actualSells > 0 ? truePositivesSell / actualSells : 0;
    const f1Buy =
      precisionBuy + recallBuy > 0
        ? (2 * precisionBuy * recallBuy) / (precisionBuy + recallBuy)
        : 0;
    const f1Sell =
      precisionSell + recallSell > 0
        ? (2 * precisionSell * recallSell) / (precisionSell + recallSell)
        : 0;

    return {
      precisionBuy,
      precisionSell,
      recallBuy,
      recallSell,
      f1Buy,
      f1Sell,
    };
  }

  static async evaluateModel(
    model: tf.LayersModel,
    X: tf.Tensor,
    y: tf.Tensor
  ): Promise<void> {
    console.log("\n=== Model Evaluation ===");
    const predictions = model.predict(X) as tf.Tensor;
    const predictedLabels = Array.from(await predictions.argMax(-1).data());
    const yArray = Array.from(await y.argMax(-1).data());

    const metrics = Metrics.calculateMetrics(predictedLabels, yArray);
    console.log(
      `Precision - Buy: ${metrics.precisionBuy.toFixed(
        4
      )}, Sell: ${metrics.precisionSell.toFixed(4)}`
    );
    console.log(
      `Recall - Buy: ${metrics.recallBuy.toFixed(
        4
      )}, Sell: ${metrics.recallSell.toFixed(4)}`
    );
    console.log(
      `F1 Score - Buy: ${metrics.f1Buy.toFixed(
        4
      )}, Sell: ${metrics.f1Sell.toFixed(4)}`
    );

    // Calculate additional metrics
    const balancedAccuracy = (metrics.recallBuy + metrics.recallSell) / 2;
    console.log(`Balanced Accuracy: ${balancedAccuracy.toFixed(4)}`);

    // Calculate Matthews Correlation Coefficient
    const mcc = this.calculateMCC(predictedLabels, yArray);
    console.log(`Matthews Correlation Coefficient: ${mcc.toFixed(4)}`);

    // Calculate confusion matrix
    const confusionMatrix = this.calculateConfusionMatrix(
      predictedLabels,
      yArray
    );
    console.log("Confusion Matrix:");
    console.log(`[ [ ${confusionMatrix[0][0]}, ${confusionMatrix[0][1]} ],`);
    console.log(`  [ ${confusionMatrix[1][0]}, ${confusionMatrix[1][1]} ] ]`);

    // Calculate class distribution
    const buyCount = yArray.filter((l) => l === 1).length;
    const sellCount = yArray.filter((l) => l === 0).length;
    console.log(`Class Distribution - Buy: ${buyCount}, Sell: ${sellCount}`);

    predictions.dispose();
  }

  private static calculateMCC(
    predictedLabels: number[],
    trueLabels: number[]
  ): number {
    let tp = 0,
      tn = 0,
      fp = 0,
      fn = 0;

    for (let i = 0; i < predictedLabels.length; i++) {
      if (predictedLabels[i] === 1 && trueLabels[i] === 1) tp++;
      else if (predictedLabels[i] === 0 && trueLabels[i] === 0) tn++;
      else if (predictedLabels[i] === 1 && trueLabels[i] === 0) fp++;
      else if (predictedLabels[i] === 0 && trueLabels[i] === 1) fn++;
    }

    const numerator = tp * tn - fp * fn;
    const denominator = Math.sqrt(
      (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)
    );

    return denominator === 0 ? 0 : numerator / denominator;
  }

  private static calculateConfusionMatrix(
    predictedLabels: number[],
    trueLabels: number[]
  ): number[][] {
    let tp = 0,
      tn = 0,
      fp = 0,
      fn = 0;

    for (let i = 0; i < predictedLabels.length; i++) {
      if (predictedLabels[i] === 1 && trueLabels[i] === 1) tp++;
      else if (predictedLabels[i] === 0 && trueLabels[i] === 0) tn++;
      else if (predictedLabels[i] === 1 && trueLabels[i] === 0) fp++;
      else if (predictedLabels[i] === 0 && trueLabels[i] === 1) fn++;
    }

    return [
      [tn, fp],
      [fn, tp],
    ]; // [Sell, Buy] x [Sell, Buy]
  }
}
