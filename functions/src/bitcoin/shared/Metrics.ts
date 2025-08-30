import * as tf from "@tensorflow/tfjs-node";
import { TRAINING_CONFIG } from "../../constants";

export class Metrics {
  static focalLoss(
    yTrue: tf.Tensor,
    yPred: tf.Tensor,
    gamma: number = TRAINING_CONFIG.GAMMA,
    alphaArr: [number, number] = TRAINING_CONFIG.ALPHA
  ): tf.Scalar {
    try {
      // Use standard cross-entropy with label smoothing for better learning
      const labelSmoothing = 0.1;

      // Apply softmax to get probabilities
      const probs = yPred.softmax();

      // Get the predicted probabilities for the true classes
      const yTrueIndices = yTrue.argMax(-1);
      const gatheredProbs = tf.gather(probs, yTrueIndices, 1);
      const pt = gatheredProbs.squeeze();

      // Apply label smoothing
      const smoothedPt = tf.add(
        tf.mul(pt, tf.sub(1, labelSmoothing)),
        tf.mul(labelSmoothing, 0.5)
      );

      // Simple cross-entropy loss with smoothing
      const ce = tf.log(tf.add(smoothedPt, 1e-8));
      const loss = tf.neg(ce);

      // Clean up tensors
      probs.dispose();
      gatheredProbs.dispose();
      pt.dispose();
      smoothedPt.dispose();
      ce.dispose();

      return loss.mean() as tf.Scalar;
    } catch (error) {
      console.warn(
        "Loss calculation failed, falling back to cross-entropy:",
        error
      );
      // Fallback to standard cross-entropy
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

  // Custom balanced accuracy metric for imbalanced datasets
  static balancedAccuracy(yTrue: tf.Tensor, yPred: tf.Tensor): tf.Scalar {
    try {
      const predLabels = yPred.argMax(-1);
      const trueLabels = yTrue.argMax(-1);

      // Calculate accuracy for each class
      const buyMask = trueLabels.equal(1);
      const sellMask = trueLabels.equal(0);

      const buyCorrect = tf
        .logicalAnd(buyMask, predLabels.equal(1))
        .cast("float32")
        .sum();
      const sellCorrect = tf
        .logicalAnd(sellMask, predLabels.equal(0))
        .cast("float32")
        .sum();

      const buyTotal = buyMask.cast("float32").sum();
      const sellTotal = sellMask.cast("float32").sum();

      const buyAccuracy = buyCorrect.div(buyTotal.add(tf.scalar(1e-8)));
      const sellAccuracy = sellCorrect.div(sellTotal.add(tf.scalar(1e-8)));

      // Balanced accuracy is the average of per-class accuracies
      const balancedAcc = tf.div(
        tf.add(buyAccuracy, sellAccuracy),
        tf.scalar(2)
      );

      return balancedAcc as tf.Scalar;
    } catch (error) {
      console.warn("Error in balancedAccuracy calculation:", error);
      return tf.scalar(0) as tf.Scalar;
    }
  }

  // Comprehensive model evaluation method
  static async evaluateModel(
    model: tf.LayersModel,
    X: tf.Tensor,
    y: tf.Tensor,
    featureName?: string
  ): Promise<{
    balancedAccuracy: number;
    buyF1: number;
    sellF1: number;
    combinedF1: number;
    matthewsCorrelation: number;
    buyPrecision: number;
    sellPrecision: number;
    buyRecall: number;
    sellRecall: number;
    confusionMatrix: number[][];
  }> {
    const featureInfo = featureName ? ` for feature: ${featureName}` : "";
    console.log(`\n=== Model Evaluation${featureInfo} ===`);

    // Get predictions
    const predictions = model.predict(X) as tf.Tensor;
    const predLabels = predictions.argMax(-1);

    // Convert to arrays
    const predArray = Array.from(await predLabels.data());
    const trueLabels = y.argMax(-1);
    const trueArray = Array.from(await trueLabels.data());

    // Calculate comprehensive metrics
    const metrics = this.calculateMetrics(predArray, trueArray);
    const mcc = this.calculateMCC(predArray, trueArray);
    const confusionMatrix = this.calculateConfusionMatrix(predArray, trueArray);

    // Calculate balanced accuracy
    const balancedAccuracy = (metrics.recallBuy + metrics.recallSell) / 2;

    // Print results
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
    console.log(`Balanced Accuracy: ${balancedAccuracy.toFixed(4)}`);
    console.log(`Matthews Correlation Coefficient: ${mcc.toFixed(4)}`);
    console.log("Confusion Matrix:");
    console.log(`[ [ ${confusionMatrix[0][0]}, ${confusionMatrix[0][1]} ],`);
    console.log(`  [ ${confusionMatrix[1][0]}, ${confusionMatrix[1][1]} ] ]`);
    console.log(
      `Class Distribution - Buy: ${
        trueArray.filter((l) => l === 1).length
      }, Sell: ${trueArray.filter((l) => l === 0).length}`
    );

    // Clean up tensors
    predictions.dispose();
    predLabels.dispose();
    trueLabels.dispose();

    return {
      balancedAccuracy,
      buyF1: metrics.f1Buy,
      sellF1: metrics.f1Sell,
      combinedF1: metrics.f1Buy + metrics.f1Sell,
      matthewsCorrelation: mcc,
      buyPrecision: metrics.precisionBuy,
      sellPrecision: metrics.precisionSell,
      buyRecall: metrics.recallBuy,
      sellRecall: metrics.recallSell,
      confusionMatrix,
    };
  }
}
