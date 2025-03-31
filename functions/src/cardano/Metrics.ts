import * as tf from "@tensorflow/tfjs-node";
import { TRAINING_CONFIG } from "../constants";

export class Metrics {
  static focalLoss(
    yTrue: tf.Tensor,
    yPred: tf.Tensor,
    gamma: number = TRAINING_CONFIG.GAMMA,
    alphaArr: [number, number] = TRAINING_CONFIG.ALPHA
  ): tf.Scalar {
    const alpha = tf.tensor1d(alphaArr);
    const ce = tf.losses.sigmoidCrossEntropy(yTrue, yPred);
    const pt = yTrue.mul(yPred).sum(-1).clipByValue(0, 1);
    const focalWeight = tf.pow(tf.sub(1, pt), gamma);
    const yTrueIndices = yTrue.argMax(-1);
    const alphaWeighted = tf.gather(alpha, yTrueIndices);
    const loss = ce.mul(focalWeight).mul(alphaWeighted).mean() as tf.Scalar;
    alpha.dispose();
    return loss;
  }

  static customPrecision(yTrue: tf.Tensor, yPred: tf.Tensor): tf.Scalar {
    const truePos = yTrue.mul(yPred.round()).sum();
    const predPos = yPred.round().sum();
    return truePos.div(predPos.add(tf.scalar(1e-6))) as tf.Scalar;
  }

  static customRecall(yTrue: tf.Tensor, yPred: tf.Tensor): tf.Scalar {
    const truePos = yTrue.mul(yPred.round()).sum();
    const actualPos = yTrue.sum();
    return truePos.div(actualPos.add(tf.scalar(1e-6))) as tf.Scalar;
  }

  static customF1(yTrue: tf.Tensor, yPred: tf.Tensor): tf.Scalar {
    const precision = Metrics.customPrecision(yTrue, yPred);
    const recall = Metrics.customRecall(yTrue, yPred);
    return tf
      .scalar(2)
      .mul(precision.mul(recall))
      .div(precision.add(recall).add(1e-6)) as tf.Scalar;
  }

  static calculateMetrics(
    predictedLabels: number[],
    yArray: number[]
  ): {
    precisionBuy: number;
    precisionSell: number;
    recallBuy: number;
    recallSell: number;
    f1Buy: number;
    f1Sell: number;
  } {
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
    const preds = model.predict(X) as tf.Tensor;
    const predArray = (await preds.array()) as number[][];
    const yArray = Array.from(await y.argMax(-1).data());

    let bestThreshold = 0.3;
    let bestF1 = 0;
    for (let t = 0.2; t <= 0.5; t += 0.05) {
      const predictedLabels = predArray.map((p) => (p[1] > t ? 1 : 0));
      const metrics = Metrics.calculateMetrics(predictedLabels, yArray);
      const avgF1 = (metrics.f1Buy + metrics.f1Sell) / 2;
      if (avgF1 > bestF1) {
        bestF1 = avgF1;
        bestThreshold = t;
      }
    }
    console.log(`Optimal threshold: ${bestThreshold}`);
    const predictedLabels = predArray.map((p) =>
      p[1] > bestThreshold ? 1 : 0
    );

    const buyCount = predictedLabels.filter((p) => p === 1).length;
    console.log(
      `Predicted Buy: ${buyCount}, Sell: ${predictedLabels.length - buyCount}`
    );

    const metrics = Metrics.calculateMetrics(predictedLabels, yArray);
    console.log(
      `Precision Buy: ${metrics.precisionBuy.toFixed(
        4
      )}, Precision Sell: ${metrics.precisionSell.toFixed(4)}`
    );
    console.log(
      `Recall Buy: ${metrics.recallBuy.toFixed(
        4
      )}, Recall Sell: ${metrics.recallSell.toFixed(4)}`
    );
    console.log(
      `F1 Buy: ${metrics.f1Buy.toFixed(4)}, F1 Sell: ${metrics.f1Sell.toFixed(
        4
      )}`
    );

    const confusionMatrix = tf.math.confusionMatrix(
      tf.tensor1d(yArray, "int32"),
      tf.tensor1d(predictedLabels, "int32"),
      2
    );
    console.log("Confusion Matrix:", await confusionMatrix.array());

    preds.dispose();
  }
}
