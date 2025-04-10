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

  static precisionBuy(yTrue: tf.Tensor, yPred: tf.Tensor): tf.Scalar {
    const predLabels = yPred.argMax(-1);
    const trueLabels = yTrue.argMax(-1);
    const buyMask = predLabels.equal(1).cast("float32");
    const truePosBuy = tf
      .logicalAnd(trueLabels.equal(1), predLabels.equal(1))
      .cast("float32")
      .sum();
    const predBuy = buyMask.sum();
    return truePosBuy.div(predBuy.add(tf.scalar(1e-6))) as tf.Scalar;
  }

  static precisionSell(yTrue: tf.Tensor, yPred: tf.Tensor): tf.Scalar {
    const predLabels = yPred.argMax(-1);
    const trueLabels = yTrue.argMax(-1);
    const sellMask = predLabels.equal(0).cast("float32");
    const truePosSell = tf
      .logicalAnd(trueLabels.equal(0), predLabels.equal(0))
      .cast("float32")
      .sum();
    const predSell = sellMask.sum();
    return truePosSell.div(predSell.add(tf.scalar(1e-6))) as tf.Scalar;
  }

  static recallBuy(yTrue: tf.Tensor, yPred: tf.Tensor): tf.Scalar {
    const predLabels = yPred.argMax(-1);
    const trueLabels = yTrue.argMax(-1);
    const truePosBuy = tf
      .logicalAnd(trueLabels.equal(1), predLabels.equal(1))
      .cast("float32")
      .sum();
    const actualBuy = trueLabels.equal(1).cast("float32").sum();
    return truePosBuy.div(actualBuy.add(tf.scalar(1e-6))) as tf.Scalar;
  }

  static recallSell(yTrue: tf.Tensor, yPred: tf.Tensor): tf.Scalar {
    const predLabels = yPred.argMax(-1);
    const trueLabels = yTrue.argMax(-1);
    const truePosSell = tf
      .logicalAnd(trueLabels.equal(0), predLabels.equal(0))
      .cast("float32")
      .sum();
    const actualSell = trueLabels.equal(0).cast("float32").sum();
    return truePosSell.div(actualSell.add(tf.scalar(1e-6))) as tf.Scalar;
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
    const preds = model.predict(X) as tf.Tensor;
    const predArray = (await preds.array()) as number[][];
    const yArray = Array.from(await y.argMax(-1).data());
    const predictedLabels = predArray.map((p) => (p[1] > 0.5 ? 1 : 0));
    const buyCount = predictedLabels.filter((p) => p === 1).length;
    console.log(
      `Predicted Buy: ${buyCount}, Sell: ${predictedLabels.length - buyCount}`
    );

    const metrics = Metrics.calculateMetrics(predictedLabels, yArray);
    // Fixed: Corrected labels for precision and recall
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
