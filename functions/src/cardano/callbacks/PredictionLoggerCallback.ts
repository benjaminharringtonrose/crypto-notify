import * as tf from "@tensorflow/tfjs-node";
import { TRAINING_CONFIG } from "../../constants";
import { TradeModelTrainer } from "../TradeModelTrainer";
import { Metrics } from "../Metrics";

export class PredictionLoggerCallback extends tf.CustomCallback {
  public readonly validationFeatures: tf.Tensor;
  public readonly validationLabels: tf.Tensor;
  private model?: tf.LayersModel;
  private trainer: TradeModelTrainer;

  constructor(
    validationFeatures: tf.Tensor,
    validationLabels: tf.Tensor,
    trainer: TradeModelTrainer
  ) {
    super({});
    this.validationFeatures = validationFeatures;
    this.validationLabels = validationLabels;
    this.trainer = trainer;
  }

  async onEpochEnd(epoch: number, logs?: tf.Logs) {
    if (!this.model)
      throw new Error("Model not set in PredictionLoggerCallback");
    const predsAllVal = this.model.predict(
      this.validationFeatures
    ) as tf.Tensor;
    const predArray = (await predsAllVal.array()) as number[][];
    const bestThreshold = this.trainer.getBestThreshold();
    const predLabelsAllVal = predArray.map((p) =>
      p[1] > bestThreshold ? 1 : 0
    );
    const yVal = Array.from(
      await this.validationLabels.argMax(-1).data()
    ) as number[];
    const buyCountAllVal = predLabelsAllVal.filter((p) => p === 1).length;
    const totalValSamples = predLabelsAllVal.length;
    const buyRatio = buyCountAllVal / totalValSamples;
    const metrics = Metrics.calculateMetrics(predLabelsAllVal, yVal);

    const alpha = tf.tensor1d(TRAINING_CONFIG.ALPHA);
    const yTrue = this.validationLabels;
    const pt = yTrue.mul(predsAllVal).sum(-1).clipByValue(0, 1);
    const focalWeight = tf.pow(tf.sub(1, pt), TRAINING_CONFIG.GAMMA);
    const yTrueIndices = yTrue.argMax(-1);
    const alphaWeighted = tf.gather(alpha, yTrueIndices);
    const effectiveWeights = focalWeight
      .mul(alphaWeighted)
      .mean()
      .dataSync()[0];
    console.log(
      `Epoch ${epoch + 1} Effective Class Weight: ${effectiveWeights.toFixed(
        4
      )}`
    );

    console.log(
      `Epoch ${
        epoch + 1
      } Validation Buy Count: ${buyCountAllVal}, Sell Count: ${
        totalValSamples - buyCountAllVal
      }, Buy Ratio: ${buyRatio.toFixed(3)}`
    );
    console.log(
      `Epoch ${epoch + 1} Val Precision Buy: ${metrics.precisionBuy.toFixed(
        4
      )}, Precision Sell: ${metrics.precisionSell.toFixed(4)}`
    );
    console.log(
      `Epoch ${epoch + 1} Val Recall Buy: ${metrics.recallBuy.toFixed(
        4
      )}, Recall Sell: ${metrics.recallSell.toFixed(4)}`
    );
    console.log(
      `Epoch ${epoch + 1} Val F1 Buy: ${metrics.f1Buy.toFixed(
        4
      )}, F1 Sell: ${metrics.f1Sell.toFixed(4)}`
    );
    if (logs)
      console.log(`Epoch ${epoch + 1} Val Loss: ${logs.val_loss.toFixed(4)}`);

    predsAllVal.dispose();
    alpha.dispose();
    pt.dispose();
    focalWeight.dispose();
    alphaWeighted.dispose();
  }

  setModel(model: tf.LayersModel) {
    this.model = model;
  }
}
