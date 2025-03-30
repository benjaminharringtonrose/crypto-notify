import * as tf from "@tensorflow/tfjs-node";

export class PredictionLoggerCallback extends tf.CustomCallback {
  public readonly validationData: tf.Tensor;
  private model?: tf.LayersModel;

  constructor(validationData: tf.Tensor) {
    super({});
    this.validationData = validationData;
  }

  async onEpochEnd(epoch: number, logs?: tf.Logs) {
    if (!this.model)
      throw new Error("Model not set in PredictionLoggerCallback");

    if (epoch % 5 !== 0) return;

    const predsAllVal = this.model.predict(this.validationData) as tf.Tensor;
    const predLabelsAllVal = predsAllVal.argMax(-1).dataSync();
    const buyCountAllVal = Array.from(predLabelsAllVal).filter(
      (p) => p === 1
    ).length;
    const totalValSamples = predLabelsAllVal.length;
    console.log(
      `Epoch ${
        epoch + 1
      } Validation Buy Count: ${buyCountAllVal}, Sell Count: ${
        totalValSamples - buyCountAllVal
      }, Buy Ratio: ${(buyCountAllVal / totalValSamples).toFixed(3)}`
    );

    if (logs)
      console.log(`Epoch ${epoch + 1} Val Loss: ${logs.val_loss.toFixed(4)}`);

    predsAllVal.dispose();
  }

  setParams(params: tf.CustomCallbackArgs) {
    // Required by tf.CustomCallback
  }

  setModel(model: tf.LayersModel) {
    this.model = model;
  }
}
