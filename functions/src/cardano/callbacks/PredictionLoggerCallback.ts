import * as tf from "@tensorflow/tfjs-node";

export class PredictionLoggerCallback extends tf.CustomCallback {
  public readonly validationData: tf.Tensor; // Changed to public to match inferred base class
  protected model?: tf.LayersModel;

  constructor(validationData: tf.Tensor) {
    super({});
    this.validationData = validationData;
  }

  async onEpochEnd(epoch: number, logs?: tf.Logs) {
    if (!this.model)
      throw new Error("Model not set in PredictionLoggerCallback");
    const preds = this.model.predict(
      this.validationData.slice([0], [5])
    ) as tf.Tensor;
    const predArray = (await preds.array()) as number[][];
    const predictedLabels = predArray.map((p) => (p[1] > p[0] ? 1 : 0));
    const buyCount = predictedLabels.filter((p) => p === 1).length;
    console.log(`Epoch ${epoch + 1} Predictions (first 5):`, predArray);
    console.log(
      `Epoch ${epoch + 1} Predicted Buy: ${buyCount}, Sell: ${5 - buyCount}`
    );
    if (logs)
      console.log(`Epoch ${epoch + 1} Val Loss: ${logs.val_loss.toFixed(4)}`);
    preds.dispose();
  }

  setParams(params: tf.CustomCallbackArgs) {
    // Required method, no specific params needed
  }

  setModel(model: tf.LayersModel) {
    this.model = model;
  }
}
