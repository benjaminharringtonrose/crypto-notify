import * as tf from "@tensorflow/tfjs-node";

export class BestWeightsCallback extends tf.CustomCallback {
  private bestValLoss = Infinity;
  private bestWeights: tf.Tensor[] | null = null;
  private currentModel: tf.LayersModel | null = null;

  constructor(args: tf.CustomCallbackArgs) {
    super(args);
  }

  async onEpochEnd(epoch: number, logs?: tf.Logs) {
    if (!this.currentModel)
      throw new Error("Model not set in BestWeightsCallback");
    if (logs && logs.val_loss < this.bestValLoss) {
      this.bestValLoss = logs.val_loss;
      this.bestWeights = this.currentModel.getWeights();
      console.log(
        `New best val_loss: ${this.bestValLoss} at epoch ${epoch + 1}`
      );
    }
  }

  setModel(model: tf.LayersModel) {
    this.currentModel = model;
  }

  applyBestWeights(model: tf.LayersModel) {
    if (this.bestWeights) model.setWeights(this.bestWeights);
  }
}
