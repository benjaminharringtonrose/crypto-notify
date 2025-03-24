import * as tf from "@tensorflow/tfjs-node";

export class ReduceLROnPlateauCallback extends tf.CustomCallback {
  private model: tf.LayersModel | null = null;
  private currentLearningRate: number;
  private patience: number;
  private factor: number;
  private minLR: number;
  private valLossHistory: number[] = [];

  constructor(
    initialLearningRate: number = 0.001,
    patience: number = 5,
    factor: number = 0.5,
    minLR: number = 0.00001
  ) {
    super({});
    this.currentLearningRate = initialLearningRate;
    this.patience = patience;
    this.factor = factor;
    this.minLR = minLR;
  }

  setModel(model: tf.LayersModel) {
    this.model = model;
  }

  setParams(_: tf.CustomCallbackArgs) {
    // Required by TensorFlow.js to initialize callback with training parameters
    // No specific params needed for this callback, but method must exist
  }

  async onEpochEnd(epoch: number, logs?: tf.Logs) {
    if (!logs || !("val_loss" in logs) || !this.model) return;

    const valLoss = logs.val_loss as number;
    this.valLossHistory.push(valLoss);

    if (epoch < this.patience) return;

    const recentLosses = this.valLossHistory.slice(-this.patience);
    const minRecentLoss = Math.min(...recentLosses);

    if (valLoss >= minRecentLoss && this.currentLearningRate > this.minLR) {
      this.currentLearningRate = Math.max(
        this.currentLearningRate * this.factor,
        this.minLR
      );
      console.log(
        `Reducing learning rate to ${this.currentLearningRate.toFixed(
          6
        )} due to plateau at epoch ${epoch + 1}`
      );

      const newOptimizer = tf.train.adam(this.currentLearningRate);
      this.model.compile({
        optimizer: newOptimizer,
        loss: this.model.loss!, // Preserve the existing loss function
        metrics: this.model.metrics as string[], // Correct type for metrics
      });
    }
  }
}
