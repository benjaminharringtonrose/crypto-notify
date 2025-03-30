import * as tf from "@tensorflow/tfjs-node";

export class CyclicLearningRateCallback extends tf.CustomCallback {
  private model: tf.LayersModel | null = null;
  private minLR: number;
  private maxLR: number;
  private cycleLength: number;

  constructor(
    minLR: number = 0.00001,
    maxLR: number = 0.0005,
    cycleLength: number = 10
  ) {
    super({});
    this.minLR = minLR;
    this.maxLR = maxLR;
    this.cycleLength = cycleLength;
  }

  setModel(model: tf.LayersModel) {
    this.model = model;
  }

  setParams(_: tf.CustomCallbackArgs) {
    // Required by TensorFlow.js, no specific params needed
  }

  private getLearningRate(epoch: number): number {
    const step = (this.maxLR - this.minLR) / (this.cycleLength / 2);
    const cyclePos = epoch % this.cycleLength;
    return cyclePos < this.cycleLength / 2
      ? this.minLR + step * cyclePos
      : this.maxLR - step * (cyclePos - this.cycleLength / 2);
  }

  async onEpochBegin(epoch: number) {
    if (!this.model)
      throw new Error("Model not set in CyclicLearningRateCallback");
    const lr = this.getLearningRate(epoch);
    const newOptimizer = tf.train.adam(lr);
    this.model.compile({
      optimizer: newOptimizer,
      loss: this.model.loss, // Preserve existing loss function
      metrics: this.model.metrics as string[], // Preserve metrics
    });
    console.log(`Epoch ${epoch + 1} Learning Rate: ${lr.toFixed(6)}`);
  }
}
