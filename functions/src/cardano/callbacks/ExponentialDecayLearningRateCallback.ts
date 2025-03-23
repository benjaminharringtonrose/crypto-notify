import * as tf from "@tensorflow/tfjs-node";

export class ExponentialDecayLearningRateCallback extends tf.CustomCallback {
  private initialLr: number;
  private decayRate: number;
  private currentModel: tf.LayersModel | null = null;

  constructor(initialLr: number, decayRate: number = 0.95) {
    super({}); // Pass empty args; we'll handle params manually
    this.initialLr = initialLr;
    this.decayRate = decayRate;
  }

  async onEpochBegin(epoch: number) {
    if (!this.currentModel) {
      throw new Error("Model not set in ExponentialDecayLearningRateCallback");
    }
    const newLr = this.initialLr * Math.pow(this.decayRate, epoch);
    console.log(`Epoch ${epoch + 1}: Learning rate set to ${newLr}`);
    const newOptimizer = tf.train.rmsprop(newLr); // Use RMSprop as per trainModel
    this.currentModel.optimizer = newOptimizer;
  }

  setModel(model: tf.LayersModel) {
    this.currentModel = model;
  }

  // Required by tf.CustomCallback to receive training params
  setParams(_: tf.CustomCallbackArgs) {
    // No specific params needed, but method must exist
  }
}
