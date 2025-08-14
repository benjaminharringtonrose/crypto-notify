import * as tf from "@tensorflow/tfjs-node";
import { TRAINING_CONFIG } from "../../constants";

export class ExponentialDecayLRCallback extends tf.CustomCallback {
  private model: tf.LayersModel | null = null;
  private initialLR: number;
  private decayRate: number;
  private warmupEpochs: number;
  private warmupInitialLR: number;

  constructor(
    initialLR: number = TRAINING_CONFIG.INITIAL_LEARNING_RATE,
    decayRate: number = TRAINING_CONFIG.LR_DECAY_RATE,
    warmupEpochs: number = TRAINING_CONFIG.WARMUP_EPOCHS,
    warmupInitialLR: number = TRAINING_CONFIG.WARMUP_INITIAL_LR
  ) {
    super({});
    this.initialLR = initialLR;
    this.decayRate = decayRate;
    this.warmupEpochs = warmupEpochs;
    this.warmupInitialLR = warmupInitialLR;
  }

  setModel(model: tf.LayersModel) {
    this.model = model;
  }

  private getLearningRate(epoch: number): number {
    if (epoch < this.warmupEpochs) {
      const slope = (this.initialLR - this.warmupInitialLR) / this.warmupEpochs;
      return this.warmupInitialLR + slope * epoch;
    }
    const decayEpochs = epoch - this.warmupEpochs;
    return this.initialLR * Math.pow(this.decayRate, decayEpochs);
  }

  async onEpochEnd(epoch: number, logs?: tf.Logs) {
    // Changed to onEpochEnd for consistency
    if (!this.model)
      throw new Error("Model not set in ExponentialDecayLRCallback");
    const lr = this.getLearningRate(epoch);
    const newOptimizer = tf.train.adam(lr);
    this.model.compile({
      optimizer: newOptimizer,
      loss: this.model.loss,
      metrics: this.model.metrics as string[],
    });
    // Learning rate logging moved to TrainingLoggerCallback
  }
}
