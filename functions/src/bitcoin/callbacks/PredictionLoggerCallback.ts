import * as tf from "@tensorflow/tfjs-node";
import { TradeModelTrainer } from "../TradeModelTrainer";

export class PredictionLoggerCallback extends tf.CustomCallback {
  public readonly validationFeatures: tf.Tensor;
  public readonly validationLabels: tf.Tensor;
  private model?: tf.LayersModel;

  constructor(
    validationFeatures: tf.Tensor,
    validationLabels: tf.Tensor,
    trainer: TradeModelTrainer
  ) {
    super({});
    this.validationFeatures = validationFeatures;
    this.validationLabels = validationLabels;
  }

  async onEpochEnd(epoch: number, logs?: tf.Logs) {
    if (!this.model)
      throw new Error("Model not set in PredictionLoggerCallback");

    // This callback is now deprecated - all logging moved to TrainingLoggerCallback
    // Keeping minimal functionality for backward compatibility
  }

  setModel(model: tf.LayersModel) {
    this.model = model;
  }
}
