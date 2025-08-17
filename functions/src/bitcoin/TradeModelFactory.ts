import * as tf from "@tensorflow/tfjs-node";
import { TRAINING_CONFIG } from "../constants";

export default class TradeModelFactory {
  private timesteps: number;
  private features: number;

  constructor(timesteps: number, features: number) {
    this.timesteps = timesteps;
    this.features = features;
  }

  public createModel(): tf.LayersModel {
    console.log("Creating SIMPLIFIED fast-learning model architecture...");
    const model = tf.sequential();

    // CRITICAL OPTIMIZATION: Much simpler architecture for faster training
    // Single Conv1D layer with minimal filters
    model.add(
      tf.layers.conv1d({
        inputShape: [this.timesteps, this.features],
        filters: 16, // REDUCED: was 32-128, now 16 for speed
        kernelSize: 3, // REDUCED: was 5-7, now 3 for speed
        activation: "relu",
        kernelInitializer: "heNormal",
        kernelRegularizer: tf.regularizers.l2({
          l2: 0.0001, // REDUCED: was 0.001, now 0.0001
        }),
        name: "conv1d_input",
      })
    );
    model.add(tf.layers.dropout({ rate: 0.1 })); // REDUCED: was 0.2, now 0.1

    // SKIP second conv layer and attention - go directly to LSTM for speed

    // Single simplified LSTM layer
    model.add(
      tf.layers.lstm({
        units: 32, // MUCH REDUCED: was 64-256, now 32
        returnSequences: false, // Don't return sequences for speed
        kernelInitializer: "heNormal",
        kernelRegularizer: tf.regularizers.l2({
          l2: 0.0001, // REDUCED regularization
        }),
        name: "lstm_simple",
      })
    );
    model.add(tf.layers.dropout({ rate: 0.1 })); // REDUCED dropout

    // Direct to output - skip intermediate dense layers for speed
    model.add(
      tf.layers.dense({
        units: TRAINING_CONFIG.OUTPUT_CLASSES, // Direct to output
        activation: "softmax",
        kernelInitializer: "heNormal",
        kernelRegularizer: tf.regularizers.l2({
          l2: 0.0001,
        }),
        name: "output",
      })
    );

    console.log("SIMPLIFIED model created - much faster training expected");
    console.log(
      `Model parameters: Conv1D(16) -> LSTM(32) -> Dense(${TRAINING_CONFIG.OUTPUT_CLASSES})`
    );
    return model;
  }
}
