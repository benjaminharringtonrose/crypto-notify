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
    console.log(
      "Creating OPTIMIZED baseline model (reverted from failed multi-scale)..."
    );
    const model = tf.sequential();

    // REVERTED: Multi-scale v2.0 also failed - back to optimal baseline
    // Conv1D layer with proven optimal settings
    model.add(
      tf.layers.conv1d({
        inputShape: [this.timesteps, this.features],
        filters: 48, // OPTIMAL: Our best performing filter count
        kernelSize: 3, // OPTIMAL: Best kernel size found
        activation: "relu",
        kernelInitializer: "heNormal",
        kernelRegularizer: tf.regularizers.l2({ l2: 0.001 }),
        name: "conv1d_input",
      })
    );
    model.add(tf.layers.batchNormalization({ name: "bn_conv1" }));
    model.add(tf.layers.dropout({ rate: 0.2, name: "dropout_conv1" }));

    // LSTM layer - optimal baseline
    model.add(
      tf.layers.lstm({
        units: 64, // REVERTED: 80 caused severe class imbalance
        returnSequences: false, // OPTIMAL: For binary classification
        kernelInitializer: "heNormal",
        recurrentDropout: 0.2,
        name: "lstm1",
      })
    );

    // Dense layer - optimal baseline
    model.add(
      tf.layers.dense({
        units: 32, // OPTIMAL: Capacity limit found
        activation: "relu",
        kernelInitializer: "heNormal",
        kernelRegularizer: tf.regularizers.l1l2({ l1: 0.0005, l2: 0.001 }), // EXPERIMENT NEXT-F: L1+L2 regularization
        name: "dense1",
      })
    );
    model.add(tf.layers.dropout({ rate: 0.3, name: "dropout_dense1" }));

    // Output layer
    model.add(
      tf.layers.dense({
        units: TRAINING_CONFIG.OUTPUT_CLASSES,
        activation: "softmax",
        kernelInitializer: "glorotUniform",
        name: "output",
      })
    );

    console.log("✅ Reverted to proven optimal baseline architecture");
    console.log(
      `Model: Conv1D(48,3) -> BN -> LSTM(64) -> Dense(32) -> Output(${TRAINING_CONFIG.OUTPUT_CLASSES}) with 25 core features`
    );
    model.summary();
    return model;
  }
}
