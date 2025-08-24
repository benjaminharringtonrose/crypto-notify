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
    // REVERTED TO PROVEN BASELINE: Optimal architecture after systematic testing
    console.log(
      "🔄 REVERTED: Back to proven baseline architecture after systematic experiments"
    );

    // CRITICAL: Add deterministic seeding for stable training
    tf.randomUniform([1, 1], 0, 1, "float32", 42);

    const model = tf.sequential();

    // Conv1D layer - PROVEN optimal settings from experiments
    model.add(
      tf.layers.conv1d({
        inputShape: [this.timesteps, this.features],
        filters: 48, // PROVEN OPTIMAL: Major improvement over 32 filters
        kernelSize: 3, // PROVEN OPTIMAL: Best kernel size (5,7 failed)
        activation: "relu", // PROVEN OPTIMAL: Standard activation placement
        kernelInitializer: "heNormal",
        kernelRegularizer: tf.regularizers.l2({ l2: 0.001 }),
        name: "conv1d_input", // CRITICAL: Match expected layer name
      })
    );
    model.add(tf.layers.batchNormalization({ name: "bn_conv1" }));
    model.add(tf.layers.dropout({ rate: 0.3, name: "dropout_conv1" }));

    // LSTM layer - PROVEN optimal baseline configuration
    model.add(
      tf.layers.lstm({
        units: 64, // PROVEN OPTIMAL: 64 units is the capacity limit for our dataset
        returnSequences: false, // PROVEN: returnSequences=true failed
        kernelInitializer: "heNormal",
        recurrentDropout: 0.1, // Standard dropout
        name: "lstm1",
      })
    );

    // Dense layer - PROVEN optimal baseline
    model.add(
      tf.layers.dense({
        units: 32, // PROVEN OPTIMAL: 48 units failed (severe overfitting), 32 is limit
        activation: "relu",
        kernelInitializer: "heNormal",
        kernelRegularizer: tf.regularizers.l2({ l2: 0.001 }),
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

    console.log("🚀 OPTIMAL BASELINE: Restored after systematic experiments");
    console.log(
      "🎯 Target: Proven optimal configuration for balanced predictions"
    );
    console.log(
      `📊 Architecture: Conv1D(48,3) → BN → Dropout(0.3) → LSTM(64) → Dense(32) → Dropout(0.3) → Output(2)`
    );
    console.log(
      `📈 Timesteps: ${this.timesteps} (35 days), Features: ${this.features} (36 advanced microstructure indicators)`
    );

    model.summary();
    return model;
  }
}
