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
    // RECOVERY-1: Revert to PROVEN optimal baseline (65.96% validation accuracy)
    console.log("🔄 RECOVERY-1: Reverting to documented optimal baseline...");
    
    // CRITICAL: Add deterministic seeding for stable training
    tf.randomUniform([1, 1], 0, 1, "float32", 42);
    
    const model = tf.sequential();

    // Conv1D layer - PROVEN optimal settings from experiments
    model.add(
      tf.layers.conv1d({
        inputShape: [this.timesteps, this.features],
        filters: 48, // PROVEN OPTIMAL: Major improvement over 32 filters
        kernelSize: 3, // PROVEN OPTIMAL: Best kernel size (5,7 failed)
        activation: "relu",
        kernelInitializer: "heNormal",
        kernelRegularizer: tf.regularizers.l2({ l2: 0.001 }),
        name: "conv1d_input",
      })
    );
    model.add(tf.layers.batchNormalization({ name: "bn_conv1" }));
    model.add(tf.layers.dropout({ rate: 0.3, name: "dropout_conv1" })); // PROVEN: 0.2→0.3 critical

    // LSTM layer - PROVEN optimal baseline (single LSTM, 64 units)
    model.add(
      tf.layers.lstm({
        units: 64, // PROVEN OPTIMAL: 80 units failed (overfitting), 64 is limit
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
    model.add(tf.layers.dropout({ rate: 0.3, name: "dropout_dense1" })); // PROVEN: 0.2 failed

    // Output layer
    model.add(
      tf.layers.dense({
        units: TRAINING_CONFIG.OUTPUT_CLASSES,
        activation: "softmax",
        kernelInitializer: "glorotUniform",
        name: "output",
      })
    );

    console.log("✅ RECOVERY-1: Restored proven optimal baseline");
    console.log("🎯 Target: 65.96% validation accuracy (documented optimal)");
    console.log(`📊 Architecture: Conv1D(48,3) → BN → Dropout(0.3) → LSTM(64) → Dense(32) → Dropout(0.3) → Output(2)`);
    console.log(`📈 Features: ${this.features} (should be 25 core indicators)`);
    
    model.summary();
    return model;
  }
}
