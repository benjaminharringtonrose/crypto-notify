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
    console.log("Creating BALANCED model architecture for stable learning...");
    const model = tf.sequential();

    // BALANCED ARCHITECTURE: Moderate complexity for stable learning
    // Conv1D layer with moderate filters
    model.add(
      tf.layers.conv1d({
        inputShape: [this.timesteps, this.features],
        filters: 48, // EXPERIMENT 3: 32 → 48 for better feature extraction
        kernelSize: 3, // REVERTED: 7 → 3, larger kernels reduce performance
        activation: "relu",
        kernelInitializer: "heNormal",
        kernelRegularizer: tf.regularizers.l2({ l2: 0.001 }),
        name: "conv1d_input",
      })
    );
    model.add(tf.layers.batchNormalization({ name: "bn_conv1" }));
    model.add(tf.layers.dropout({ rate: 0.2, name: "dropout_conv1" }));

    // Single LSTM layer with increased capacity
    model.add(
      tf.layers.lstm({
        units: 64, // REVERTED: 80 → 64, increased capacity caused overfitting
        returnSequences: false,
        kernelInitializer: "heNormal",
        recurrentDropout: 0.2,
        name: "lstm1",
      })
    );

    // Single dense layer
    model.add(
      tf.layers.dense({
        units: 32, // REVERTED: 48 → 32, increased capacity caused overfitting
        activation: "relu",
        kernelInitializer: "heNormal",
        kernelRegularizer: tf.regularizers.l2({ l2: 0.001 }),
        name: "dense1",
      })
    );
    model.add(tf.layers.dropout({ rate: 0.3, name: "dropout_dense1" })); // REVERTED: 0.2 → 0.3, less dropout caused overfitting

    // Output layer
    model.add(
      tf.layers.dense({
        units: TRAINING_CONFIG.OUTPUT_CLASSES,
        activation: "softmax",
        kernelInitializer: "heNormal",
        name: "output",
      })
    );

    console.log(
      `✅ Balanced model created with ~50K parameters for stable learning`
    );
    console.log(
      `Model architecture: Conv1D(32,3) -> BN -> Dropout -> LSTM(64) -> Dense(32) -> Dropout -> Output(${TRAINING_CONFIG.OUTPUT_CLASSES})`
    );
    return model;
  }
}
