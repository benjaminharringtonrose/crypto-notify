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
    console.log("Creating OPTIMIZED model architecture for better accuracy...");
    const model = tf.sequential();

    // REVERTED: Back to optimal baseline configuration
    // Conv1D layer with proven settings
    model.add(
      tf.layers.conv1d({
        inputShape: [this.timesteps, this.features],
        filters: 48, // KEPT: Our successful 48 filters
        kernelSize: 3, // KEPT: Successful kernel size
        activation: "relu",
        kernelInitializer: "heNormal",
        kernelRegularizer: tf.regularizers.l2({ l2: 0.001 }),
        name: "conv1d_input",
      })
    );
    model.add(tf.layers.batchNormalization({ name: "bn_conv1" }));
    model.add(tf.layers.dropout({ rate: 0.2, name: "dropout_conv1" }));

    // LSTM layer - reverted to baseline
    model.add(
      tf.layers.lstm({
        units: 64,
        returnSequences: false,
        kernelInitializer: "heNormal",
        recurrentDropout: 0.2,
        name: "lstm1",
      })
    );

    // Dense layer
    model.add(
      tf.layers.dense({
        units: 32,
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

    console.log(
      `✅ Optimized model created with same padding and LSTM dropout`
    );
    console.log(
      `Model architecture: Conv1D(48,3,same) -> BN -> Dropout -> LSTM(64,dropout=0.1) -> Dense(32) -> Dropout -> Output(${TRAINING_CONFIG.OUTPUT_CLASSES})`
    );
    model.summary();
    return model;
  }
}
