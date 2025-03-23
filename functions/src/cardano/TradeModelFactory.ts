import * as tf from "@tensorflow/tfjs-node";

export default class TradeModelFactory {
  private timesteps: number;
  private features: number;

  constructor(timesteps: number, features: number) {
    this.timesteps = timesteps;
    this.features = features;
  }

  public createModel(): tf.LayersModel {
    const model = tf.sequential();

    // Convolutional Layer
    model.add(
      tf.layers.conv1d({
        inputShape: [this.timesteps, this.features],
        filters: 32,
        kernelSize: 3,
        activation: "relu",
        kernelInitializer: "orthogonal",
        name: "conv1d",
      })
    );

    // LSTM Layers
    model.add(
      tf.layers.lstm({
        units: 128,
        returnSequences: true,
        kernelInitializer: "orthogonal",
        name: "lstm1",
      })
    );
    model.add(
      tf.layers.lstm({
        units: 32,
        kernelInitializer: "orthogonal",
        name: "lstm2",
      })
    );

    // Batch Normalization
    model.add(
      tf.layers.batchNormalization({
        name: "batchNormalization",
      })
    );

    // Dense Layers
    model.add(
      tf.layers.dense({
        units: 32,
        activation: "relu",
        kernelInitializer: "heNormal",
        name: "dense",
      })
    );
    model.add(
      tf.layers.dense({
        units: 2,
        activation: "softmax",
        kernelInitializer: "heNormal",
        name: "dense_1",
      })
    );

    return model;
  }
}
