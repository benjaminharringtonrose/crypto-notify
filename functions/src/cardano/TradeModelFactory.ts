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

    model.add(
      tf.layers.conv1d({
        inputShape: [this.timesteps, this.features],
        filters: 12,
        kernelSize: 5,
        activation: "relu",
        kernelInitializer: "orthogonal",
        name: "conv1d",
      })
    );

    model.add(
      tf.layers.conv1d({
        filters: 24,
        kernelSize: 3,
        activation: "relu",
        kernelInitializer: "orthogonal",
        name: "conv1d_2",
      })
    );

    model.add(
      tf.layers.lstm({
        units: 64, // Reduced from 128
        returnSequences: true,
        kernelInitializer: "orthogonal",
        kernelRegularizer: tf.regularizers.l2({ l2: 0.01 }), // Increased from 0.005
        name: "lstm1",
      })
    );
    model.add(tf.layers.dropout({ rate: 0.5 })); // Increased from 0.3

    model.add(
      tf.layers.lstm({
        units: 32, // Reduced from 64
        returnSequences: true,
        kernelInitializer: "orthogonal",
        kernelRegularizer: tf.regularizers.l2({ l2: 0.01 }),
        name: "lstm2",
      })
    );
    model.add(tf.layers.dropout({ rate: 0.5 }));

    model.add(
      tf.layers.lstm({
        units: 8, // Reduced from 16
        returnSequences: true,
        kernelInitializer: "orthogonal",
        kernelRegularizer: tf.regularizers.l2({ l2: 0.01 }),
        name: "lstm3",
      })
    );
    model.add(tf.layers.dropout({ rate: 0.5 }));

    model.add(
      tf.layers.timeDistributed({
        layer: tf.layers.dense({
          units: 16,
          activation: "relu",
          name: "time_distributed_dense",
        }),
        name: "time_distributed",
      })
    );
    model.add(tf.layers.flatten());

    model.add(tf.layers.batchNormalization({ name: "batchNormalization" }));

    model.add(
      tf.layers.dense({
        units: 24,
        activation: "relu",
        kernelInitializer: "heNormal",
        kernelRegularizer: tf.regularizers.l2({ l2: 0.01 }),
        name: "dense",
      })
    );
    model.add(tf.layers.dropout({ rate: 0.5 }));

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
