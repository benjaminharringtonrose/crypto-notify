import * as tf from "@tensorflow/tfjs-node";
import { MODEL_ARCHITECTURE_CONSTANTS } from "../constants";

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
        filters: MODEL_ARCHITECTURE_CONSTANTS.CONV1D_FILTERS_1,
        kernelSize: MODEL_ARCHITECTURE_CONSTANTS.CONV1D_KERNEL_SIZE_1,
        activation: "relu",
        kernelInitializer: "orthogonal",
        name: "conv1d",
      })
    );

    model.add(
      tf.layers.conv1d({
        filters: MODEL_ARCHITECTURE_CONSTANTS.CONV1D_FILTERS_2,
        kernelSize: MODEL_ARCHITECTURE_CONSTANTS.CONV1D_KERNEL_SIZE_2,
        activation: "relu",
        kernelInitializer: "orthogonal",
        name: "conv1d_2",
      })
    );

    model.add(
      tf.layers.lstm({
        units: MODEL_ARCHITECTURE_CONSTANTS.LSTM_UNITS_1,
        returnSequences: true,
        kernelInitializer: "orthogonal",
        kernelRegularizer: tf.regularizers.l2({
          l2: MODEL_ARCHITECTURE_CONSTANTS.L2_REGULARIZATION,
        }),
        name: "lstm1",
      })
    );
    model.add(
      tf.layers.dropout({ rate: MODEL_ARCHITECTURE_CONSTANTS.DROPOUT_RATE })
    );

    model.add(
      tf.layers.lstm({
        units: MODEL_ARCHITECTURE_CONSTANTS.LSTM_UNITS_2,
        returnSequences: true,
        kernelInitializer: "orthogonal",
        kernelRegularizer: tf.regularizers.l2({
          l2: MODEL_ARCHITECTURE_CONSTANTS.L2_REGULARIZATION,
        }),
        name: "lstm2",
      })
    );
    model.add(
      tf.layers.dropout({ rate: MODEL_ARCHITECTURE_CONSTANTS.DROPOUT_RATE })
    );

    model.add(
      tf.layers.lstm({
        units: MODEL_ARCHITECTURE_CONSTANTS.LSTM_UNITS_3,
        returnSequences: true,
        kernelInitializer: "orthogonal",
        kernelRegularizer: tf.regularizers.l2({
          l2: MODEL_ARCHITECTURE_CONSTANTS.L2_REGULARIZATION,
        }),
        name: "lstm3",
      })
    );
    model.add(
      tf.layers.dropout({ rate: MODEL_ARCHITECTURE_CONSTANTS.DROPOUT_RATE })
    );

    model.add(
      tf.layers.timeDistributed({
        layer: tf.layers.dense({
          units: MODEL_ARCHITECTURE_CONSTANTS.TIME_DISTRIBUTED_DENSE_UNITS,
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
        units: MODEL_ARCHITECTURE_CONSTANTS.DENSE_UNITS_1,
        activation: "relu",
        kernelInitializer: "heNormal",
        kernelRegularizer: tf.regularizers.l2({
          l2: MODEL_ARCHITECTURE_CONSTANTS.L2_REGULARIZATION,
        }),
        name: "dense",
      })
    );
    model.add(
      tf.layers.dropout({ rate: MODEL_ARCHITECTURE_CONSTANTS.DROPOUT_RATE })
    );

    model.add(
      tf.layers.dense({
        units: MODEL_ARCHITECTURE_CONSTANTS.OUTPUT_UNITS,
        activation: "softmax",
        kernelInitializer: "heNormal",
        name: "dense_1",
      })
    );

    return model;
  }
}
