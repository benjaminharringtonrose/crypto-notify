import * as tf from "@tensorflow/tfjs-node";
import { MODEL_CONFIG } from "../constants";

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
        filters: MODEL_CONFIG.CONV1D_FILTERS_1,
        kernelSize: MODEL_CONFIG.CONV1D_KERNEL_SIZE_1,
        activation: "relu",
        kernelInitializer: "heNormal", // Changed from orthogonal for better performance
        name: "conv1d",
      })
    );
    model.add(tf.layers.batchNormalization({ name: "bn_conv1" }));
    model.add(tf.layers.dropout({ rate: 0.2 })); // Added dropout after conv1

    model.add(
      tf.layers.conv1d({
        filters: MODEL_CONFIG.CONV1D_FILTERS_2,
        kernelSize: MODEL_CONFIG.CONV1D_KERNEL_SIZE_2,
        activation: "relu",
        kernelInitializer: "heNormal", // Changed from orthogonal
        name: "conv1d_2",
      })
    );
    model.add(tf.layers.batchNormalization({ name: "bn_conv2" }));
    model.add(tf.layers.dropout({ rate: 0.2 })); // Added dropout after conv2

    model.add(
      tf.layers.lstm({
        units: MODEL_CONFIG.LSTM_UNITS_1,
        returnSequences: true,
        kernelInitializer: "heNormal", // Changed from orthogonal
        kernelRegularizer: tf.regularizers.l2({
          l2: MODEL_CONFIG.L2_REGULARIZATION,
        }),
        recurrentRegularizer: tf.regularizers.l2({
          // Added recurrent regularization
          l2: MODEL_CONFIG.L2_REGULARIZATION * 0.5,
        }),
        name: "lstm1",
      })
    );
    model.add(tf.layers.batchNormalization({ name: "bn_lstm1" }));
    model.add(tf.layers.dropout({ rate: MODEL_CONFIG.DROPOUT_RATE }));

    model.add(
      tf.layers.lstm({
        units: MODEL_CONFIG.LSTM_UNITS_2,
        returnSequences: true,
        kernelInitializer: "heNormal", // Changed from orthogonal
        kernelRegularizer: tf.regularizers.l2({
          l2: MODEL_CONFIG.L2_REGULARIZATION,
        }),
        recurrentRegularizer: tf.regularizers.l2({
          // Added recurrent regularization
          l2: MODEL_CONFIG.L2_REGULARIZATION * 0.5,
        }),
        name: "lstm2",
      })
    );
    model.add(tf.layers.batchNormalization({ name: "bn_lstm2" }));
    model.add(tf.layers.dropout({ rate: MODEL_CONFIG.DROPOUT_RATE }));

    model.add(
      tf.layers.lstm({
        units: MODEL_CONFIG.LSTM_UNITS_3,
        returnSequences: true,
        kernelInitializer: "heNormal", // Changed from orthogonal
        kernelRegularizer: tf.regularizers.l2({
          l2: MODEL_CONFIG.L2_REGULARIZATION,
        }),
        recurrentRegularizer: tf.regularizers.l2({
          // Added recurrent regularization
          l2: MODEL_CONFIG.L2_REGULARIZATION * 0.5,
        }),
        name: "lstm3",
      })
    );
    model.add(tf.layers.batchNormalization({ name: "bn_lstm3" }));
    model.add(tf.layers.dropout({ rate: MODEL_CONFIG.DROPOUT_RATE }));

    model.add(
      tf.layers.timeDistributed({
        layer: tf.layers.dense({
          units: MODEL_CONFIG.TIME_DISTRIBUTED_DENSE_UNITS,
          activation: "relu",
          kernelRegularizer: tf.regularizers.l2({
            // Added regularization
            l2: MODEL_CONFIG.L2_REGULARIZATION,
          }),
          name: "time_distributed_dense",
        }),
        name: "time_distributed",
      })
    );
    model.add(tf.layers.flatten());
    model.add(tf.layers.batchNormalization({ name: "batchNormalization" }));

    model.add(
      tf.layers.dense({
        units: MODEL_CONFIG.DENSE_UNITS_1,
        activation: "relu",
        kernelInitializer: "heNormal",
        kernelRegularizer: tf.regularizers.l2({
          l2: MODEL_CONFIG.L2_REGULARIZATION,
        }),
        name: "dense",
      })
    );
    model.add(tf.layers.batchNormalization({ name: "bn_dense1" }));
    model.add(tf.layers.dropout({ rate: MODEL_CONFIG.DROPOUT_RATE }));

    model.add(
      tf.layers.dense({
        units: MODEL_CONFIG.OUTPUT_UNITS,
        activation: "softmax",
        kernelInitializer: "heNormal",
        name: "dense_1",
      })
    );

    return model;
  }
}
