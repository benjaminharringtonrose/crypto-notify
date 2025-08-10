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

    // Input layer with proper shape
    model.add(
      tf.layers.conv1d({
        inputShape: [this.timesteps, this.features],
        filters: MODEL_CONFIG.CONV1D_FILTERS_1,
        kernelSize: MODEL_CONFIG.CONV1D_KERNEL_SIZE_1,
        activation: "relu",
        kernelInitializer: "heNormal",
        kernelRegularizer: tf.regularizers.l2({
          l2: MODEL_CONFIG.L2_REGULARIZATION,
        }),
        name: "conv1d_input",
      })
    );
    model.add(tf.layers.batchNormalization({ name: "bn_conv1" }));
    model.add(tf.layers.dropout({ rate: 0.2 }));

    // Second convolutional layer for feature extraction
    model.add(
      tf.layers.conv1d({
        filters: MODEL_CONFIG.CONV1D_FILTERS_2,
        kernelSize: MODEL_CONFIG.CONV1D_KERNEL_SIZE_2,
        activation: "relu",
        kernelInitializer: "heNormal",
        kernelRegularizer: tf.regularizers.l2({
          l2: MODEL_CONFIG.L2_REGULARIZATION,
        }),
        name: "conv1d_2",
      })
    );
    model.add(tf.layers.batchNormalization({ name: "bn_conv2" }));
    model.add(tf.layers.dropout({ rate: 0.2 }));

    // First LSTM layer with return sequences for temporal modeling
    model.add(
      tf.layers.lstm({
        units: MODEL_CONFIG.LSTM_UNITS_1,
        returnSequences: true,
        kernelInitializer: "heNormal",
        kernelRegularizer: tf.regularizers.l2({
          l2: MODEL_CONFIG.L2_REGULARIZATION,
        }),
        recurrentRegularizer: tf.regularizers.l2({
          l2: MODEL_CONFIG.L2_REGULARIZATION * 0.5,
        }),
        dropout: 0.1,
        recurrentDropout: 0.1,
        name: "lstm1",
      })
    );
    model.add(tf.layers.batchNormalization({ name: "bn_lstm1" }));
    model.add(tf.layers.dropout({ rate: MODEL_CONFIG.DROPOUT_RATE }));

    // Second LSTM layer with return sequences
    model.add(
      tf.layers.lstm({
        units: MODEL_CONFIG.LSTM_UNITS_2,
        returnSequences: true,
        kernelInitializer: "heNormal",
        kernelRegularizer: tf.regularizers.l2({
          l2: MODEL_CONFIG.L2_REGULARIZATION,
        }),
        recurrentRegularizer: tf.regularizers.l2({
          l2: MODEL_CONFIG.L2_REGULARIZATION * 0.5,
        }),
        dropout: 0.1,
        recurrentDropout: 0.1,
        name: "lstm2",
      })
    );
    model.add(tf.layers.batchNormalization({ name: "bn_lstm2" }));
    model.add(tf.layers.dropout({ rate: MODEL_CONFIG.DROPOUT_RATE }));

    // Third LSTM layer without return sequences for final temporal features
    model.add(
      tf.layers.lstm({
        units: MODEL_CONFIG.LSTM_UNITS_3,
        returnSequences: false,
        kernelInitializer: "heNormal",
        kernelRegularizer: tf.regularizers.l2({
          l2: MODEL_CONFIG.L2_REGULARIZATION,
        }),
        recurrentRegularizer: tf.regularizers.l2({
          l2: MODEL_CONFIG.L2_REGULARIZATION * 0.5,
        }),
        dropout: 0.1,
        recurrentDropout: 0.1,
        name: "lstm3",
      })
    );
    model.add(tf.layers.batchNormalization({ name: "bn_lstm3" }));
    model.add(tf.layers.dropout({ rate: MODEL_CONFIG.DROPOUT_RATE }));

    // Dense layers for feature learning
    model.add(
      tf.layers.dense({
        units: MODEL_CONFIG.DENSE_UNITS_1,
        activation: "relu",
        kernelInitializer: "heNormal",
        kernelRegularizer: tf.regularizers.l2({
          l2: MODEL_CONFIG.L2_REGULARIZATION,
        }),
        name: "dense_1",
      })
    );
    model.add(tf.layers.batchNormalization({ name: "bn_dense1" }));
    model.add(tf.layers.dropout({ rate: MODEL_CONFIG.DROPOUT_RATE }));

    // Additional dense layer for better feature learning
    model.add(
      tf.layers.dense({
        units: Math.floor(MODEL_CONFIG.DENSE_UNITS_1 / 2),
        activation: "relu",
        kernelInitializer: "heNormal",
        kernelRegularizer: tf.regularizers.l2({
          l2: MODEL_CONFIG.L2_REGULARIZATION * 0.5,
        }),
        name: "dense_1_5",
      })
    );
    model.add(tf.layers.batchNormalization({ name: "bn_dense1_5" }));
    model.add(tf.layers.dropout({ rate: MODEL_CONFIG.DROPOUT_RATE * 0.5 }));

    // Output layer with softmax activation
    model.add(
      tf.layers.dense({
        units: MODEL_CONFIG.OUTPUT_UNITS,
        activation: "softmax",
        kernelInitializer: "heNormal",
        name: "output",
      })
    );

    return model;
  }
}
