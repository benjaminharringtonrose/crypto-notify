import * as tf from "@tensorflow/tfjs-node";
import { MODEL_CONFIG, TRAINING_CONFIG } from "../constants";

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

    // Enhanced feature extraction layer (replaces attention mechanism)
    if (TRAINING_CONFIG.USE_ATTENTION) {
      model.add(
        tf.layers.dense({
          units: MODEL_CONFIG.ATTENTION_UNITS_1,
          activation: "tanh",
          kernelInitializer: "heNormal",
          kernelRegularizer: tf.regularizers.l2({
            l2: MODEL_CONFIG.L2_REGULARIZATION,
          }),
          name: "enhanced_features1",
        })
      );
      model.add(tf.layers.batchNormalization({ name: "bn_enhanced1" }));
      model.add(tf.layers.dropout({ rate: TRAINING_CONFIG.ATTENTION_DROPOUT }));
    }

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

    // Second enhanced feature layer (replaces attention mechanism)
    if (TRAINING_CONFIG.USE_ATTENTION) {
      model.add(
        tf.layers.dense({
          units: MODEL_CONFIG.ATTENTION_UNITS_2,
          activation: "tanh",
          kernelInitializer: "heNormal",
          kernelRegularizer: tf.regularizers.l2({
            l2: MODEL_CONFIG.L2_REGULARIZATION,
          }),
          name: "enhanced_features2",
        })
      );
      model.add(tf.layers.batchNormalization({ name: "bn_enhanced2" }));
      model.add(tf.layers.dropout({ rate: TRAINING_CONFIG.ATTENTION_DROPOUT }));
    }

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

    // Final dense layers with improved architecture
    model.add(
      tf.layers.dense({
        units: MODEL_CONFIG.DENSE_UNITS_1,
        activation: "relu",
        kernelInitializer: "heNormal",
        kernelRegularizer: tf.regularizers.l2({
          l2: MODEL_CONFIG.L2_REGULARIZATION,
        }),
        name: "dense1",
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
          l2: MODEL_CONFIG.L2_REGULARIZATION,
        }),
        name: "dense2",
      })
    );
    model.add(tf.layers.batchNormalization({ name: "bn_dense2" }));
    model.add(tf.layers.dropout({ rate: MODEL_CONFIG.DROPOUT_RATE * 0.5 }));

    // Output layer with linear activation for logits (temperature scaling applied in predictor)
    model.add(
      tf.layers.dense({
        units: MODEL_CONFIG.OUTPUT_UNITS,
        activation: "linear", // Use linear activation for logits
        kernelInitializer: "glorotNormal",
        kernelRegularizer: tf.regularizers.l2({
          l2: MODEL_CONFIG.L2_REGULARIZATION * 0.5, // Reduced regularization for output
        }),
        name: "output",
      })
    );

    return model;
  }
}
