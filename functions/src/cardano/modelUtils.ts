import * as tf from "@tensorflow/tfjs-node";

export function createTradeModel(
  timesteps: number,
  featureDim: number
): tf.LayersModel {
  const input = tf.input({ shape: [timesteps, featureDim] });
  const conv1 = tf.layers
    .conv1d({
      name: "conv1d",
      filters: 32,
      kernelSize: 3,
      activation: "relu",
      kernelRegularizer: tf.regularizers.l2({ l2: 0.01 }),
      padding: "same",
    })
    .apply(input) as tf.SymbolicTensor;
  const pool1 = tf.layers
    .maxPooling1d({ name: "maxPooling1d", poolSize: 2, strides: 2 })
    .apply(conv1) as tf.SymbolicTensor;
  const lstm1 = tf.layers
    .lstm({
      name: "lstm1",
      units: 128,
      returnSequences: true,
      kernelRegularizer: tf.regularizers.l2({ l2: 0.01 }),
    })
    .apply(pool1) as tf.SymbolicTensor;
  const lstm2 = tf.layers
    .lstm({
      name: "lstm2",
      units: 32,
      returnSequences: false,
      kernelRegularizer: tf.regularizers.l2({ l2: 0.01 }),
    })
    .apply(lstm1) as tf.SymbolicTensor;
  const bn = tf.layers
    .batchNormalization({ name: "batchNormalization" })
    .apply(lstm2) as tf.SymbolicTensor;
  const dropout = tf.layers
    .dropout({ name: "dropout", rate: 0.3 })
    .apply(bn) as tf.SymbolicTensor;
  const dense1 = tf.layers
    .dense({
      name: "dense",
      units: 32,
      activation: "relu",
      kernelRegularizer: tf.regularizers.l2({ l2: 0.01 }),
    })
    .apply(dropout) as tf.SymbolicTensor;
  const output = tf.layers
    .dense({
      name: "dense_1",
      units: 2,
      activation: "softmax",
      kernelRegularizer: tf.regularizers.l2({ l2: 0.01 }),
    })
    .apply(dense1) as tf.SymbolicTensor;
  return tf.model({ inputs: input, outputs: output });
}
