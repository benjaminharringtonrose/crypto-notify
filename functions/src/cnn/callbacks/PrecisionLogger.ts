import * as tf from "@tensorflow/tfjs-node";

export class PrecisionLogger extends tf.CustomCallback {
  constructor(args: tf.CustomCallbackArgs) {
    super(args);
  }

  async onEpochEnd(epoch: number, logs?: tf.Logs) {
    console.log(
      `Epoch ${epoch + 1} Precision: ${logs?.customPrecision.toFixed(4)}`
    );
  }
}
