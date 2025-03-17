import * as tf from "@tensorflow/tfjs-node";

export class ExponentialDecayLR extends tf.CustomCallback {
  private initialLr: number;
  private decayRate: number;
  private decaySteps: number;
  private model!: tf.Sequential;

  constructor(initialLr: number, decaySteps: number, decayRate: number) {
    super({});
    this.initialLr = initialLr;
    this.decayRate = decayRate;
    this.decaySteps = decaySteps;
  }

  setModel(model: tf.Sequential) {
    this.model = model;
  }

  async onEpochBegin(epoch: number) {
    const steps = epoch * (1807 / 32);
    const newLr =
      this.initialLr * Math.pow(1 - this.decayRate, steps / this.decaySteps);
    console.log(
      `Epoch ${epoch + 1}: Setting learning rate to ${newLr.toFixed(6)}`
    );
    this.model.compile({
      optimizer: tf.train.adam(newLr),
      loss: "categoricalCrossentropy",
      metrics: [
        tf.metrics.categoricalAccuracy,
        this.customPrecision,
        this.customRecall,
      ],
    });
  }

  private customPrecision(yTrue: tf.Tensor, yPred: tf.Tensor): tf.Scalar {
    const truePos = yTrue.mul(yPred.round()).sum();
    const predPos = yPred.round().sum();
    return truePos.div(predPos.add(1e-6)) as tf.Scalar;
  }

  private customRecall(yTrue: tf.Tensor, yPred: tf.Tensor): tf.Scalar {
    const truePos = yTrue.mul(yPred.round()).sum();
    const actualPos = yTrue.sum();
    return truePos.div(actualPos.add(1e-6)) as tf.Scalar;
  }
}
