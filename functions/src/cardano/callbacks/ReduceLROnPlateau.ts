import * as tf from "@tensorflow/tfjs-node";

export class ReduceLROnPlateau extends tf.CustomCallback {
  private patience: number;
  private factor: number;
  private minLr: number;
  private monitor: string;
  private best: number;
  private wait: number;
  private currentLr: number;
  private model!: tf.Sequential;

  constructor(
    args: tf.CustomCallbackArgs & {
      patience?: number;
      factor?: number;
      minLr?: number;
      monitor?: string;
      initialLr?: number;
    }
  ) {
    super(args);
    this.patience = args.patience ?? 10;
    this.factor = args.factor ?? 0.5;
    this.minLr = args.minLr ?? 0.0001;
    this.monitor = args.monitor ?? "val_loss";
    this.best = Infinity;
    this.wait = 0;
    this.currentLr = args.initialLr ?? 0.001;
  }

  setModel(model: tf.Sequential) {
    this.model = model;
  }

  async onEpochEnd(epoch: number, logs?: tf.Logs) {
    const current = logs?.[this.monitor];
    if (current === undefined) return;

    if (current < this.best) {
      this.best = current;
      this.wait = 0;
    } else {
      this.wait += 1;
      if (this.wait >= this.patience) {
        const newLr = Math.max(this.currentLr * this.factor, this.minLr);
        if (newLr !== this.currentLr) {
          this.currentLr = newLr;
          console.log(
            `Epoch ${epoch + 1}: Reducing learning rate to ${newLr.toFixed(6)}`
          );
          this.model.compile({
            optimizer: tf.train.adam(this.currentLr),
            loss: "categoricalCrossentropy",
            metrics: [
              tf.metrics.categoricalAccuracy,
              this.customPrecision,
              this.customRecall,
            ],
          });
        }
        this.wait = 0;
      }
    }
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
