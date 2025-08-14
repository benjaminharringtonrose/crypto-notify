import * as tf from "@tensorflow/tfjs-node";
import { TRAINING_CONFIG } from "../../constants";

export class GradientClippingCallback extends tf.CustomCallback {
  private model: tf.LayersModel | null = null;
  private clipNorm: number;
  private xVal: tf.Tensor;
  private yVal: tf.Tensor;
  private trainingLogger?: any; // Reference to TrainingLoggerCallback

  constructor(
    xVal: tf.Tensor,
    yVal: tf.Tensor,
    clipNorm: number = TRAINING_CONFIG.GRADIENT_CLIP_NORM
  ) {
    super({});
    this.xVal = xVal;
    this.yVal = yVal;
    this.clipNorm = clipNorm;
  }

  setModel(model: tf.LayersModel) {
    this.model = model;
  }

  setTrainingLogger(logger: any) {
    this.trainingLogger = logger;
  }

  private calculateLoss(predictions: tf.Tensor): tf.Scalar {
    return tf.losses.softmaxCrossEntropy(this.yVal, predictions).mean() as tf.Scalar;
  }

  async onEpochEnd(epoch: number, logs?: tf.Logs) {
    if (!this.model || !logs) return;
    if (epoch % 5 !== 0) return; // Log every 5 epochs to match PredictionLoggerCallback

    const trainableWeights = this.model.trainableWeights;
    if (trainableWeights.length === 0) return;

    await tf.tidy(() => {
      const computeLoss = () => {
        const predictions = this.model!.apply(this.xVal, {
          training: false,
        }) as tf.Tensor;
        const loss = this.calculateLoss(predictions);
        return loss;
      };

      const gradFn = tf.grads(computeLoss);
      const grads = gradFn(trainableWeights.map((w) => w.read()));

      const gradNorm = tf.sqrt(
        grads
          .map((g) => tf.sum(tf.square(g)))
          .reduce((acc, curr) => acc.add(curr), tf.scalar(0))
      );
      
      // Send gradient norm to training logger instead of logging directly
      if (this.trainingLogger) {
        this.trainingLogger.setGradientNorm(gradNorm.dataSync()[0]);
      }
      
      const clippedGrads: { [key: string]: tf.Tensor } = {};
      trainableWeights.forEach((weight, i) => {
        const grad = grads[i];
        clippedGrads[weight.name] = gradNorm.greater(this.clipNorm)
          ? grad.mul(this.clipNorm).div(gradNorm.add(tf.scalar(1e-6)))
          : grad;
      });

      this.model!.optimizer.applyGradients(clippedGrads);

      gradNorm.dispose();
      grads.forEach((g) => g.dispose());
    });
  }
}
