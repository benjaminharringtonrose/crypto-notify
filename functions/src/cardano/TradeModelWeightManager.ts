import * as tf from "@tensorflow/tfjs-node";
import { FirebaseService } from "../api/FirebaseService";
import { Bucket } from "@google-cloud/storage";
import { MODEL_CONFIG, TRADE_PREDICTOR_WEIGHTS } from "../constants";

export class ModelWeightManager {
  private weights: any;
  private bucket: Bucket;

  constructor() {
    this.bucket = FirebaseService.getInstance().getBucket();
  }

  public async loadWeights(): Promise<void> {
    const file = this.bucket.file(TRADE_PREDICTOR_WEIGHTS);
    const [weightsData] = await file.download();
    this.weights = JSON.parse(weightsData.toString("utf8")).weights;
    this.validateWeights();
  }

  private validateWeights(): void {
    if (
      !this.weights ||
      Object.values(this.weights).some((w: any) => !w || isNaN(w[0]))
    ) {
      console.warn(
        "Invalid weights detected, falling back to random initialization."
      );
      this.weights = null;
    }
  }

  public setWeights(model: tf.LayersModel): void {
    if (!this.weights) {
      console.log("Weights not loaded, using random initialization.");
      return;
    }

    model
      .getLayer("conv1d")
      .setWeights([
        tf.tensor3d(
          this.weights.conv1Weights,
          MODEL_CONFIG.CONV1D_1_WEIGHT_SHAPE
        ),
        tf.tensor1d(this.weights.conv1Bias),
      ]);
    model
      .getLayer("conv1d_2")
      .setWeights([
        tf.tensor3d(
          this.weights.conv2Weights,
          MODEL_CONFIG.CONV1D_2_WEIGHT_SHAPE
        ),
        tf.tensor1d(this.weights.conv2Bias),
      ]);
    model
      .getLayer("lstm1")
      .setWeights([
        tf.tensor2d(this.weights.lstm1Weights, MODEL_CONFIG.LSTM1_WEIGHT_SHAPE),
        tf.tensor2d(
          this.weights.lstm1RecurrentWeights,
          MODEL_CONFIG.LSTM1_RECURRENT_SHAPE
        ),
        tf.tensor1d(this.weights.lstm1Bias),
      ]);
    model
      .getLayer("lstm2")
      .setWeights([
        tf.tensor2d(this.weights.lstm2Weights, MODEL_CONFIG.LSTM2_WEIGHT_SHAPE),
        tf.tensor2d(
          this.weights.lstm2RecurrentWeights,
          MODEL_CONFIG.LSTM2_RECURRENT_SHAPE
        ),
        tf.tensor1d(this.weights.lstm2Bias),
      ]);
    model
      .getLayer("lstm3")
      .setWeights([
        tf.tensor2d(this.weights.lstm3Weights, MODEL_CONFIG.LSTM3_WEIGHT_SHAPE),
        tf.tensor2d(
          this.weights.lstm3RecurrentWeights,
          MODEL_CONFIG.LSTM3_RECURRENT_SHAPE
        ),
        tf.tensor1d(this.weights.lstm3Bias),
      ]);
    model
      .getLayer("time_distributed")
      .setWeights([
        tf.tensor2d(
          this.weights.timeDistributedWeights,
          MODEL_CONFIG.TIME_DISTRIBUTED_WEIGHT_SHAPE
        ),
        tf.tensor1d(this.weights.timeDistributedBias),
      ]);
    model
      .getLayer("batchNormalization")
      .setWeights([
        tf.tensor1d(this.weights.bnGamma),
        tf.tensor1d(this.weights.bnBeta),
        tf.tensor1d(this.weights.bnMovingMean),
        tf.tensor1d(this.weights.bnMovingVariance),
      ]);
    model
      .getLayer("dense")
      .setWeights([
        tf.tensor2d(
          this.weights.dense1Weights,
          MODEL_CONFIG.DENSE_1_WEIGHT_SHAPE
        ),
        tf.tensor1d(this.weights.dense1Bias),
      ]);
    model
      .getLayer("dense_1")
      .setWeights([
        tf.tensor2d(
          this.weights.dense2Weights,
          MODEL_CONFIG.DENSE_2_WEIGHT_SHAPE
        ),
        tf.tensor1d(this.weights.dense2Bias),
      ]);
  }

  public getFeatureMeans(): number[] {
    return (
      this.weights?.featureMeans || Array(MODEL_CONFIG.FEATURE_COUNT).fill(0)
    );
  }

  public getFeatureStds(): number[] {
    return (
      this.weights?.featureStds || Array(MODEL_CONFIG.FEATURE_COUNT).fill(1)
    );
  }
}
