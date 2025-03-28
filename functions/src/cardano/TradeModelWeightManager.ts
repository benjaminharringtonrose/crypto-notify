import * as tf from "@tensorflow/tfjs-node";
import { FirebaseService } from "../api/FirebaseService";
import { Bucket } from "@google-cloud/storage";
import { TRADE_PREDICTOR_WEIGHTS } from "../constants";

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
        tf.tensor3d(this.weights.conv1Weights, [5, 61, 12]),
        tf.tensor1d(this.weights.conv1Bias),
      ]);
    model
      .getLayer("conv1d_2")
      .setWeights([
        tf.tensor3d(this.weights.conv2Weights, [3, 12, 24]),
        tf.tensor1d(this.weights.conv2Bias),
      ]);
    model.getLayer("lstm1").setWeights([
      tf.tensor2d(this.weights.lstm1Weights, [24, 256]), // 64 units * 4 = 256
      tf.tensor2d(this.weights.lstm1RecurrentWeights, [64, 256]),
      tf.tensor1d(this.weights.lstm1Bias),
    ]);
    model.getLayer("lstm2").setWeights([
      tf.tensor2d(this.weights.lstm2Weights, [64, 128]), // 32 units * 4 = 128
      tf.tensor2d(this.weights.lstm2RecurrentWeights, [32, 128]),
      tf.tensor1d(this.weights.lstm2Bias),
    ]);
    model.getLayer("lstm3").setWeights([
      tf.tensor2d(this.weights.lstm3Weights, [32, 32]), // 8 units * 4 = 32
      tf.tensor2d(this.weights.lstm3RecurrentWeights, [8, 32]),
      tf.tensor1d(this.weights.lstm3Bias),
    ]);
    model
      .getLayer("time_distributed")
      .setWeights([
        tf.tensor2d(this.weights.timeDistributedWeights, [8, 16]),
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
        tf.tensor2d(this.weights.dense1Weights, [384, 24]),
        tf.tensor1d(this.weights.dense1Bias),
      ]);
    model
      .getLayer("dense_1")
      .setWeights([
        tf.tensor2d(this.weights.dense2Weights, [24, 2]),
        tf.tensor1d(this.weights.dense2Bias),
      ]);
  }

  public getFeatureMeans(): number[] {
    return this.weights?.featureMeans || Array(61).fill(0);
  }

  public getFeatureStds(): number[] {
    return this.weights?.featureStds || Array(61).fill(1);
  }
}
