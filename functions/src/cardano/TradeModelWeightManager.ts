import * as tf from "@tensorflow/tfjs-node";
import { FirebaseService } from "../api/FirebaseService";
import { Bucket } from "@google-cloud/storage";
import { FILE_NAMES, MODEL_CONFIG } from "../constants";

export class ModelWeightManager {
  private weights: any;
  private bucket: Bucket;

  constructor() {
    this.bucket = FirebaseService.getInstance().getBucket();
  }

  public async loadWeights(): Promise<void> {
    const file = this.bucket.file(FILE_NAMES.WEIGHTS);
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

    try {
      // Debug: Log the actual weights data
      console.log("Loading weights with shapes:");
      console.log("conv1Weights length:", this.weights.conv1Weights?.length);
      console.log("Expected conv1 shape:", MODEL_CONFIG.CONV1D_1_WEIGHT_SHAPE);
      console.log(
        "Expected conv1 size:",
        MODEL_CONFIG.CONV1D_1_WEIGHT_SHAPE.reduce((a, b) => a * b, 1)
      );

      // Check if weights match expected shape
      const expectedConv1Size = MODEL_CONFIG.CONV1D_1_WEIGHT_SHAPE.reduce(
        (a, b) => a * b,
        1
      );
      if (this.weights.conv1Weights?.length !== expectedConv1Size) {
        console.warn(
          `Conv1 weights size mismatch: expected ${expectedConv1Size}, got ${this.weights.conv1Weights?.length}`
        );
        console.warn("Using random initialization due to shape mismatch");
        return;
      }

      model
        .getLayer("conv1d_input")
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
          tf.tensor2d(
            this.weights.lstm1Weights,
            MODEL_CONFIG.LSTM1_WEIGHT_SHAPE
          ),
          tf.tensor2d(
            this.weights.lstm1RecurrentWeights,
            MODEL_CONFIG.LSTM1_RECURRENT_SHAPE
          ),
          tf.tensor1d(this.weights.lstm1Bias),
        ]);
      model
        .getLayer("lstm2")
        .setWeights([
          tf.tensor2d(
            this.weights.lstm2Weights,
            MODEL_CONFIG.LSTM2_WEIGHT_SHAPE
          ),
          tf.tensor2d(
            this.weights.lstm2RecurrentWeights,
            MODEL_CONFIG.LSTM2_RECURRENT_SHAPE
          ),
          tf.tensor1d(this.weights.lstm2Bias),
        ]);
      model
        .getLayer("lstm3")
        .setWeights([
          tf.tensor2d(
            this.weights.lstm3Weights,
            MODEL_CONFIG.LSTM3_WEIGHT_SHAPE
          ),
          tf.tensor2d(
            this.weights.lstm3RecurrentWeights,
            MODEL_CONFIG.LSTM3_RECURRENT_SHAPE
          ),
          tf.tensor1d(this.weights.lstm3Bias),
        ]);
      // Set batch normalization weights for conv layers
      model
        .getLayer("bn_conv1")
        .setWeights([
          tf.tensor1d(this.weights.bnConv1Gamma),
          tf.tensor1d(this.weights.bnConv1Beta),
          tf.tensor1d(this.weights.bnConv1MovingMean),
          tf.tensor1d(this.weights.bnConv1MovingVariance),
        ]);
      model
        .getLayer("bn_conv2")
        .setWeights([
          tf.tensor1d(this.weights.bnConv2Gamma),
          tf.tensor1d(this.weights.bnConv2Beta),
          tf.tensor1d(this.weights.bnConv2MovingMean),
          tf.tensor1d(this.weights.bnConv2MovingVariance),
        ]);

      // Set batch normalization weights for LSTM layers
      model
        .getLayer("bn_lstm1")
        .setWeights([
          tf.tensor1d(this.weights.bnLstm1Gamma),
          tf.tensor1d(this.weights.bnLstm1Beta),
          tf.tensor1d(this.weights.bnLstm1MovingMean),
          tf.tensor1d(this.weights.bnLstm1MovingVariance),
        ]);
      model
        .getLayer("bn_lstm2")
        .setWeights([
          tf.tensor1d(this.weights.bnLstm2Gamma),
          tf.tensor1d(this.weights.bnLstm2Beta),
          tf.tensor1d(this.weights.bnLstm2MovingMean),
          tf.tensor1d(this.weights.bnLstm2MovingVariance),
        ]);
      model
        .getLayer("bn_lstm3")
        .setWeights([
          tf.tensor1d(this.weights.bnLstm3Gamma),
          tf.tensor1d(this.weights.bnLstm3Beta),
          tf.tensor1d(this.weights.bnLstm3MovingMean),
          tf.tensor1d(this.weights.bnLstm3MovingVariance),
        ]);

      // Set dense layer weights
      model
        .getLayer("dense_1")
        .setWeights([
          tf.tensor2d(
            this.weights.dense1Weights,
            MODEL_CONFIG.DENSE_1_WEIGHT_SHAPE
          ),
          tf.tensor1d(this.weights.dense1Bias),
        ]);

      // Set batch normalization for dense layers
      model
        .getLayer("bn_dense1")
        .setWeights([
          tf.tensor1d(this.weights.bnDense1Gamma),
          tf.tensor1d(this.weights.bnDense1Beta),
          tf.tensor1d(this.weights.bnDense1MovingMean),
          tf.tensor1d(this.weights.bnDense1MovingVariance),
        ]);

      model
        .getLayer("dense_1_5")
        .setWeights([
          tf.tensor2d(
            this.weights.dense2Weights,
            MODEL_CONFIG.DENSE_2_WEIGHT_SHAPE
          ),
          tf.tensor1d(this.weights.dense2Bias),
        ]);

      model
        .getLayer("bn_dense1_5")
        .setWeights([
          tf.tensor1d(this.weights.bnDense2Gamma),
          tf.tensor1d(this.weights.bnDense2Beta),
          tf.tensor1d(this.weights.bnDense2MovingVariance),
          tf.tensor1d(this.weights.bnDense2MovingMean),
        ]);

      // Set output layer weights
      model
        .getLayer("output")
        .setWeights([
          tf.tensor2d(
            this.weights.outputWeights,
            MODEL_CONFIG.DENSE_2_WEIGHT_SHAPE
          ),
          tf.tensor1d(this.weights.outputBias),
        ]);
    } catch (error) {
      console.error("Error setting weights:", error);
      console.warn("Using random initialization due to error");
      return;
    }
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
