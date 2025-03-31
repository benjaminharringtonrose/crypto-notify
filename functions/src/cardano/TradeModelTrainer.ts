import * as admin from "firebase-admin";
import * as tf from "@tensorflow/tfjs-node";
import dotenv from "dotenv";
import { BestWeightsCallback } from "./callbacks/BestWeightsCallback";
import { PredictionLoggerCallback } from "./callbacks/PredictionLoggerCallback";
import { CyclicLearningRateCallback } from "./callbacks/CyclicLearningRateCallback";
import TradeModelFactory from "./TradeModelFactory";
import { ModelConfig } from "../types";
import { FirebaseService } from "../api/FirebaseService";
import {
  MODEL_CONFIG,
  TRADE_PREDICTOR_WEIGHTS,
  TRAINING_CONFIG,
} from "../constants";
import { DataProcessor } from "./DataProcessor";
import { Metrics } from "./Metrics";

dotenv.config();

FirebaseService.getInstance();

export class TradeModelTrainer {
  private readonly config: ModelConfig = {
    timesteps: MODEL_CONFIG.TIMESTEPS,
    epochs: TRAINING_CONFIG.EPOCHS,
    batchSize: TRAINING_CONFIG.BATCH_SIZE,
    initialLearningRate: TRAINING_CONFIG.INITIAL_LEARNING_RATE,
  };
  private bucket = admin.storage().bucket();
  private model: tf.LayersModel | null = null;
  private lossFn = Metrics.focalLoss.bind(Metrics);
  private dataProcessor: DataProcessor;

  constructor() {
    this.dataProcessor = new DataProcessor(
      this.config,
      TRAINING_CONFIG.LOOKBACK_DAYS,
      TRAINING_CONFIG.PREDICTION_DAYS
    );
    console.log("TradeModelTrainer initialized");
  }

  private async trainModel(
    X: number[][][],
    y: number[]
  ): Promise<{ X_mean: tf.Tensor; X_std: tf.Tensor }> {
    console.log(
      `Input data dimensions: [${X.length}, ${X[0].length}, ${X[0][0].length}]`
    );
    if (X[0].length !== this.config.timesteps) {
      throw new Error(
        `Data timestep mismatch: expected ${this.config.timesteps}, got ${X[0].length}`
      );
    }
    const X_tensor = tf.tensor3d(X, [
      X.length,
      this.config.timesteps,
      MODEL_CONFIG.FEATURE_COUNT,
    ]);
    const y_tensor = tf.tensor2d(
      y.map((label) => [label === 0 ? 1 : 0, label === 1 ? 1 : 0]),
      [y.length, TRAINING_CONFIG.OUTPUT_CLASSES]
    );
    const X_mean = tf.tensor1d(
      this.dataProcessor.computeFeatureStats(X.flat(1)).mean
    );
    const X_std = tf.tensor1d(
      this.dataProcessor.computeFeatureStats(X.flat(1)).std
    );
    const X_normalized = X_tensor.sub(X_mean).div(X_std);

    try {
      const totalSamples = X.length;
      const trainSize = Math.floor(totalSamples * TRAINING_CONFIG.TRAIN_SPLIT);
      const indices = Array.from({ length: totalSamples }, (_, i) => i);
      tf.util.shuffle(indices);

      const trainIndices = indices.slice(0, trainSize);
      const valIndices = indices.slice(trainSize);

      const X_train = tf.gather(X_normalized, trainIndices);
      const y_train = tf.gather(y_tensor, trainIndices);
      const X_val = tf.gather(X_normalized, valIndices);
      const y_val = tf.gather(y_tensor, valIndices);

      const trainDataset = tf.data
        .zip({
          xs: tf.data.array(X_train.arraySync() as number[][][]),
          ys: tf.data.array(y_train.arraySync() as number[][]),
        })
        .shuffle(trainSize)
        .batch(this.config.batchSize)
        .prefetch(TRAINING_CONFIG.PREFETCH_BUFFER);

      const valDataset = tf.data
        .zip({
          xs: tf.data.array(X_val.arraySync() as number[][][]),
          ys: tf.data.array(y_val.arraySync() as number[][]),
        })
        .batch(this.config.batchSize);

      const factory = new TradeModelFactory(
        this.config.timesteps,
        MODEL_CONFIG.FEATURE_COUNT
      );
      this.model = factory.createModel();

      const bestWeightsCallback = new BestWeightsCallback();
      const predictionLoggerCallback = new PredictionLoggerCallback(X_val);
      const cyclicLRCallback = new CyclicLearningRateCallback(
        TRAINING_CONFIG.MIN_LEARNING_RATE,
        this.config.initialLearningRate,
        TRAINING_CONFIG.CYCLIC_LR_STEP_SIZE
      );

      if (!this.model) throw new Error("Model initialization failed");

      this.model.compile({
        optimizer: tf.train.adam(this.config.initialLearningRate),
        loss: this.lossFn,
        metrics: [
          tf.metrics.binaryAccuracy,
          Metrics.customPrecision,
          Metrics.customRecall,
          Metrics.customF1,
        ],
      });

      bestWeightsCallback.setModel(this.model);
      predictionLoggerCallback.setModel(this.model);
      cyclicLRCallback.setModel(this.model);

      console.log("Model summary:");
      this.model.summary();

      console.log("Starting model training...");
      await this.model.fitDataset(trainDataset, {
        epochs: this.config.epochs,
        validationData: valDataset,
        callbacks: [
          tf.callbacks.earlyStopping({
            monitor: "val_loss",
            patience: TRAINING_CONFIG.PATIENCE,
          }),
          bestWeightsCallback,
          predictionLoggerCallback,
          cyclicLRCallback,
        ],
      });

      bestWeightsCallback.applyBestWeights(this.model);
      console.log("Model training completed.");

      await Metrics.evaluateModel(this.model, X_normalized, y_tensor);

      console.log(
        `Training Buy Ratio: ${y.filter((l) => l === 1).length / y.length}`
      );

      console.log(
        "Memory after training:",
        tf.memory().numBytes / TRAINING_CONFIG.BYTES_TO_MB,
        "MB"
      );

      return { X_mean, X_std };
    } finally {
      X_tensor.dispose();
      y_tensor.dispose();
      X_normalized.dispose();
    }
  }

  private async saveModelWeights(
    featureStats: { mean: number[]; std: number[] },
    X_mean: tf.Tensor,
    X_std: tf.Tensor
  ): Promise<void> {
    if (!this.model) throw new Error("Model not initialized");
    const weights = {
      conv1Weights: Array.from(
        await this.model.getLayer("conv1d").getWeights()[0].data()
      ),
      conv1Bias: Array.from(
        await this.model.getLayer("conv1d").getWeights()[1].data()
      ),
      conv2Weights: Array.from(
        await this.model.getLayer("conv1d_2").getWeights()[0].data()
      ),
      conv2Bias: Array.from(
        await this.model.getLayer("conv1d_2").getWeights()[1].data()
      ),
      lstm1Weights: Array.from(
        await this.model.getLayer("lstm1").getWeights()[0].data()
      ),
      lstm1RecurrentWeights: Array.from(
        await this.model.getLayer("lstm1").getWeights()[1].data()
      ),
      lstm1Bias: Array.from(
        await this.model.getLayer("lstm1").getWeights()[2].data()
      ),
      lstm2Weights: Array.from(
        await this.model.getLayer("lstm2").getWeights()[0].data()
      ),
      lstm2RecurrentWeights: Array.from(
        await this.model.getLayer("lstm2").getWeights()[1].data()
      ),
      lstm2Bias: Array.from(
        await this.model.getLayer("lstm2").getWeights()[2].data()
      ),
      lstm3Weights: Array.from(
        await this.model.getLayer("lstm3").getWeights()[0].data()
      ),
      lstm3RecurrentWeights: Array.from(
        await this.model.getLayer("lstm3").getWeights()[1].data()
      ),
      lstm3Bias: Array.from(
        await this.model.getLayer("lstm3").getWeights()[2].data()
      ),
      timeDistributedWeights: Array.from(
        await this.model.getLayer("time_distributed").getWeights()[0].data()
      ),
      timeDistributedBias: Array.from(
        await this.model.getLayer("time_distributed").getWeights()[1].data()
      ),
      bnGamma: Array.from(
        await this.model.getLayer("batchNormalization").getWeights()[0].data()
      ),
      bnBeta: Array.from(
        await this.model.getLayer("batchNormalization").getWeights()[1].data()
      ),
      bnMovingMean: Array.from(
        await this.model.getLayer("batchNormalization").getWeights()[2].data()
      ),
      bnMovingVariance: Array.from(
        await this.model.getLayer("batchNormalization").getWeights()[3].data()
      ),
      dense1Weights: Array.from(
        await this.model.getLayer("dense").getWeights()[0].data()
      ),
      dense1Bias: Array.from(
        await this.model.getLayer("dense").getWeights()[1].data()
      ),
      dense2Weights: Array.from(
        await this.model.getLayer("dense_1").getWeights()[0].data()
      ),
      dense2Bias: Array.from(
        await this.model.getLayer("dense_1").getWeights()[1].data()
      ),
      featureMeans: Array.from(await X_mean.data()),
      featureStds: Array.from(await X_std.data()),
    };
    const weightsJson = JSON.stringify({ weights });
    const file = this.bucket.file(TRADE_PREDICTOR_WEIGHTS);
    await file.save(weightsJson, { contentType: "application/json" });
    console.log("Model weights saved to Firebase Storage");
    X_mean.dispose();
    X_std.dispose();
  }

  public async train(): Promise<void> {
    try {
      const startTime = performance.now();
      const { X, y, featureStats } = await this.dataProcessor.prepareData();
      const { X_mean, X_std } = await this.trainModel(X, y);
      await this.saveModelWeights(featureStats, X_mean, X_std);
      const endTime = performance.now();
      const executionTime =
        (endTime - startTime) / TRAINING_CONFIG.MS_TO_SECONDS;
      console.log(`Execution time: ${executionTime} seconds`);
    } catch (error) {
      console.error("Training failed:", error);
      throw error;
    }
  }
}
