import * as admin from "firebase-admin";
import * as tf from "@tensorflow/tfjs-node";
import dotenv from "dotenv";
import { PredictionLoggerCallback } from "./callbacks/PredictionLoggerCallback";
import { ExponentialDecayLRCallback } from "./callbacks/ExponentialDecayLRCallback";
import { GradientClippingCallback } from "./callbacks/GradientClippingCallback";
import { ModelConfig } from "../types";
import TradeModelFactory from "./TradeModelFactory";
import { FirebaseService } from "../api/FirebaseService";
import { FILE_NAMES, MODEL_CONFIG, TRAINING_CONFIG } from "../constants";
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
  private bestThreshold: number = 0.5;
  private bestWeights: tf.Tensor[] = []; // Store best weights
  private bestValF1Buy: number = -Infinity; // Track best F1 Buy

  constructor() {
    this.dataProcessor = new DataProcessor(
      this.config,
      TRAINING_CONFIG.START_DAYS_AGO
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
    const featureStats = this.dataProcessor.computeFeatureStats(X.flat(1));
    const X_mean = tf.tensor1d(featureStats.mean);
    const X_std = tf.tensor1d(featureStats.std);
    const X_normalized = X_tensor.sub(X_mean).div(X_std);

    try {
      const totalSamples = X.length;
      const numChunks = Math.ceil(
        totalSamples / TRAINING_CONFIG.SHUFFLE_CHUNK_SIZE
      );
      const chunkedIndices: number[][] = [];
      for (let i = 0; i < numChunks; i++) {
        const start = i * TRAINING_CONFIG.SHUFFLE_CHUNK_SIZE;
        const end = Math.min(
          (i + 1) * TRAINING_CONFIG.SHUFFLE_CHUNK_SIZE,
          totalSamples
        );
        chunkedIndices.push(
          Array.from({ length: end - start }, (_, j) => start + j)
        );
      }
      const shuffledChunks = [...chunkedIndices];
      tf.util.shuffle(shuffledChunks);
      const shuffledIndices = shuffledChunks.flat();
      const trainSize = Math.floor(totalSamples * TRAINING_CONFIG.TRAIN_SPLIT);
      const trainIndices = shuffledIndices.slice(0, trainSize);
      const valIndices = shuffledIndices.slice(trainSize);

      const X_train = tf.gather(X_normalized, trainIndices);
      const y_train = tf.gather(y_tensor, trainIndices);
      const X_val = tf.gather(X_normalized, valIndices);
      const y_val = tf.gather(y_tensor, valIndices);

      const y_train_array = await y_train.argMax(-1).data();
      const trainBuyCount = Array.from(y_train_array).filter(
        (l) => l === 1
      ).length;
      console.log(
        `Training set - Buy: ${trainBuyCount}, Sell: ${
          trainSize - trainBuyCount
        }, Buy Ratio: ${(trainBuyCount / trainSize).toFixed(3)}`
      );
      const y_val_array = await y_val.argMax(-1).data();
      const valBuyCount = Array.from(y_val_array).filter((l) => l === 1).length;
      console.log(
        `Validation set - Buy: ${valBuyCount}, Sell: ${
          totalSamples - trainSize - valBuyCount
        }, Buy Ratio: ${(valBuyCount / (totalSamples - trainSize)).toFixed(3)}`
      );

      const trainDataset = tf.data
        .zip({
          xs: tf.data.array(X_train.arraySync() as number[][][]),
          ys: tf.data.array(y_train.arraySync() as number[][]),
        })
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

      const predictionLoggerCallback = new PredictionLoggerCallback(
        X_val,
        y_val,
        this
      );
      const lrCallback = new ExponentialDecayLRCallback();
      const gradientClippingCallback = new GradientClippingCallback(
        this.lossFn,
        X_val,
        y_val
      );

      if (!this.model) throw new Error("Model initialization failed");

      this.model.compile({
        optimizer: tf.train.adam(this.config.initialLearningRate),
        loss: this.lossFn,
        metrics: [
          "binaryAccuracy",
          Metrics.precisionBuy,
          Metrics.precisionSell,
          Metrics.recallBuy,
          Metrics.recallSell,
          Metrics.customF1Buy,
          Metrics.customF1Sell,
        ],
      });

      predictionLoggerCallback.setModel(this.model);
      lrCallback.setModel(this.model);
      gradientClippingCallback.setModel(this.model);

      console.log("Model summary:");
      this.model.summary();

      console.log("Starting model training...");
      await this.model.fitDataset(trainDataset, {
        epochs: this.config.epochs,
        validationData: valDataset,
        callbacks: [
          {
            onEpochEnd: async (epoch: number, logs?: tf.Logs) => {
              if (logs) {
                const trainAcc = logs["binaryAccuracy"] || 0;
                const valAcc = logs["val_binaryAccuracy"] || 0;
                const accSpread = trainAcc - valAcc;
                console.log(
                  `Epoch ${epoch + 1} - Binary Accuracy: ${trainAcc.toFixed(
                    4
                  )}, Val Binary Accuracy: ${valAcc.toFixed(
                    4
                  )}, Spread: ${accSpread.toFixed(4)}`
                );

                // Manual best weights tracking
                const valF1Buy = logs["val_customF1Buy"] || 0;
                if (valF1Buy > this.bestValF1Buy) {
                  this.bestValF1Buy = valF1Buy;
                  this.bestWeights = this.model!.getWeights().map((w) =>
                    w.clone()
                  );
                  console.log(
                    `New best val_customF1Buy: ${this.bestValF1Buy.toFixed(
                      4
                    )} at epoch ${epoch + 1}`
                  );
                }

                await predictionLoggerCallback.onEpochEnd(epoch, logs);
                await lrCallback.onEpochEnd(epoch, logs);
                await gradientClippingCallback.onEpochEnd(epoch, logs);
              }
            },
            onTrainEnd: async () => {
              if (this.bestWeights.length > 0) {
                this.model!.setWeights(this.bestWeights);
                console.log(
                  `Restored weights from best val_customF1Buy: ${this.bestValF1Buy.toFixed(
                    4
                  )}`
                );
                this.bestWeights.forEach((w) => w.dispose());
              }
            },
          },
        ],
      });

      console.log("Model training completed.");
      await Metrics.evaluateModel(this.model, X_normalized, y_tensor);
      console.log(
        `Training Buy Ratio: ${y.filter((l) => l === 1).length / y.length}`
      );
      console.log(
        `Memory after training: ${
          tf.memory().numBytes / TRAINING_CONFIG.BYTES_TO_MB
        } MB`
      );

      return { X_mean, X_std };
    } finally {
      X_tensor.dispose();
      y_tensor.dispose();
      X_normalized.dispose();
    }
  }

  private async saveModelWeights(
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
    const file = this.bucket.file(FILE_NAMES.WEIGHTS);
    await file.save(weightsJson, { contentType: "application/json" });
    console.log("Model weights saved to Firebase Storage");
    X_mean.dispose();
    X_std.dispose();
  }

  public async train(): Promise<void> {
    try {
      const startTime = performance.now();
      const { X, y } = await this.dataProcessor.prepareData();
      const { X_mean, X_std } = await this.trainModel(X, y);
      await this.saveModelWeights(X_mean, X_std);
      const endTime = performance.now();
      const executionTime =
        (endTime - startTime) / TRAINING_CONFIG.MS_TO_SECONDS;
      console.log(`Execution time: ${executionTime} seconds`);
    } catch (error) {
      console.error("Training failed:", error);
      throw error;
    }
  }

  public getBestThreshold(): number {
    return this.bestThreshold;
  }
}
