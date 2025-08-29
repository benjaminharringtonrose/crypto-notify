import * as admin from "firebase-admin";
import * as tf from "@tensorflow/tfjs-node";
import dotenv from "dotenv";

import { ExponentialDecayLRCallback } from "./callbacks/ExponentialDecayLRCallback";
import { GradientClippingCallback } from "./callbacks/GradientClippingCallback";
// import { CurriculumLearningCallback } from "./callbacks/CurriculumLearningCallback"; // RECOVERY-3: Disabled
import { TrainingLoggerCallback } from "./callbacks/TrainingLoggerCallback";
import { ModelConfig } from "../types";
import TradeModelFactory from "./TradeModelFactory";
import { FirebaseService } from "../api/FirebaseService";
import { FILE_NAMES, MODEL_CONFIG, TRAINING_CONFIG } from "../constants";
import { DataProcessor } from "./DataProcessor";
import { Metrics } from "./Metrics";
import { FeatureDetector } from "./FeatureDetector";

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
  private dataProcessor: DataProcessor;
  private bestThreshold: number = 0.5;
  private bestWeights: tf.Tensor[] = []; // Store best weights
  private bestValF1Buy: number = -Infinity; // Track best F1 Buy
  private bestValLoss: number = Infinity; // Track best validation loss
  private patienceCounter: number = 0; // Early stopping counter
  private seed: number;
  private currentFeatureName?: string; // For tracking which feature is being evaluated

  // Phase 1: Dynamic feature management
  private selectedFeatureIndices: number[] = [];
  private dynamicFeatureCount: number = 0;

  // Final metrics for analysis
  private finalMetrics: {
    balancedAccuracy: number;
    buyF1: number;
    sellF1: number;
    combinedF1: number;
    matthewsCorrelation: number;
    buyPrecision: number;
    sellPrecision: number;
    buyRecall: number;
    sellRecall: number;
    finalEpoch: number;
  } = {
    balancedAccuracy: 0,
    buyF1: 0,
    sellF1: 0,
    combinedF1: 0,
    matthewsCorrelation: 0,
    buyPrecision: 0,
    sellPrecision: 0,
    buyRecall: 0,
    sellRecall: 0,
    finalEpoch: 0,
  };

  constructor(seed?: number) {
    // Use provided seed or default to 42 for consistency
    this.seed = seed || 42;
    tf.randomUniform([1, 1], 0, 1, "float32", this.seed); // Seed TensorFlow random
    console.log(
      `🔧 RECOVERY-2: Deterministic seeding enabled with seed: ${this.seed}`
    );

    this.dataProcessor = new DataProcessor(
      this.config,
      TRAINING_CONFIG.START_DAYS_AGO
    );
    console.log("TradeModelTrainer initialized with Phase 1 enhancements");
  }

  /**
   * Set the current feature name being evaluated (for logging purposes)
   */
  public setCurrentFeatureName(featureName: string): void {
    this.currentFeatureName = featureName;
  }

  /**
   * Initialize dynamic feature detection before training
   */
  private async initializeFeatureDetection(): Promise<number> {
    const featureCount = await FeatureDetector.detectFeatureCount();
    console.log(
      `🔍 Dynamic feature detection: ${featureCount} features detected`
    );
    return featureCount;
  }

  private async trainModel(
    X: number[][][],
    y: number[]
  ): Promise<{ X_mean: tf.Tensor; X_std: tf.Tensor }> {
    // Phase 1: First compute feature stats to get selected features
    console.log("🔧 Phase 1: Computing feature statistics and selection...");
    const featureStats = this.dataProcessor.computeFeatureStats(
      X.flat(1) // Use original X for feature stats computation
    );

    // Get selected feature indices from the computed stats
    this.selectedFeatureIndices = featureStats.selectedFeatures;
    this.dynamicFeatureCount = this.selectedFeatureIndices.length;

    console.log(
      `🔧 Phase 1: Selected ${this.dynamicFeatureCount} features from ${X[0][0].length} original features`
    );

    // Phase 1: Apply feature selection to training data
    const selectedX = X.map((sample) =>
      sample.map((timestep) =>
        this.selectedFeatureIndices.map((index) => timestep[index])
      )
    );

    // Dynamic feature detection and validation
    const detectedFeatureCount = FeatureDetector.getFeatureCount();
    const actualFeatureCount = selectedX[0][0].length;

    console.log(
      `Input data dimensions: [${selectedX.length}, ${selectedX[0].length}, ${actualFeatureCount}]`
    );
    console.log(
      `🔍 Expected features: ${detectedFeatureCount}, Actual: ${actualFeatureCount}`
    );

    if (selectedX[0].length !== this.config.timesteps) {
      throw new Error(
        `Data timestep mismatch: expected ${this.config.timesteps}, got ${selectedX[0].length}`
      );
    }

    // Phase 1: Use dynamic feature count for model creation
    const X_tensor = tf.tensor3d(selectedX, [
      selectedX.length,
      this.config.timesteps,
      this.dynamicFeatureCount, // Use dynamic feature count!
    ]);
    const y_tensor = tf.tensor2d(
      y.map((label) => [label === 0 ? 1 : 0, label === 1 ? 1 : 0]),
      [y.length, TRAINING_CONFIG.OUTPUT_CLASSES]
    );

    // Extract only the selected features from the computed stats
    const selectedMeans = this.selectedFeatureIndices.map(
      (i) => featureStats.mean[i]
    );
    const selectedStds = this.selectedFeatureIndices.map(
      (i) => featureStats.std[i]
    );

    const X_mean = tf.tensor1d(selectedMeans);
    const X_std = tf.tensor1d(selectedStds);
    const X_normalized = X_tensor.sub(X_mean).div(X_std);

    try {
      const totalSamples = selectedX.length;
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

      // Phase 1: Use dynamic feature count for model creation
      const factory = new TradeModelFactory(
        this.config.timesteps,
        this.dynamicFeatureCount // Use dynamic feature count!
      );
      this.model = factory.createModel(this.seed);

      // Create centralized training logger
      const trainingLoggerCallback = new TrainingLoggerCallback(
        X_val,
        y_val,
        this
      );

      const lrCallback = new ExponentialDecayLRCallback();
      const gradientClippingCallback = new GradientClippingCallback(
        X_val,
        y_val
      );
      // RECOVERY-3: Disable curriculum learning to prevent class collapse
      // const curriculumLearningCallback = new CurriculumLearningCallback(
      //   this.config.epochs,
      //   this.dataProcessor
      // );
      console.log(
        "🔧 RECOVERY-3: Curriculum learning disabled to stabilize class balance"
      );

      if (!this.model) throw new Error("Model initialization failed");

      this.model.compile({
        optimizer: tf.train.adam(this.config.initialLearningRate),
        loss: (yTrue: tf.Tensor, yPred: tf.Tensor) => {
          return Metrics.focalLoss(
            yTrue,
            yPred,
            TRAINING_CONFIG.GAMMA,
            TRAINING_CONFIG.ALPHA
          );
        },
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

      trainingLoggerCallback.setModel(this.model);
      lrCallback.setModel(this.model);
      gradientClippingCallback.setModel(this.model);
      // curriculumLearningCallback.setModel(this.model); // DISABLED

      // Connect gradient clipping callback to training logger
      gradientClippingCallback.setTrainingLogger(trainingLoggerCallback);

      console.log("Model summary:");
      this.model.summary();

      console.log("Starting model training...");
      await this.model.fitDataset(trainDataset, {
        epochs: this.config.epochs,
        validationData: valDataset,
        verbose: TRAINING_CONFIG.TRAINING_VERBOSE, // Configurable training output
        callbacks: [
          {
            onEpochEnd: async (epoch: number, logs?: tf.Logs) => {
              if (logs) {
                // Improved best weights tracking using validation loss and F1 score
                const valF1Buy = logs["val_customF1Buy"] || 0;
                const valF1Sell = logs["val_customF1Sell"] || 0;
                const valLoss = logs["val_loss"] || Infinity;
                const combinedScore = valF1Buy + valF1Sell - valLoss * 0.1; // Balance F1 scores and loss

                if (
                  combinedScore > this.bestValF1Buy ||
                  valLoss < this.bestValLoss
                ) {
                  if (combinedScore > this.bestValF1Buy) {
                    this.bestValF1Buy = combinedScore;
                  }
                  if (valLoss < this.bestValLoss) {
                    this.bestValLoss = valLoss;
                  }

                  // Store best weights
                  this.bestWeights = this.model!.getWeights().map((w) =>
                    w.clone()
                  );
                  this.patienceCounter = 0; // Reset patience counter
                  console.log(
                    `New best weights - Combined Score: ${combinedScore.toFixed(
                      4
                    )}, Val Loss: ${valLoss.toFixed(4)} at epoch ${epoch + 1}`
                  );
                } else {
                  this.patienceCounter++;
                  if (this.patienceCounter >= TRAINING_CONFIG.PATIENCE) {
                    console.log(
                      `Early stopping triggered at epoch ${epoch + 1}`
                    );
                    this.finalMetrics.finalEpoch = epoch + 1; // Track actual final epoch
                    this.model!.stopTraining = true;
                  }
                }

                // Use centralized training logger
                await trainingLoggerCallback.onEpochEnd(epoch, logs);
                await lrCallback.onEpochEnd(epoch, logs);
                await gradientClippingCallback.onEpochEnd(epoch, logs);
                // await curriculumLearningCallback.onEpochEnd(epoch, logs); // DISABLED
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

              // Print training summary and complete history
              trainingLoggerCallback.printTrainingSummary();
              trainingLoggerCallback.printAllEpochsTable();
            },
          },
        ],
      });

      console.log("Model training completed.");

      // Capture final metrics for multi-run analysis
      const evaluationResults = await Metrics.evaluateModel(
        this.model,
        X_normalized,
        y_tensor,
        this.currentFeatureName
      );
      this.finalMetrics = {
        balancedAccuracy: evaluationResults.balancedAccuracy,
        buyF1: evaluationResults.buyF1,
        sellF1: evaluationResults.sellF1,
        combinedF1: evaluationResults.combinedF1,
        matthewsCorrelation: evaluationResults.matthewsCorrelation,
        buyPrecision: evaluationResults.buyPrecision,
        sellPrecision: evaluationResults.sellPrecision,
        buyRecall: evaluationResults.buyRecall,
        sellRecall: evaluationResults.sellRecall,
        finalEpoch: this.config.epochs, // Will be updated with actual final epoch if early stopping
      };

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
    console.log("Saving simplified model weights...");

    // Phase 1: Save selected feature indices and dynamic feature count
    const weights = {
      // Phase 1: Feature selection metadata
      selectedFeatureIndices: this.selectedFeatureIndices,
      dynamicFeatureCount: this.dynamicFeatureCount,

      // Conv1D layer
      conv1Weights: Array.from(
        await this.model.getLayer("conv1d_input").getWeights()[0].data()
      ),
      conv1Bias: Array.from(
        await this.model.getLayer("conv1d_input").getWeights()[1].data()
      ),
      bnConv1Gamma: Array.from(
        await this.model.getLayer("bn_conv1").getWeights()[0].data()
      ),
      bnConv1Beta: Array.from(
        await this.model.getLayer("bn_conv1").getWeights()[1].data()
      ),
      bnConv1MovingMean: Array.from(
        await this.model.getLayer("bn_conv1").getWeights()[2].data()
      ),
      bnConv1MovingVariance: Array.from(
        await this.model.getLayer("bn_conv1").getWeights()[3].data()
      ),

      // Output layer
      outputWeights: Array.from(
        await this.model.getLayer("output").getWeights()[0].data()
      ),
      outputBias: Array.from(
        await this.model.getLayer("output").getWeights()[1].data()
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

      // Initialize dynamic feature detection
      await this.initializeFeatureDetection();

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

  public getFinalMetrics() {
    return this.finalMetrics;
  }

  public getBalancedAccuracy(): number {
    return this.finalMetrics.balancedAccuracy;
  }

  public getBuyF1(): number {
    return this.finalMetrics.buyF1;
  }

  public getSellF1(): number {
    return this.finalMetrics.sellF1;
  }

  public getCombinedF1(): number {
    return this.finalMetrics.combinedF1;
  }

  public getMatthewsCorrelation(): number {
    return this.finalMetrics.matthewsCorrelation;
  }

  // Phase 1: Get selected feature indices
  public getSelectedFeatureIndices(): number[] {
    return this.selectedFeatureIndices;
  }

  // Phase 1: Get dynamic feature count
  public getDynamicFeatureCount(): number {
    return this.dynamicFeatureCount;
  }
}
