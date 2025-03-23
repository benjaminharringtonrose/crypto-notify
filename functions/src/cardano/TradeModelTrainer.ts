import dotenv from "dotenv";
import * as tf from "@tensorflow/tfjs-node";
import * as admin from "firebase-admin";
import serviceAccount from "../../../serviceAccount.json";
import { getHistoricalData } from "../api/getHistoricalData";
import { BestWeightsCallback } from "./callbacks/BestWeightsCallback";
import { ExponentialDecayLearningRateCallback } from "./callbacks/ExponentialDecayLearningRateCallback";
import { FeatureCalculator } from "./FeatureCalculator";

dotenv.config();

interface HistoricalData {
  prices: number[];
  volumes: number[];
}

interface ModelConfig {
  timesteps: number;
  epochs: number;
  batchSize: number;
  initialLearningRate: number;
}

interface FeatureStats {
  mean: number;
  std: number;
}

export class TradeModelTrainer {
  private readonly config: ModelConfig = {
    timesteps: 14,
    epochs: 60,
    batchSize: 64,
    initialLearningRate: 0.003, // Lowered from 0.005
  };

  private readonly backtestStartDays: number = 365;
  private readonly trainingPeriodDays: number = 365;
  private model: tf.LayersModel | null = null;

  constructor() {
    this.initializeFirebase();
  }

  private initializeFirebase(): void {
    try {
      admin.initializeApp({
        credential: admin.credential.cert(
          serviceAccount as admin.ServiceAccount
        ),
        storageBucket: process.env.STORAGE_BUCKET,
      });
    } catch (error) {
      console.error("Firebase initialization failed:", error);
      throw error;
    }
  }

  private customPrecision(yTrue: tf.Tensor, yPred: tf.Tensor): tf.Scalar {
    const truePos = yTrue.mul(yPred.round()).sum();
    const predPos = yPred.round().sum();
    return truePos.div(predPos.add(tf.scalar(1e-6))) as tf.Scalar;
  }

  private customRecall(yTrue: tf.Tensor, yPred: tf.Tensor): tf.Scalar {
    const truePos = yTrue.mul(yPred.round()).sum();
    const actualPos = yTrue.sum();
    return truePos.div(actualPos.add(tf.scalar(1e-6))) as tf.Scalar;
  }

  private focalLoss(yTrue: tf.Tensor, yPred: tf.Tensor): tf.Scalar {
    const gamma = 3.0;
    const alpha = tf.tensor1d([1.2, 0.8, 1.5]); // Adjusted: Sell 1.2, Hold 0.8, Buy 1.5
    const ce = tf.losses.softmaxCrossEntropy(yTrue, yPred);
    const pt = yTrue.mul(yPred).sum(-1);
    const focalWeight = tf.pow(tf.sub(1, pt), gamma);
    const yTrueIndices = yTrue.argMax(-1);
    const alphaWeighted = tf.gather(alpha, yTrueIndices);
    const loss = ce.mul(focalWeight).mul(alphaWeighted).mean() as tf.Scalar;
    alpha.dispose();
    return loss;
  }

  private async fetchHistoricalData(): Promise<{
    adaData: HistoricalData;
    btcData: HistoricalData;
  }> {
    const startDaysAgo = this.backtestStartDays + this.trainingPeriodDays;
    console.log("Fetching historical data...");

    let adaData: HistoricalData;
    let btcData: HistoricalData;

    try {
      // Fetch ADA data
      adaData = await getHistoricalData("ADA", startDaysAgo);

      // Add a 50ms delay between ADA and BTC fetches to stay under rate limit
      await new Promise((resolve) => setTimeout(resolve, 50));

      // Fetch BTC data
      btcData = await getHistoricalData("BTC", startDaysAgo);
    } catch (error) {
      console.error("Error during historical data fetch:", error);
      throw new Error(
        `Failed to fetch historical data: ${
          error instanceof Error ? error.message : String(error)
        }`
      );
    }

    console.log(
      `ADA data length: ${adaData.prices.length}, BTC data length: ${btcData.prices.length}`
    );
    console.log(
      `Sample ADA price: ${adaData.prices[0]}, Sample BTC price: ${btcData.prices[0]}`
    );
    return { adaData, btcData };
  }

  private buildSequence(
    adaData: HistoricalData,
    btcData: HistoricalData,
    index: number
  ): number[][] | null {
    const sequence: number[][] = [];
    for (let j = index - this.config.timesteps + 1; j <= index; j++) {
      const [adaFeatures, btcFeatures] = this.computeFeaturePair(
        adaData,
        btcData,
        j
      );
      if (!this.validateFeatures(adaFeatures, btcFeatures)) return null;

      const scale = 0.95 + Math.random() * 0.1;
      const noisyFeatures = [
        ...this.addNoise(adaFeatures, scale),
        ...this.addNoise(btcFeatures, scale),
      ];
      sequence.push(noisyFeatures);
    }
    return this.adjustSequenceLength(sequence);
  }

  private computeFeaturePair(
    adaData: HistoricalData,
    btcData: HistoricalData,
    index: number
  ): [number[], number[]] {
    const adaCalculator = new FeatureCalculator({
      prices: adaData.prices,
      volumes: adaData.volumes,
      dayIndex: index,
      currentPrice: adaData.prices[index],
      isBTC: false,
    });
    const btcCalculator = new FeatureCalculator({
      prices: btcData.prices,
      volumes: btcData.volumes,
      dayIndex: index,
      currentPrice: btcData.prices[index],
      isBTC: true,
    });
    return [adaCalculator.compute(), btcCalculator.compute()];
  }

  private validateFeatures(
    adaFeatures: number[],
    btcFeatures: number[]
  ): boolean {
    const isValid =
      Array.isArray(adaFeatures) &&
      adaFeatures.length === 30 && // Updated to 30 for ADA
      Array.isArray(btcFeatures) &&
      btcFeatures.length === 29; // Updated to 29 for BTC
    if (!isValid) {
      console.warn(
        `Invalid features - ADA: ${adaFeatures?.length || "undefined"}, BTC: ${
          btcFeatures?.length || "undefined"
        }`
      );
    }
    return isValid;
  }

  private addNoise(features: number[], scale: number): number[] {
    return features.map((f) => f * scale + (Math.random() - 0.5) * 0.02);
  }

  private adjustSequenceLength(sequence: number[][]): number[][] {
    while (sequence.length < this.config.timesteps) {
      sequence.push([...sequence[sequence.length - 1]]);
    }
    while (sequence.length > this.config.timesteps) {
      sequence.pop();
    }
    return sequence;
  }

  private balanceDataset(
    X: number[][][],
    y: number[]
  ): { X: number[][][]; y: number[] } {
    const buySamples = X.filter((_, i) => y[i] === 2);
    const sellSamples = X.filter((_, i) => y[i] === 0);
    const holdSamples = X.filter((_, i) => y[i] === 1);

    const maxSamples = Math.max(
      buySamples.length,
      sellSamples.length,
      holdSamples.length
    );
    const balancedX: number[][][] = [];
    const balancedY: number[] = [];

    [sellSamples, holdSamples, buySamples].forEach((samples, label) => {
      for (let i = 0; i < maxSamples; i++) {
        const idx = i % samples.length;
        balancedX.push(samples[idx]);
        balancedY.push(label);
      }
    });

    return { X: balancedX, y: balancedY };
  }

  private computeFeatureStats(features: number[]): FeatureStats {
    const mean = features.reduce((a, b) => a + b, 0) / features.length;
    const std =
      Math.sqrt(
        features.reduce((a, b) => a + (b - mean) ** 2, 0) / features.length
      ) || 1;
    return { mean, std };
  }

  private labelData({
    prices,
    dayIndex,
    threshold = 0.02, // Reverted to 0.02 from 0.015
    horizon = 5, // Reverted to 5 from 7
  }: {
    prices: number[];
    dayIndex: number;
    threshold: number;
    horizon: number;
  }): number {
    if (dayIndex + horizon >= prices.length) return 1; // Default to hold if insufficient data
    const futureAvg =
      prices
        .slice(dayIndex + 1, dayIndex + horizon + 1)
        .reduce((a, b) => a + b, 0) / horizon;
    const priceChangePercent =
      (futureAvg - prices[dayIndex]) / prices[dayIndex];

    if (priceChangePercent > threshold) return 2; // Buy
    if (priceChangePercent < -threshold) return 0; // Sell
    return 1; // Hold
  }

  private async prepareData(): Promise<{
    X: number[][][];
    y: number[];
    featureStats: FeatureStats;
  }> {
    const { adaData, btcData } = await this.fetchHistoricalData();
    const X: number[][][] = [];
    const y: number[] = [];
    const allFeatures: number[] = [];

    for (
      let i = 34 + this.config.timesteps - 1;
      i < adaData.prices.length;
      i++
    ) {
      const sequence = this.buildSequence(adaData, btcData, i);
      if (!sequence) continue;

      const label = this.labelData({
        prices: adaData.prices,
        dayIndex: i,
        threshold: 0.02,
        horizon: 5,
      });

      X.push(sequence);
      y.push(label);
      allFeatures.push(...sequence.flat());
    }

    const balancedData = this.balanceDataset(X, y);
    return {
      X: balancedData.X,
      y: balancedData.y,
      featureStats: this.computeFeatureStats(allFeatures),
    };
  }

  private createModel(): tf.LayersModel {
    const input = tf.input({ shape: [this.config.timesteps, 59] }); // Updated from 57 to 59

    const conv1 = tf.layers
      .conv1d({
        filters: 64,
        kernelSize: 3,
        activation: "relu",
        kernelRegularizer: tf.regularizers.l2({ l2: 0.03 }),
        padding: "same",
      })
      .apply(input) as tf.SymbolicTensor;

    const bn1 = tf.layers
      .batchNormalization({ momentum: 0.9 })
      .apply(conv1) as tf.SymbolicTensor;
    const pool1 = tf.layers
      .maxPooling1d({ poolSize: 2, strides: 2 })
      .apply(bn1) as tf.SymbolicTensor;

    const conv2 = tf.layers
      .conv1d({
        filters: 32,
        kernelSize: 3,
        activation: "relu",
        kernelRegularizer: tf.regularizers.l2({ l2: 0.03 }),
        padding: "same",
      })
      .apply(pool1) as tf.SymbolicTensor;

    const bn2 = tf.layers
      .batchNormalization({ momentum: 0.9 })
      .apply(conv2) as tf.SymbolicTensor;

    const gru1 = tf.layers
      .gru({
        units: 64,
        returnSequences: true,
        kernelRegularizer: tf.regularizers.l2({ l2: 0.03 }),
      })
      .apply(bn2) as tf.SymbolicTensor;

    const bn3 = tf.layers
      .batchNormalization({ momentum: 0.9 })
      .apply(gru1) as tf.SymbolicTensor;

    const gru2 = tf.layers
      .gru({
        units: 64,
        returnSequences: false,
        kernelRegularizer: tf.regularizers.l2({ l2: 0.03 }),
      })
      .apply(bn3) as tf.SymbolicTensor;

    const bn4 = tf.layers
      .batchNormalization({ momentum: 0.9 })
      .apply(gru2) as tf.SymbolicTensor;

    const residual = tf.layers
      .dense({
        units: 64,
        kernelRegularizer: tf.regularizers.l2({ l2: 0.03 }),
      })
      .apply(tf.layers.flatten().apply(bn3)) as tf.SymbolicTensor;

    const add = tf.layers.add().apply([bn4, residual]) as tf.SymbolicTensor;
    const dropout = tf.layers
      .dropout({ rate: 0.4 })
      .apply(add) as tf.SymbolicTensor;

    const dense1 = tf.layers
      .dense({
        units: 32,
        activation: "relu",
        kernelRegularizer: tf.regularizers.l2({ l2: 0.03 }),
      })
      .apply(dropout) as tf.SymbolicTensor;

    const dense2 = tf.layers
      .dense({
        units: 16,
        activation: "relu",
        kernelRegularizer: tf.regularizers.l2({ l2: 0.03 }),
      })
      .apply(dense1) as tf.SymbolicTensor;

    const dense3 = tf.layers
      .dense({
        units: 3,
        activation: "softmax",
        kernelRegularizer: tf.regularizers.l2({ l2: 0.03 }),
      })
      .apply(dense2) as tf.SymbolicTensor;

    return tf.model({ inputs: input, outputs: dense3 });
  }

  private async trainModel(
    X: number[][][],
    y: number[]
  ): Promise<{ X_min: tf.Tensor; X_max: tf.Tensor }> {
    const X_tensor = tf.tensor3d(X);
    const y_tensor = tf.oneHot(tf.tensor1d(y, "int32"), 3);
    const X_min = X_tensor.min([0, 1]);
    const X_max = X_tensor.max([0, 1]);
    const X_normalized = X_tensor.sub(X_min).div(X_max.sub(X_min).add(1e-6));

    try {
      const totalSamples = X.length;
      const trainSize = Math.floor(totalSamples * 0.8);
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
        .prefetch(2);

      const valDataset = tf.data
        .zip({
          xs: tf.data.array(X_val.arraySync() as number[][][]),
          ys: tf.data.array(y_val.arraySync() as number[][]),
        })
        .batch(this.config.batchSize);

      this.model = this.createModel();
      const optimizer = tf.train.adam(0.0025); // Lowered from 0.003
      const bestWeightsCallback = new BestWeightsCallback({});
      const lrCallback = new ExponentialDecayLearningRateCallback(
        {},
        0.0025, // Lowered from 0.003
        0.95
      );

      this.model.compile({
        optimizer,
        loss: this.focalLoss,
        metrics: [
          tf.metrics.categoricalAccuracy,
          this.customPrecision,
          this.customRecall,
        ],
      });

      bestWeightsCallback.setModel(this.model);
      lrCallback.setModel(this.model);

      console.log("Model summary:");
      this.model.summary();

      console.log("Starting model training...");
      await this.model.fitDataset(trainDataset, {
        epochs: this.config.epochs,
        validationData: valDataset,
        callbacks: [
          tf.callbacks.earlyStopping({
            monitor: "val_categoricalAccuracy", // Changed from val_loss
            patience: 25, // Increased from 20
          }),
          bestWeightsCallback,
          lrCallback,
        ],
      });

      bestWeightsCallback.applyBestWeights(this.model);
      console.log("Model training completed.");

      await this.evaluateModel(X_normalized, y_tensor);

      return { X_min, X_max };
    } finally {
      X_tensor.dispose();
      y_tensor.dispose();
      X_normalized.dispose();
    }
  }

  private async evaluateModel(X: tf.Tensor, y: tf.Tensor): Promise<void> {
    if (!this.model) throw new Error("Model not initialized");

    const preds = this.model.predict(X) as tf.Tensor;
    const predArray = (await preds.array()) as number[][];
    const yArray = Array.from(await y.argMax(-1).data());
    const predictedLabels = predArray.map((p) => p.indexOf(Math.max(...p)));

    console.log("Sample predictions:", predArray.slice(0, 5));
    console.log(
      "Predictions vs Labels (first 5):",
      predictedLabels.slice(0, 5),
      yArray.slice(0, 5)
    );

    const metrics = this.calculateMetrics(predictedLabels, yArray);
    console.log(
      `Precision Buy: ${metrics.precisionBuy.toFixed(
        4
      )}, Precision Sell: ${metrics.precisionSell.toFixed(4)}`
    );

    const confusionMatrix = tf.math.confusionMatrix(
      tf.tensor1d(yArray, "int32"),
      tf.tensor1d(predictedLabels, "int32"),
      3
    );
    console.log("Confusion Matrix:", await confusionMatrix.array());

    preds.dispose();
  }

  private calculateMetrics(
    predictedLabels: number[],
    yArray: number[]
  ): { precisionBuy: number; precisionSell: number } {
    const truePositivesBuy = predictedLabels.reduce(
      (sum, p, i) => sum + (p === 2 && yArray[i] === 2 ? 1 : 0),
      0
    );
    const truePositivesSell = predictedLabels.reduce(
      (sum, p, i) => sum + (p === 0 && yArray[i] === 0 ? 1 : 0),
      0
    );
    const predictedBuys = predictedLabels.reduce(
      (sum, p) => sum + (p === 2 ? 1 : 0),
      0
    );
    const predictedSells = predictedLabels.reduce(
      (sum, p) => sum + (p === 0 ? 1 : 0),
      0
    );

    return {
      precisionBuy: predictedBuys > 0 ? truePositivesBuy / predictedBuys : 0,
      precisionSell:
        predictedSells > 0 ? truePositivesSell / predictedSells : 0,
    };
  }

  private async saveModelWeights(
    featureStats: FeatureStats,
    X_min: tf.Tensor,
    X_max: tf.Tensor
  ): Promise<void> {
    if (!this.model) throw new Error("Model not initialized");

    const weights = {
      conv1Weights: Array.from(
        await this.model.getLayer("conv1d_Conv1D1").getWeights()[0].data()
      ),
      conv1Bias: Array.from(
        await this.model.getLayer("conv1d_Conv1D1").getWeights()[1].data()
      ),
      conv2Weights: Array.from(
        await this.model.getLayer("conv1d_Conv1D2").getWeights()[0].data()
      ),
      conv2Bias: Array.from(
        await this.model.getLayer("conv1d_Conv1D2").getWeights()[1].data()
      ),
      gru1Weights: Array.from(
        await this.model.getLayer("gru_GRU1").getWeights()[0].data()
      ),
      gru1RecurrentWeights: Array.from(
        await this.model.getLayer("gru_GRU1").getWeights()[1].data()
      ),
      gru1Bias: Array.from(
        await this.model.getLayer("gru_GRU1").getWeights()[2].data()
      ),
      gru2Weights: Array.from(
        await this.model.getLayer("gru_GRU2").getWeights()[0].data()
      ),
      gru2RecurrentWeights: Array.from(
        await this.model.getLayer("gru_GRU2").getWeights()[1].data()
      ),
      gru2Bias: Array.from(
        await this.model.getLayer("gru_GRU2").getWeights()[2].data()
      ),
      residualWeights: Array.from(
        await this.model.getLayer("dense_Dense1").getWeights()[0].data()
      ),
      residualBias: Array.from(
        await this.model.getLayer("dense_Dense1").getWeights()[1].data()
      ),
      dense1Weights: Array.from(
        await this.model.getLayer("dense_Dense2").getWeights()[0].data()
      ),
      dense1Bias: Array.from(
        await this.model.getLayer("dense_Dense2").getWeights()[1].data()
      ),
      dense2Weights: Array.from(
        await this.model.getLayer("dense_Dense3").getWeights()[0].data()
      ),
      dense2Bias: Array.from(
        await this.model.getLayer("dense_Dense3").getWeights()[1].data()
      ),
      dense3Weights: Array.from(
        await this.model.getLayer("dense_Dense4").getWeights()[0].data()
      ),
      dense3Bias: Array.from(
        await this.model.getLayer("dense_Dense4").getWeights()[1].data()
      ),
      featureMins: Array.from(await X_min.data()),
      featureMaxs: Array.from(await X_max.data()),
      featureMean: featureStats.mean,
      featureStd: featureStats.std,
    };

    const weightsJson = { weights };
    const bucket = admin.storage().bucket();
    await bucket
      .file("tradePredictorWeights.json")
      .save(JSON.stringify(weightsJson));
    console.log("Model weights saved to Cloud Storage");

    X_min.dispose();
    X_max.dispose();
  }

  public async train(): Promise<void> {
    try {
      const { X, y, featureStats } = await this.prepareData();

      console.log(`Total samples: ${X.length}`);
      console.log(
        `Feature sequence sample (first entry): ${JSON.stringify(X[0])}`
      );
      console.log(`Label sample (first 5): ${y.slice(0, 5)}`);

      const buyCount = y.filter((label) => label === 2).length;
      const holdCount = y.filter((label) => label === 1).length;
      const sellCount = y.filter((label) => label === 0).length;
      console.log(
        `Training data: ${buyCount} buy (${(
          (buyCount / y.length) *
          100
        ).toFixed(2)}%), ` +
          `${holdCount} hold (${((holdCount / y.length) * 100).toFixed(
            2
          )}%), ` +
          `${sellCount} sell (${((sellCount / y.length) * 100).toFixed(
            2
          )}%), total: ${y.length}`
      );

      const { X_min, X_max } = await this.trainModel(X, y);
      await this.saveModelWeights(featureStats, X_min, X_max);
    } catch (error) {
      console.error("Training failed:", error);
      throw error;
    }
  }
}
