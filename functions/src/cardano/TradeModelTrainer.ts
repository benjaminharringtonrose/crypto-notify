import * as admin from "firebase-admin";
import * as tf from "@tensorflow/tfjs-node";
import dotenv from "dotenv";
import { getHistoricalData } from "../api/getHistoricalData";
import { BestWeightsCallback } from "./callbacks/BestWeightsCallback";
import { ExponentialDecayLearningRateCallback } from "./callbacks/ExponentialDecayLearningRateCallback";
import { PredictionLoggerCallback } from "./callbacks/PredictionLoggerCallback";
import FeatureCalculator from "./FeatureCalculator";
import TradeModelFactory from "./TradeModelFactory";
import serviceAccount from "../../../serviceAccount.json";
import { FeatureStats, HistoricalData, ModelConfig } from "../types";

dotenv.config();

if (!admin.apps.length) {
  admin.initializeApp({
    credential: admin.credential.cert(serviceAccount as admin.ServiceAccount),
    storageBucket:
      process.env.STORAGE_BUCKET || "crypto-notify-ee5bc.firebasestorage.app",
  });
}
const bucket = admin.storage().bucket();

export class TradeModelTrainer {
  private readonly config: ModelConfig = {
    timesteps: 14,
    epochs: 40,
    batchSize: 64,
    initialLearningRate: 0.001,
  };
  private readonly backtestStartDays: number = 365;
  private readonly trainingPeriodDays: number = 365;
  private model: tf.LayersModel | null = null;

  constructor() {
    console.log(
      "TradeModelTrainer initialized with Firebase app:",
      admin.apps[0]?.name || "none"
    );
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
    const gamma = 1.0;
    const alpha = tf.tensor1d([0.6, 1.4]);
    const ce = tf.losses.sigmoidCrossEntropy(yTrue, yPred);
    const pt = yTrue.mul(yPred).sum(-1).clipByValue(0, 1);
    const focalWeight = tf.pow(tf.sub(1, pt), gamma);
    const yTrueIndices = yTrue.argMax(-1);
    const alphaWeighted = tf.gather(alpha, yTrueIndices);
    const loss = ce.mul(focalWeight).mul(alphaWeighted).mean() as tf.Scalar;
    console.log(
      `Focal Loss: ${loss.dataSync()[0].toFixed(4)}, CE: ${ce
        .mean()
        .dataSync()[0]
        .toFixed(4)}`
    );
    alpha.dispose();
    return loss;
  }

  private async fetchHistoricalData(): Promise<{
    adaData: HistoricalData;
    btcData: HistoricalData;
  }> {
    const startDaysAgo = this.backtestStartDays + this.trainingPeriodDays;
    console.log("Fetching historical data...");
    const adaData = await getHistoricalData("ADA", startDaysAgo);
    await new Promise((resolve) => setTimeout(resolve, 50));
    const btcData = await getHistoricalData("BTC", startDaysAgo);
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
    const adaFeatures = adaCalculator.compute();
    const btcFeatures = btcCalculator.compute();
    const adaBtcRatio = adaData.prices[index] / btcData.prices[index];
    return [[...adaFeatures, adaBtcRatio], btcFeatures];
  }

  private validateFeatures(
    adaFeatures: number[],
    btcFeatures: number[]
  ): boolean {
    const isValid =
      Array.isArray(adaFeatures) &&
      adaFeatures.length === 32 &&
      Array.isArray(btcFeatures) &&
      btcFeatures.length === 29;
    if (!isValid)
      console.warn(
        `Invalid features - ADA: ${adaFeatures?.length || "undefined"}, BTC: ${
          btcFeatures?.length || "undefined"
        }`
      );
    return isValid;
  }

  private addNoise(features: number[], scale: number): number[] {
    return features.map((f) => f * scale + (Math.random() - 0.5) * 0.02);
  }

  private adjustSequenceLength(sequence: number[][]): number[][] {
    while (sequence.length < this.config.timesteps)
      sequence.push([...sequence[sequence.length - 1]]);
    while (sequence.length > this.config.timesteps) sequence.pop();
    return sequence;
  }

  private balanceDataset(
    X: number[][][],
    y: number[]
  ): { X: number[][][]; y: number[] } {
    const buySamples = X.filter((_, i) => y[i] === 1);
    const sellSamples = X.filter((_, i) => y[i] === 0);
    const maxSamples = Math.max(buySamples.length, sellSamples.length);
    const balancedX: number[][][] = [];
    const balancedY: number[] = [];
    [sellSamples, buySamples].forEach((samples, label) => {
      for (let i = 0; i < maxSamples; i++) {
        const idx = i % samples.length;
        balancedX.push(samples[idx]);
        balancedY.push(label);
      }
    });
    console.log(
      `Buy samples: ${buySamples.length}, Sell samples: ${sellSamples.length}`
    );
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
    threshold = 0.03,
    horizon = 10,
  }: {
    prices: number[];
    dayIndex: number;
    threshold?: number;
    horizon?: number;
  }): number {
    if (dayIndex + horizon >= prices.length) return 0;
    const futureAvg =
      prices
        .slice(dayIndex + 1, dayIndex + horizon + 1)
        .reduce((a, b) => a + b, 0) / horizon;
    const priceChangePercent =
      (futureAvg - prices[dayIndex]) / prices[dayIndex];
    return priceChangePercent > threshold ? 1 : 0;
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
      const label = this.labelData({ prices: adaData.prices, dayIndex: i });
      X.push(sequence);
      y.push(label);
      allFeatures.push(...sequence.flat());
    }

    const balancedData = this.balanceDataset(X, y);
    console.log(`Total samples: ${balancedData.X.length}`);
    console.log(
      `Feature sequence sample (first entry): ${JSON.stringify(
        balancedData.X[0]
      )}`
    );
    console.log(`Label sample (first 5): ${balancedData.y.slice(0, 5)}`);
    const buyCount = balancedData.y.filter((label) => label === 1).length;
    const sellCount = balancedData.y.filter((label) => label === 0).length;
    console.log(
      `Training data: ${buyCount} buy (${(
        (buyCount / balancedData.y.length) *
        100
      ).toFixed(2)}%), ${sellCount} sell (${(
        (sellCount / balancedData.y.length) *
        100
      ).toFixed(2)}%), total: ${balancedData.y.length}`
    );
    const featureStats = this.computeFeatureStats(allFeatures);
    console.log("Feature Means:", featureStats.mean);
    console.log("Feature StdDev:", featureStats.std);
    return { X: balancedData.X, y: balancedData.y, featureStats };
  }

  private async trainModel(
    X: number[][][],
    y: number[]
  ): Promise<{ X_min: tf.Tensor; X_max: tf.Tensor }> {
    const X_tensor = tf.tensor3d(X, [X.length, this.config.timesteps, 61]);
    const y_tensor = tf.tensor2d(
      y.map((label) => [label === 0 ? 1 : 0, label === 1 ? 1 : 0]),
      [y.length, 2]
    );
    const X_min = X_tensor.min([0, 1]);
    const X_max = X_tensor.max([0, 1]);
    const X_normalized = X_tensor.sub(X_min).div(X_max.sub(X_min).add(1e-6));

    try {
      const totalSamples = X.length;
      const trainSize = Math.floor(totalSamples * 0.75);
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

      const factory = new TradeModelFactory(this.config.timesteps, 61);
      this.model = factory.createModel();

      const optimizer = tf.train.adam(this.config.initialLearningRate);
      const bestWeightsCallback = new BestWeightsCallback();
      const lrCallback = new ExponentialDecayLearningRateCallback(
        this.config.initialLearningRate,
        0.96
      );
      const predictionLoggerCallback = new PredictionLoggerCallback(X_val);

      if (!this.model) throw new Error("Model initialization failed");

      this.model.compile({
        optimizer,
        loss: this.focalLoss.bind(this),
        metrics: [
          tf.metrics.binaryAccuracy,
          this.customPrecision,
          this.customRecall,
        ],
      });

      bestWeightsCallback.setModel(this.model);
      lrCallback.setModel(this.model);
      predictionLoggerCallback.setModel(this.model);

      console.log("Model summary:");
      this.model.summary();

      console.log("Starting model training...");
      await this.model.fitDataset(trainDataset, {
        epochs: this.config.epochs,
        validationData: valDataset,
        callbacks: [
          tf.callbacks.earlyStopping({ monitor: "val_loss", patience: 10 }),
          bestWeightsCallback,
          lrCallback,
          predictionLoggerCallback,
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
    const predictedLabels = predArray.map((p) => (p[1] > p[0] ? 1 : 0));

    console.log("Sample predictions:", predArray.slice(0, 5));
    console.log(
      "Predictions vs Labels (first 5):",
      predictedLabels.slice(0, 5),
      yArray.slice(0, 5)
    );
    const buyCount = predictedLabels.filter((p) => p === 1).length;
    console.log(
      `Predicted Buy: ${buyCount}, Sell: ${predictedLabels.length - buyCount}`
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
      2
    );
    console.log("Confusion Matrix:", await confusionMatrix.array());

    preds.dispose();
  }

  private calculateMetrics(
    predictedLabels: number[],
    yArray: number[]
  ): { precisionBuy: number; precisionSell: number } {
    const truePositivesBuy = predictedLabels.reduce(
      (sum, p, i) => sum + (p === 1 && yArray[i] === 1 ? 1 : 0),
      0
    );
    const truePositivesSell = predictedLabels.reduce(
      (sum, p, i) => sum + (p === 0 && yArray[i] === 0 ? 1 : 0),
      0
    );
    const predictedBuys = predictedLabels.reduce(
      (sum, p) => sum + (p === 1 ? 1 : 0),
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
        await this.model.getLayer("conv1d").getWeights()[0].data()
      ),
      conv1Bias: Array.from(
        await this.model.getLayer("conv1d").getWeights()[1].data()
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
      featureMins: Array.from(await X_min.data()),
      featureMaxs: Array.from(await X_max.data()),
      featureMean: featureStats.mean,
      featureStd: featureStats.std,
    };
    const weightsJson = JSON.stringify({ weights });
    const file = bucket.file("tradePredictorWeights.json");
    await file.save(weightsJson, { contentType: "application/json" });
    console.log(
      "Model weights saved to Firebase Storage at tradePredictorWeights.json"
    );
    X_min.dispose();
    X_max.dispose();
  }

  public async train(): Promise<void> {
    try {
      const { X, y, featureStats } = await this.prepareData();
      const { X_min, X_max } = await this.trainModel(X, y);
      await this.saveModelWeights(featureStats, X_min, X_max);
    } catch (error) {
      console.error("Training failed:", error);
      throw error;
    }
  }
}
