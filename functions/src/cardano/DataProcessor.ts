import { CryptoCompareService } from "../api/CryptoCompareService";
import { MODEL_CONFIG, PERIODS } from "../constants";
import { HistoricalData, ModelConfig } from "../types";
import FeatureCalculator from "./FeatureCalculator";

const cryptoCompare = new CryptoCompareService();

export class DataProcessor {
  private readonly config: ModelConfig;
  private readonly startDaysAgo: number;

  constructor(config: ModelConfig, startDaysAgo: number) {
    this.config = config;
    this.startDaysAgo = startDaysAgo;
  }

  private async fetchHistoricalData(): Promise<{
    adaData: HistoricalData;
    btcData: HistoricalData;
  }> {
    console.log("Fetching historical data...");
    const [adaData, btcData] = await Promise.all([
      cryptoCompare.getHistoricalData("ADA", this.startDaysAgo),
      cryptoCompare.getHistoricalData("BTC", this.startDaysAgo),
    ]);
    console.log(
      `ADA data length: ${adaData.prices.length}, BTC data length: ${btcData.prices.length}`
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
      if (!this.validateFeatures(adaFeatures, btcFeatures)) {
        console.log(
          `Invalid features at index ${j}: adaFeatures length=${adaFeatures.length}, btcFeatures length=${btcFeatures.length}`
        );
        return null;
      }
      const atr = new FeatureCalculator().calculateATR(
        adaData.prices.slice(0, j + 1),
        PERIODS.ATR
      );
      const scale = 1 + atr * (0.9 + Math.random() * 0.2); // Volatility-based noise
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
    const featureCalculator = new FeatureCalculator();
    const adaFeatures = featureCalculator.compute({
      prices: adaData.prices,
      volumes: adaData.volumes,
      dayIndex: index,
      currentPrice: adaData.prices[index],
      isBTC: false,
      btcPrice: btcData.prices[index],
    });
    const btcFeatures = featureCalculator.compute({
      prices: btcData.prices,
      volumes: btcData.volumes,
      dayIndex: index,
      currentPrice: btcData.prices[index],
      isBTC: true,
    });
    return [adaFeatures, btcFeatures];
  }

  private validateFeatures(
    adaFeatures: number[],
    btcFeatures: number[]
  ): boolean {
    return (
      Array.isArray(adaFeatures) &&
      adaFeatures.length === MODEL_CONFIG.ADA_FEATURE_COUNT &&
      Array.isArray(btcFeatures) &&
      btcFeatures.length === MODEL_CONFIG.BTC_FEATURE_COUNT
    );
  }

  private addNoise(features: number[], scale: number): number[] {
    return features.map((f) => f * scale + (Math.random() - 0.5) * 0.05); // Reduced noise amplitude
  }

  private adjustSequenceLength(sequence: number[][]): number[][] {
    while (sequence.length < this.config.timesteps)
      sequence.unshift(sequence[0]);
    while (sequence.length > this.config.timesteps) sequence.pop();
    return sequence;
  }

  private balanceDataset(
    X: number[][][],
    y: number[]
  ): { X: number[][][]; y: number[] } {
    const buySamples = X.filter((_, i) => y[i] === 1);
    const sellSamples = X.filter((_, i) => y[i] === 0);
    console.log(
      `Before balancing - Buy samples: ${buySamples.length}, Sell samples: ${sellSamples.length}`
    );

    // Use SMOTE-like oversampling for minority class
    const minorityClass =
      buySamples.length < sellSamples.length ? buySamples : sellSamples;
    const majorityClass =
      buySamples.length < sellSamples.length ? sellSamples : buySamples;
    const minorityLabel = buySamples.length < sellSamples.length ? 1 : 0;

    const balancedX: number[][][] = [];
    const balancedY: number[] = [];

    // Add all majority class samples
    majorityClass.forEach((sample, i) => {
      balancedX.push(sample);
      balancedY.push(buySamples.length < sellSamples.length ? 0 : 1);
    });

    // Add minority class samples with augmentation
    minorityClass.forEach((sample, i) => {
      balancedX.push(sample);
      balancedY.push(minorityLabel);

      // Add augmented samples for minority class
      const augmentedSamples = this.augmentSample(sample, 2); // Create 2 augmented samples
      augmentedSamples.forEach((augSample) => {
        balancedX.push(augSample);
        balancedY.push(minorityLabel);
      });
    });

    // Shuffle the balanced dataset
    const indices = Array.from({ length: balancedX.length }, (_, i) => i);
    for (let i = indices.length - 1; i > 0; i--) {
      const j = Math.floor(Math.random() * (i + 1));
      [indices[i], indices[j]] = [indices[j], indices[i]];
    }

    const shuffledX = indices.map((i) => balancedX[i]);
    const shuffledY = indices.map((i) => balancedY[i]);

    console.log(
      `After balancing - Buy samples: ${
        shuffledY.filter((l) => l === 1).length
      }, Sell samples: ${shuffledY.filter((l) => l === 0).length}`
    );
    return { X: shuffledX, y: shuffledY };
  }

  private augmentSample(
    sample: number[][],
    numAugmentations: number
  ): number[][][] {
    const augmented: number[][][] = [];

    for (let i = 0; i < numAugmentations; i++) {
      const augmentedSample = sample.map((timestep) =>
        timestep.map((feature) => {
          // Add small random noise and scaling
          const noise = (Math.random() - 0.5) * 0.02; // Reduced noise
          const scale = 1 + (Math.random() - 0.5) * 0.1; // Small scaling variation
          return feature * scale + noise;
        })
      );
      augmented.push(augmentedSample);
    }

    return augmented;
  }

  private labelData({
    prices,
    dayIndex,
    threshold = 0.07, // Increased from 0.05
    horizon = 7,
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

  public computeFeatureStats(features: number[][]): {
    mean: number[];
    std: number[];
  } {
    const numFeatures = features[0].length;
    const means = Array(numFeatures).fill(0);
    const stds = Array(numFeatures).fill(0);

    // Compute means
    features.forEach((seq) => seq.forEach((val, i) => (means[i] += val)));
    means.forEach((sum, i) => (means[i] = sum / features.length));

    // Compute robust standard deviations
    features.forEach((seq) =>
      seq.forEach((val, i) => (stds[i] += (val - means[i]) ** 2))
    );
    stds.forEach((sum, i) => {
      const variance = sum / features.length;
      // Use robust std calculation with minimum threshold
      stds[i] = Math.sqrt(variance) || 0.01; // Minimum std to prevent division by zero
    });

    // Clip extreme values to prevent outliers from dominating
    const maxStd = 10; // Maximum allowed standard deviation
    stds.forEach((std, i) => {
      if (std > maxStd) {
        stds[i] = maxStd;
      }
    });

    console.log(
      `Feature means: ${means.slice(0, 5).map((m) => m.toFixed(4))}...`
    );
    console.log(
      `Feature stds: ${stds.slice(0, 5).map((s) => s.toFixed(4))}...`
    );
    return { mean: means, std: stds };
  }

  public async prepareData(): Promise<{ X: number[][][]; y: number[] }> {
    const { adaData, btcData } = await this.fetchHistoricalData();
    const X: number[][][] = [];
    const y: number[] = [];

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
    }

    const balancedData = this.balanceDataset(X, y);
    console.log(`Total samples: ${balancedData.X.length}`);
    return { X: balancedData.X, y: balancedData.y };
  }
}
