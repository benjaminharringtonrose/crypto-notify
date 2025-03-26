import { getHistoricalData } from "../api/getHistoricalData";
import { HistoricalData, ModelConfig } from "../types";
import FeatureCalculator from "./FeatureCalculator";

export class DataProcessor {
  private readonly config: ModelConfig;
  private readonly backtestStartDays: number;
  private readonly trainingPeriodDays: number;

  constructor(
    config: ModelConfig,
    backtestStartDays: number,
    trainingPeriodDays: number
  ) {
    this.config = config;
    this.backtestStartDays = backtestStartDays;
    this.trainingPeriodDays = trainingPeriodDays;
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
      const scale = 0.9 + Math.random() * 0.2; // Increased range
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
    return [adaFeatures, btcFeatures];
  }

  private validateFeatures(
    adaFeatures: number[],
    btcFeatures: number[]
  ): boolean {
    return (
      Array.isArray(adaFeatures) &&
      adaFeatures.length === 32 &&
      Array.isArray(btcFeatures) &&
      btcFeatures.length === 29
    );
  }

  private addNoise(features: number[], scale: number): number[] {
    return features.map((f) => f * scale + (Math.random() - 0.5) * 0.1); // Increased noise
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
      `After balancing - Buy samples: ${
        balancedY.filter((l) => l === 1).length
      }, Sell samples: ${balancedY.filter((l) => l === 0).length}`
    );
    return { X: balancedX, y: balancedY };
  }

  private computeFeatureStats(features: number[][]): {
    mean: number[];
    std: number[];
  } {
    const numFeatures = features[0].length;
    const means = Array(numFeatures).fill(0);
    const stds = Array(numFeatures).fill(0);
    features.forEach((seq) => seq.forEach((val, i) => (means[i] += val)));
    means.forEach((sum, i) => (means[i] = sum / features.length));
    features.forEach((seq) =>
      seq.forEach((val, i) => (stds[i] += (val - means[i]) ** 2))
    );
    stds.forEach((sum, i) => (stds[i] = Math.sqrt(sum / features.length) || 1));
    console.log(`Feature means: ${means.slice(0, 5)}...`);
    console.log(`Feature stds: ${stds.slice(0, 5)}...`);
    return { mean: means, std: means };
  }

  private labelData({
    prices,
    dayIndex,
    threshold = 0.05,
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

  public async prepareData(): Promise<{
    X: number[][][];
    y: number[];
    featureStats: { mean: number[]; std: number[] };
  }> {
    const { adaData, btcData } = await this.fetchHistoricalData();
    const X: number[][][] = [];
    const y: number[] = [];
    const allFeatures: number[][] = [];

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
      allFeatures.push(sequence.flat());
    }

    const balancedData = this.balanceDataset(X, y);
    console.log(`Total samples: ${balancedData.X.length}`);
    const featureStats = this.computeFeatureStats(allFeatures);
    return { X: balancedData.X, y: balancedData.y, featureStats };
  }
}
