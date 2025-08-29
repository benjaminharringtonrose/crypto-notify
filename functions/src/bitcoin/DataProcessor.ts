import { CryptoCompareService } from "../api/CryptoCompareService";
import { HistoricalData, ModelConfig } from "../types";
import FeatureCalculator from "./FeatureCalculator";
import { FeatureDetector } from "./FeatureDetector";
import { EnhancedDataProcessor } from "./EnhancedDataProcessor";

const cryptoCompare = new CryptoCompareService();

export class DataProcessor {
  private readonly config: ModelConfig;
  private readonly startDaysAgo: number;
  private difficultyLevel: number = 1.0; // New: curriculum learning difficulty level
  private enhancedProcessor: EnhancedDataProcessor; // Phase 1: Enhanced data processing

  constructor(config: ModelConfig, startDaysAgo: number) {
    this.config = config;
    this.startDaysAgo = startDaysAgo;
    this.enhancedProcessor = new EnhancedDataProcessor(); // Initialize enhanced processor
  }

  // New: Set difficulty level for curriculum learning
  public setDifficultyLevel(level: number): void {
    this.difficultyLevel = Math.max(0.1, Math.min(1.0, level));
    console.log(
      `DataProcessor: Difficulty level set to ${(
        this.difficultyLevel * 100
      ).toFixed(1)}%`
    );
  }

  // New: Get current difficulty level
  public getDifficultyLevel(): number {
    return this.difficultyLevel;
  }

  // New: Filter data based on difficulty level
  private filterDataByDifficulty(
    X: number[][][],
    y: number[]
  ): { X: number[][][]; y: number[] } {
    if (this.difficultyLevel >= 1.0) {
      return { X, y }; // Use all data
    }

    // Calculate sample difficulty scores based on volatility and trend complexity
    const difficultyScores: { index: number; score: number }[] = [];

    for (let i = 0; i < X.length; i++) {
      const sample = X[i];
      let volatilityScore = 0;
      let trendScore = 0;

      // Calculate volatility score
      for (let j = 1; j < sample.length; j++) {
        const priceChange = Math.abs(sample[j][0] - sample[j - 1][0]); // Assuming first feature is price-like first feature
        volatilityScore += priceChange;
      }

      // Calculate trend complexity score
      let trendChanges = 0;
      for (let j = 2; j < sample.length; j++) {
        const prevTrend = sample[j - 1][0] - sample[j - 2][0];
        const currTrend = sample[j][0] - sample[j - 1][0];
        if (
          (prevTrend > 0 && currTrend < 0) ||
          (prevTrend < 0 && currTrend > 0)
        ) {
          trendChanges++;
        }
      }
      trendScore = trendChanges / (sample.length - 2);

      const totalScore = (volatilityScore + trendScore * 10) / sample.length;
      difficultyScores.push({ index: i, score: totalScore });
    }

    // Sort by difficulty score (easiest first)
    difficultyScores.sort((a, b) => a.score - b.score);

    // Select easiest samples based on difficulty level
    const numSamples = Math.max(
      50,
      Math.floor(X.length * this.difficultyLevel)
    );
    const selectedIndices = difficultyScores
      .slice(0, numSamples)
      .map((item) => item.index);

    const filteredX = selectedIndices.map((i) => X[i]);
    const filteredY = selectedIndices.map((i) => y[i]);

    console.log(
      `Curriculum Learning: Selected ${filteredX.length}/${
        X.length
      } samples (${(this.difficultyLevel * 100).toFixed(1)}% difficulty)`
    );

    return { X: filteredX, y: filteredY };
  }

  private async fetchHistoricalData(): Promise<{
    btcData: HistoricalData;
  }> {
    const btcData = await cryptoCompare.getHistoricalData(
      "BTC",
      this.startDaysAgo
    );

    console.log(`Retrieved ${btcData.prices.length} days of BTC data`);
    console.log(`Latest price: $${btcData.prices[btcData.prices.length - 1]}`);
    console.log(
      `Latest volume: ${btcData.volumes[btcData.volumes.length - 1]}`
    );

    return { btcData };
  }

  private buildSequence(
    btcData: HistoricalData,
    index: number
  ): number[][] | null {
    if (index < this.config.timesteps - 1) {
      return null;
    }

    const sequence: number[][] = [];
    const startIndex = index - this.config.timesteps + 1;

    for (let i = startIndex; i <= index; i++) {
      const btcFeatures = this.computeFeatures(btcData, i);
      if (!this.validateFeatures(btcFeatures)) {
        return null;
      }
      sequence.push(btcFeatures);
    }

    // Ensure sequence has correct length
    while (sequence.length < this.config.timesteps) {
      sequence.unshift(
        sequence[0] || Array(FeatureDetector.getFeatureCount()).fill(0)
      );
    }

    return sequence;
  }

  private computeFeatures(btcData: HistoricalData, index: number): number[] {
    const calculator = new FeatureCalculator();
    const btcFeatures = calculator.compute({
      prices: btcData.prices,
      volumes: btcData.volumes,
      dayIndex: index,
      currentPrice: btcData.prices[index],
    });

    return btcFeatures;
  }

  private validateFeatures(btcFeatures: number[]): boolean {
    if (!btcFeatures || btcFeatures.length === 0) {
      return false;
    }

    // Check for NaN or Infinity values
    for (const feature of btcFeatures) {
      if (isNaN(feature) || !isFinite(feature)) {
        return false;
      }
    }

    return true;
  }

  // Phase 1: Enhanced dataset balancing with SMOTE
  private balanceDataset(
    X: number[][][],
    y: number[]
  ): { X: number[][][]; y: number[] } {
    console.log("🔧 Phase 1: Applying enhanced dataset balancing...");

    // Use enhanced processor for SMOTE-based balancing
    return this.enhancedProcessor.enhancedBalanceDataset(X, y);
  }

  private labelData({
    prices,
    dayIndex,
    threshold = 0.001, // PROVEN OPTIMAL: 0.001 threshold prevents class imbalance
    horizon = 7, // USER REQUIREMENT: Predict 7 days ahead (buy/sell signals)
  }: {
    prices: number[];
    dayIndex: number;
    threshold?: number;
    horizon?: number;
  }): number {
    if (dayIndex + horizon >= prices.length) return 0;

    const currentPrice = prices[dayIndex];
    const nextPrice = prices[dayIndex + horizon];

    const priceChangePercent = (nextPrice - currentPrice) / currentPrice;

    // SIMPLIFIED: Use simple threshold without complex adjustments
    const label = priceChangePercent > threshold ? 1 : 0;

    // Debug: Enhanced logging to track labeling distribution
    if (dayIndex % 50 === 0) {
      console.log(
        `Labeling Day ${dayIndex}: Current=${currentPrice.toFixed(
          2
        )}, Next=${nextPrice.toFixed(2)}, Change=${(
          priceChangePercent * 100
        ).toFixed(3)}%, Threshold=${(threshold * 100).toFixed(3)}%, Label=${
          label === 1 ? "BUY" : "SELL"
        }`
      );
    }

    return label;
  }

  // Phase 1: Enhanced feature statistics with robust normalization
  public computeFeatureStats(features: number[][]): {
    mean: number[];
    std: number[];
    selectedFeatures: number[];
    normalizationStats: any;
  } {
    console.log("🔧 Phase 1: Computing enhanced feature statistics...");

    // Apply robust normalization and feature selection
    const { stats, selectedFeatures } =
      this.enhancedProcessor.robustNormalize(features);

    // Convert back to mean/std format for compatibility with existing code
    const means = stats.median; // Use median instead of mean for robustness
    const stds = stats.mad.map((mad: number) => {
      const std = mad * 1.4826; // Convert MAD to approximate std
      return std === 0 ? 1.0 : std; // Use 1.0 as fallback for zero MAD
    });

    console.log(
      `Enhanced feature stats - Selected features: ${selectedFeatures.length}, Original: ${features[0].length}`
    );
    console.log(
      `Feature medians: ${means
        .slice(0, 5)
        .map((m: number) => m.toFixed(4))}...`
    );
    console.log(
      `Feature MADs: ${stds.slice(0, 5).map((s: number) => s.toFixed(4))}...`
    );

    return {
      mean: means,
      std: stds,
      selectedFeatures,
      normalizationStats: stats,
    };
  }

  public async prepareData(): Promise<{ X: number[][][]; y: number[] }> {
    console.log("🚀 Phase 1: Starting enhanced data preparation...");

    const { btcData } = await this.fetchHistoricalData();
    const X: number[][][] = [];
    const y: number[] = [];

    for (
      let i = 34 + this.config.timesteps - 1;
      i < btcData.prices.length - 7; // ADJUSTED: Reserve 7 days for prediction horizon
      i++
    ) {
      const sequence = this.buildSequence(btcData, i);
      if (!sequence) continue;
      const label = this.labelData({
        prices: btcData.prices,
        dayIndex: i,
        threshold: 0.001,
        horizon: 7, // USER REQUIREMENT: 7-day prediction horizon
      });
      X.push(sequence);
      y.push(label);
    }

    console.log(`📊 Raw data - Total samples: ${X.length}`);
    console.log(
      `📊 Raw data - Buy samples: ${
        y.filter((l) => l === 1).length
      }, Sell samples: ${y.filter((l) => l === 0).length}`
    );

    // Phase 1: Apply enhanced balancing with SMOTE
    const balancedData = this.balanceDataset(X, y);

    // Apply curriculum learning filter based on difficulty level
    const filteredData = this.filterDataByDifficulty(
      balancedData.X,
      balancedData.y
    );

    console.log(
      `✅ Phase 1 Complete - Final dataset: ${filteredData.X.length} samples`
    );
    console.log(
      `📊 Final distribution - Buy: ${
        filteredData.y.filter((l) => l === 1).length
      }, Sell: ${filteredData.y.filter((l) => l === 0).length}`
    );

    return { X: filteredData.X, y: filteredData.y };
  }

  // Phase 1: Get selected feature indices for model creation
  public getSelectedFeatureIndices(): number[] {
    return this.enhancedProcessor.getSelectedFeatureIndices();
  }

  // Phase 1: Get normalization statistics
  public getNormalizationStats(): any {
    return this.enhancedProcessor.getNormalizationStats();
  }
}
