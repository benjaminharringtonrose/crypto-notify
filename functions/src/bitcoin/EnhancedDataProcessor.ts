import { FeatureRegistry } from "./FeatureRegistry";

/**
 * Enhanced Data Processor for Phase 1 Improvements
 * Implements robust normalization, feature selection, and SMOTE for time series
 */
export class EnhancedDataProcessor {
  private correlationThreshold: number = 0.85;
  private selectedFeatureIndices: number[] = [];
  private normalizationStats: {
    median: number[];
    mad: number[];
    iqr: number[];
    winsorized: number[][];
  } | null = null;

  constructor() {
    console.log(
      "🔧 EnhancedDataProcessor initialized for Phase 1 improvements"
    );
  }

  /**
   * Phase 1.1: Robust normalization with outlier handling
   */
  public robustNormalize(features: number[][]): {
    normalized: number[][];
    stats: {
      median: number[];
      mad: number[];
      iqr: number[];
      winsorized: number[][];
    };
    selectedFeatures: number[];
  } {
    console.log("🔧 Phase 1.1: Applying robust normalization...");

    if (features.length === 0 || features[0].length === 0) {
      console.log("⚠️ No features to normalize");
      return {
        normalized: [],
        stats: { median: [], mad: [], iqr: [], winsorized: [] },
        selectedFeatures: [],
      };
    }

    const numFeatures = features[0].length;
    const numSamples = features.length;

    // Compute robust statistics
    const stats = this.computeRobustStatistics(features);

    // Check for invalid statistics and filter them out
    const validFeatureIndices: number[] = [];
    for (let i = 0; i < numFeatures; i++) {
      const median = stats.median[i];
      const mad = stats.mad[i];

      // Check if statistics are valid
      if (isNaN(median) || isNaN(mad) || !isFinite(median) || !isFinite(mad)) {
        console.log(
          `⚠️ Feature ${i} has invalid statistics - median: ${median}, mad: ${mad}`
        );
        continue;
      }

      validFeatureIndices.push(i);
    }

    console.log(
      `✅ Valid features: ${validFeatureIndices.length} out of ${numFeatures}`
    );

    // Apply correlation-based feature selection only on valid features
    const selectedFeatures = this.correlationBasedFeatureSelection(
      features.map((row) => validFeatureIndices.map((i) => row[i]))
    );

    // Map back to original feature indices
    const finalSelectedFeatures = selectedFeatures.map(
      (i) => validFeatureIndices[i]
    );

    // Apply robust normalization to selected features
    const normalized: number[][] = [];
    for (let sampleIdx = 0; sampleIdx < numSamples; sampleIdx++) {
      const normalizedSample: number[] = [];
      for (const featureIdx of finalSelectedFeatures) {
        const normalizedValue = this.robustNormalizeValue(
          features[sampleIdx][featureIdx],
          stats,
          featureIdx
        );
        normalizedSample.push(normalizedValue);
      }
      normalized.push(normalizedSample);
    }

    // Store selected feature indices for later use
    this.selectedFeatureIndices = finalSelectedFeatures;
    this.normalizationStats = stats;

    console.log(
      `✅ Robust normalization applied to ${finalSelectedFeatures.length} features (reduced from ${numFeatures})`
    );

    return {
      normalized,
      stats,
      selectedFeatures: finalSelectedFeatures,
    };
  }

  /**
   * Phase 1.2: Correlation-based feature selection
   */
  private correlationBasedFeatureSelection(features: number[][]): number[] {
    console.log(
      "🔧 Phase 1.2: Performing correlation-based feature selection..."
    );

    const numFeatures = features[0].length;
    const correlationMatrix = this.computeCorrelationMatrix(features);

    // Track which features to keep
    const keepFeatures: boolean[] = Array(numFeatures).fill(true);
    const removedFeatures: number[] = [];

    // Remove highly correlated features
    for (let i = 0; i < numFeatures; i++) {
      if (!keepFeatures[i]) continue;

      for (let j = i + 1; j < numFeatures; j++) {
        if (!keepFeatures[j]) continue;

        const correlation = Math.abs(correlationMatrix[i][j]);
        if (correlation > this.correlationThreshold) {
          // Keep the feature with higher importance
          const importanceI = this.getFeatureImportance(i);
          const importanceJ = this.getFeatureImportance(j);

          if (importanceI >= importanceJ) {
            keepFeatures[j] = false;
            removedFeatures.push(j);
          } else {
            keepFeatures[i] = false;
            removedFeatures.push(i);
            break; // Move to next i
          }
        }
      }
    }

    // Get indices of features to keep
    const selectedIndices: number[] = [];
    for (let i = 0; i < numFeatures; i++) {
      if (keepFeatures[i]) {
        selectedIndices.push(i);
      }
    }

    // Ensure we don't remove too many features - keep at least 30 features
    const minFeatures = Math.min(30, numFeatures);
    if (selectedIndices.length < minFeatures) {
      console.log(
        `⚠️ Too many features removed (${selectedIndices.length}), keeping at least ${minFeatures}`
      );

      // Add back some removed features based on importance
      const removedByImportance = removedFeatures
        .map((i) => ({ index: i, importance: this.getFeatureImportance(i) }))
        .sort((a, b) => b.importance - a.importance);

      const needed = minFeatures - selectedIndices.length;
      for (let i = 0; i < needed && i < removedByImportance.length; i++) {
        selectedIndices.push(removedByImportance[i].index);
      }

      // Sort the final selection
      selectedIndices.sort((a, b) => a - b);
    }

    console.log(
      `✅ Selected ${selectedIndices.length} features from ${numFeatures} total features`
    );
    console.log(
      `📊 Feature reduction: ${(
        ((numFeatures - selectedIndices.length) / numFeatures) *
        100
      ).toFixed(1)}%`
    );

    return selectedIndices;
  }

  /**
   * Compute correlation matrix for feature selection
   */
  private computeCorrelationMatrix(features: number[][]): number[][] {
    const numFeatures = features[0].length;
    const correlationMatrix: number[][] = Array(numFeatures)
      .fill(0)
      .map(() => Array(numFeatures).fill(0));

    for (let i = 0; i < numFeatures; i++) {
      for (let j = i; j < numFeatures; j++) {
        if (i === j) {
          correlationMatrix[i][j] = 1.0;
        } else {
          const correlation = this.calculateCorrelation(
            features.map((row) => row[i]),
            features.map((row) => row[j])
          );
          correlationMatrix[i][j] = correlation;
          correlationMatrix[j][i] = correlation;
        }
      }
    }

    return correlationMatrix;
  }

  /**
   * Calculate Pearson correlation between two arrays
   */
  private calculateCorrelation(x: number[], y: number[]): number {
    const n = x.length;
    const sumX = x.reduce((a, b) => a + b, 0);
    const sumY = y.reduce((a, b) => a + b, 0);
    const sumXY = x.reduce((sum, xi, i) => sum + xi * y[i], 0);
    const sumX2 = x.reduce((sum, xi) => sum + xi * xi, 0);
    const sumY2 = y.reduce((sum, yi) => sum + yi * yi, 0);

    const numerator = n * sumXY - sumX * sumY;
    const denominator = Math.sqrt(
      (n * sumX2 - sumX * sumX) * (n * sumY2 - sumY * sumY)
    );

    return denominator === 0 ? 0 : numerator / denominator;
  }

  /**
   * Get feature importance based on feature registry
   */
  private getFeatureImportance(featureIndex: number): number {
    try {
      const feature = FeatureRegistry.getFeatureByIndex(featureIndex);
      if (!feature) return 0;

      const importanceOrder = { high: 3, medium: 2, low: 1, unknown: 0 };
      return importanceOrder[feature.importance || "unknown"];
    } catch (error) {
      return 0; // Default importance for unknown features
    }
  }

  /**
   * Compute robust statistics for normalization
   */
  private computeRobustStatistics(features: number[][]): {
    median: number[];
    mad: number[];
    iqr: number[];
    winsorized: number[][];
  } {
    const numFeatures = features[0].length;
    const stats = {
      median: Array(numFeatures).fill(0),
      mad: Array(numFeatures).fill(0),
      iqr: Array(numFeatures).fill(0),
      winsorized: Array(numFeatures)
        .fill(0)
        .map(() => [] as number[]),
    };

    for (let i = 0; i < numFeatures; i++) {
      const column = features.map((row) => row[i]);

      // Compute median
      const sorted = [...column].sort((a, b) => a - b);
      stats.median[i] = this.median(sorted);

      // Compute Median Absolute Deviation (MAD)
      const deviations = column.map((val) => Math.abs(val - stats.median[i]));
      stats.mad[i] = this.median(deviations.sort((a, b) => a - b));

      // Compute Interquartile Range (IQR)
      const q1 = this.percentile(sorted, 25);
      const q3 = this.percentile(sorted, 75);
      stats.iqr[i] = q3 - q1;

      // Winsorize outliers (cap at 1st/99th percentiles)
      const p1 = this.percentile(sorted, 1);
      const p99 = this.percentile(sorted, 99);
      stats.winsorized[i] = column.map((val) =>
        Math.max(p1, Math.min(p99, val))
      );
    }

    return stats;
  }

  /**
   * Apply robust normalization to a single value
   */
  private robustNormalizeValue(
    value: number,
    stats: any,
    featureIndex: number
  ): number {
    const median = stats.median[featureIndex];
    const mad = stats.mad[featureIndex];

    // Use MAD-based normalization (more robust than mean/std)
    if (mad === 0) {
      return 0; // Avoid division by zero
    }

    const std = mad * 1.4826; // Convert MAD to approximate std
    if (std === 0) {
      return 0; // Additional check for zero standard deviation
    }

    return (value - median) / std;
  }

  /**
   * Phase 1.3: SMOTE for Time Series Data
   */
  public smoteTimeSeries(
    minoritySamples: number[][][],
    k: number = 5,
    targetRatio: number = 1.0
  ): number[][][] {
    console.log("🔧 Phase 1.3: Applying SMOTE for time series data...");

    if (minoritySamples.length === 0) {
      console.log("⚠️ No minority samples to augment");
      return [];
    }

    const synthetic: number[][][] = [];

    // Calculate how many synthetic samples we need to achieve target ratio
    const samplesPerMinority = Math.max(1, Math.floor(targetRatio));

    console.log(
      `📊 SMOTE target: ${samplesPerMinority} samples per minority sample`
    );

    for (const sample of minoritySamples) {
      // Find k nearest neighbors using DTW distance
      const neighbors = this.findKNearestNeighbors(sample, minoritySamples, k);

      // Generate synthetic samples per minority sample
      for (let i = 0; i < samplesPerMinority; i++) {
        if (neighbors.length > 0) {
          const neighbor =
            neighbors[Math.floor(Math.random() * neighbors.length)];

          // Use more sophisticated interpolation with noise
          const syntheticSample = this.interpolateSequencesWithNoise(
            sample,
            neighbor
          );
          synthetic.push(syntheticSample);
        }
      }
    }

    console.log(
      `✅ Generated ${synthetic.length} synthetic samples from ${minoritySamples.length} minority samples`
    );
    return synthetic;
  }

  /**
   * Enhanced interpolation with noise for more realistic synthetic samples
   */
  private interpolateSequencesWithNoise(
    seq1: number[][],
    seq2: number[][]
  ): number[][] {
    const timesteps = seq1.length;
    const features = seq1[0].length;
    const synthetic: number[][] = [];

    for (let t = 0; t < timesteps; t++) {
      const syntheticTimestep: number[] = [];

      for (let f = 0; f < features; f++) {
        // Linear interpolation with random weight
        const weight = Math.random();
        const interpolatedValue =
          weight * seq1[t][f] + (1 - weight) * seq2[t][f];

        // Add small amount of noise to make it more realistic
        const noise =
          (Math.random() - 0.5) * 0.01 * Math.abs(interpolatedValue);
        const finalValue = interpolatedValue + noise;

        syntheticTimestep.push(finalValue);
      }

      synthetic.push(syntheticTimestep);
    }

    return synthetic;
  }

  /**
   * Find k nearest neighbors using Dynamic Time Warping (DTW)
   */
  private findKNearestNeighbors(
    target: number[][],
    samples: number[][][],
    k: number
  ): number[][][] {
    const distances: { sample: number[][]; distance: number }[] = [];

    for (const sample of samples) {
      if (sample === target) continue; // Skip self

      const distance = this.dtwDistance(target, sample);
      distances.push({ sample, distance });
    }

    // Sort by distance and return k nearest
    distances.sort((a, b) => a.distance - b.distance);
    return distances.slice(0, k).map((item) => item.sample);
  }

  /**
   * Calculate DTW distance between two time series
   */
  private dtwDistance(series1: number[][], series2: number[][]): number {
    const n = series1.length;
    const m = series2.length;

    // Initialize DTW matrix
    const dtw: number[][] = Array(n + 1)
      .fill(0)
      .map(() => Array(m + 1).fill(Infinity));
    dtw[0][0] = 0;

    // Fill DTW matrix
    for (let i = 1; i <= n; i++) {
      for (let j = 1; j <= m; j++) {
        const cost = this.euclideanDistance(series1[i - 1], series2[j - 1]);
        dtw[i][j] =
          cost + Math.min(dtw[i - 1][j], dtw[i][j - 1], dtw[i - 1][j - 1]);
      }
    }

    return dtw[n][m];
  }

  /**
   * Calculate Euclidean distance between two feature vectors
   */
  private euclideanDistance(vec1: number[], vec2: number[]): number {
    return Math.sqrt(
      vec1.reduce((sum, val, i) => sum + Math.pow(val - vec2[i], 2), 0)
    );
  }

  /**
   * Apply enhanced balancing with SMOTE
   */
  public enhancedBalanceDataset(
    X: number[][][],
    y: number[]
  ): { X: number[][][]; y: number[] } {
    console.log(
      "🔧 Phase 1.4: Applying enhanced dataset balancing with SMOTE..."
    );

    // Separate buy and sell samples
    const buySamples: number[][][] = [];
    const sellSamples: number[][][] = [];

    y.forEach((label, index) => {
      if (label === 1) {
        buySamples.push(X[index]);
      } else {
        sellSamples.push(X[index]);
      }
    });

    console.log(
      `📊 Original distribution - Buy: ${buySamples.length}, Sell: ${sellSamples.length}`
    );

    // Determine minority class (the class with fewer samples)
    const buyCount = buySamples.length;
    const sellCount = sellSamples.length;
    const isBuyMinority = buyCount < sellCount;

    console.log(
      `🔍 Minority class analysis: Buy=${buyCount}, Sell=${sellCount}, Buy is ${
        isBuyMinority ? "minority" : "majority"
      }`
    );

    const minoritySamples = isBuyMinority ? buySamples : sellSamples;
    const majoritySamples = isBuyMinority ? sellSamples : buySamples;
    const minorityLabel = isBuyMinority ? 1 : 0; // 1 for Buy, 0 for Sell
    const majorityLabel = isBuyMinority ? 0 : 1; // 0 for Sell, 1 for Buy

    // Calculate target ratio to achieve better balance (aim for 0.8:1 ratio instead of 1:1 to avoid over-sampling)
    const targetRatio = Math.min(
      2.0,
      Math.ceil((majoritySamples.length / minoritySamples.length) * 0.8)
    );
    console.log(
      `📊 Target SMOTE ratio: ${targetRatio.toFixed(
        2
      )} samples per minority sample`
    );

    // Apply SMOTE to minority class
    const syntheticSamples = this.smoteTimeSeries(
      minoritySamples,
      5,
      targetRatio
    );

    // Combine original and synthetic samples
    const balancedX: number[][][] = [];
    const balancedY: number[] = [];

    // Add majority samples
    majoritySamples.forEach((sample) => {
      balancedX.push(sample);
      balancedY.push(majorityLabel);
    });

    // Add original minority samples
    minoritySamples.forEach((sample) => {
      balancedX.push(sample);
      balancedY.push(minorityLabel);
    });

    // Add synthetic minority samples
    syntheticSamples.forEach((sample) => {
      balancedX.push(sample);
      balancedY.push(minorityLabel);
    });

    // Shuffle the balanced dataset
    const indices = Array.from({ length: balancedX.length }, (_, i) => i);
    for (let i = indices.length - 1; i > 0; i--) {
      const j = Math.floor(Math.random() * (i + 1));
      [indices[i], indices[j]] = [indices[j], indices[i]];
    }

    const shuffledX = indices.map((i) => balancedX[i]);
    const shuffledY = indices.map((i) => balancedY[i]);

    const finalBuyCount = shuffledY.filter((l) => l === 1).length;
    const finalSellCount = shuffledY.filter((l) => l === 0).length;

    console.log(
      `✅ Balanced distribution - Buy: ${finalBuyCount}, Sell: ${finalSellCount}`
    );
    console.log(
      `📊 Balance ratio: ${(finalBuyCount / finalSellCount).toFixed(3)}`
    );

    return { X: shuffledX, y: shuffledY };
  }

  /**
   * Get selected feature indices for model creation
   */
  public getSelectedFeatureIndices(): number[] {
    return this.selectedFeatureIndices;
  }

  /**
   * Get normalization statistics
   */
  public getNormalizationStats(): any {
    return this.normalizationStats;
  }

  /**
   * Utility functions
   */
  private median(sortedArray: number[]): number {
    const mid = Math.floor(sortedArray.length / 2);
    return sortedArray.length % 2 === 0
      ? (sortedArray[mid - 1] + sortedArray[mid]) / 2
      : sortedArray[mid];
  }

  private percentile(sortedArray: number[], p: number): number {
    const index = Math.floor((p / 100) * (sortedArray.length - 1));
    return sortedArray[index];
  }
}
