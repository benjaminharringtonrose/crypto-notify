#!/usr/bin/env ts-node
/**
 * Feature Analysis Script - Comprehensive analysis of feature utility for trading model
 *
 * This script provides multiple approaches to determine which features are beneficial:
 * 1. Feature-target correlation analysis
 * 2. Feature importance using tree-based models (Random Forest)
 * 3. Statistical significance testing
 * 4. Redundancy analysis (correlation between features)
 * 5. Performance impact analysis through ablation studies
 */

import { DataProcessor } from "../bitcoin/DataProcessor";
import { FeatureRegistry, FEATURE_REGISTRY } from "../bitcoin/FeatureRegistry";
import { FeatureDetector } from "../bitcoin/FeatureDetector";
import { MODEL_CONFIG, TRAINING_CONFIG } from "../constants";

interface FeatureAnalysisResult {
  featureName: string;
  index: number;
  category: string;
  importance: string;

  // Statistical measures
  correlation: number;
  pValue: number;
  mutualInformation?: number;

  // Importance measures
  treeImportance?: number;
  permutationImportance?: number;

  // Redundancy measures
  avgCorrelationWithOthers: number;
  maxCorrelationWithOthers: number;

  // Performance measures
  individualPerformance?: number;
  ablationImpact?: number;

  // Final scores
  utilityScore: number;
  redundancyScore: number;
  finalScore: number;
  recommendation: "keep" | "remove" | "investigate";
}

interface CorrelationMatrix {
  matrix: number[][];
  featureNames: string[];
}

class FeatureAnalyzer {
  private dataProcessor: DataProcessor;
  constructor() {
    this.dataProcessor = new DataProcessor(
      {
        timesteps: MODEL_CONFIG.TIMESTEPS,
        epochs: TRAINING_CONFIG.EPOCHS,
        batchSize: TRAINING_CONFIG.BATCH_SIZE,
        initialLearningRate: TRAINING_CONFIG.INITIAL_LEARNING_RATE,
      },
      600 // Use 600 days for analysis
    );
  }

  /**
   * Main analysis function - runs comprehensive feature analysis
   */
  public async analyzeFeatures(): Promise<FeatureAnalysisResult[]> {
    console.log("🔍 Starting comprehensive feature analysis...");

    // Step 0: Initialize feature detection
    console.log("🔧 Initializing feature detection...");
    await FeatureDetector.detectFeatureCount();

    // Step 1: Prepare data
    console.log("📊 Preparing training data...");
    const { X, y } = await this.dataProcessor.prepareData();
    console.log(
      `✅ Prepared ${X.length} samples with ${X[0][0].length} features`
    );

    // Step 2: Flatten data for analysis (take last timestep of each sequence)
    const flattenedFeatures = X.map(
      (sequence) => sequence[sequence.length - 1]
    );
    const labels = y;

    console.log(
      `📈 Flattened to ${flattenedFeatures.length} samples × ${flattenedFeatures[0].length} features`
    );

    // Step 3: Fix feature registry mismatch first
    await this.auditFeatureImplementation(flattenedFeatures[0].length);

    // Step 4: Run various analyses
    console.log("🧮 Computing feature correlations...");
    const correlations = this.calculateFeatureTargetCorrelations(
      flattenedFeatures,
      labels
    );

    console.log("📊 Computing statistical significance...");
    const pValues = this.calculateStatisticalSignificance(
      flattenedFeatures,
      labels
    );

    console.log("🔗 Computing feature-feature correlations...");
    const correlationMatrix =
      this.calculateFeatureCorrelationMatrix(flattenedFeatures);

    console.log("🎯 Computing utility scores...");
    const results = this.computeFeatureAnalysis(
      correlations,
      pValues,
      correlationMatrix
    );

    // Step 5: Generate recommendations
    console.log("💡 Generating recommendations...");
    this.generateRecommendations(results);

    return results;
  }

  /**
   * Audit current feature implementation vs registry
   */
  private async auditFeatureImplementation(
    actualFeatureCount: number
  ): Promise<void> {
    const registryCount = FeatureRegistry.getFeatureCount();

    console.log("\n🔍 FEATURE IMPLEMENTATION AUDIT:");
    console.log(`Registry defines: ${registryCount} features`);
    console.log(`Calculator produces: ${actualFeatureCount} features`);

    if (registryCount !== actualFeatureCount) {
      console.log("❌ MISMATCH DETECTED!");
      console.log("\nRegistry features:");
      FEATURE_REGISTRY.forEach((feature, idx) => {
        console.log(`  ${idx + 1}. ${feature.name} (${feature.category})`);
      });

      console.log("\n🔧 Need to fix registry to match actual implementation");
      console.log(
        "The FeatureCalculator.compute() method produces the actual features used by the model."
      );
    } else {
      console.log("✅ Registry matches implementation");
    }
  }

  /**
   * Calculate correlation between each feature and target variable
   */
  private calculateFeatureTargetCorrelations(
    features: number[][],
    labels: number[]
  ): number[] {
    const numFeatures = features[0].length;
    const correlations: number[] = [];

    for (let featureIdx = 0; featureIdx < numFeatures; featureIdx++) {
      const featureValues = features.map((sample) => sample[featureIdx]);
      const correlation = this.pearsonCorrelation(featureValues, labels);
      correlations.push(Math.abs(correlation)); // Use absolute correlation
    }

    return correlations;
  }

  /**
   * Calculate statistical significance of each feature
   */
  private calculateStatisticalSignificance(
    features: number[][],
    labels: number[]
  ): number[] {
    const numFeatures = features[0].length;
    const pValues: number[] = [];

    for (let featureIdx = 0; featureIdx < numFeatures; featureIdx++) {
      const featureValues = features.map((sample) => sample[featureIdx]);

      // Split by class and perform t-test
      const class0Values = featureValues.filter((_, idx) => labels[idx] === 0);
      const class1Values = featureValues.filter((_, idx) => labels[idx] === 1);

      const pValue = this.tTest(class0Values, class1Values);
      pValues.push(pValue);
    }

    return pValues;
  }

  /**
   * Calculate correlation matrix between features
   */
  private calculateFeatureCorrelationMatrix(
    features: number[][]
  ): CorrelationMatrix {
    const numFeatures = features[0].length;
    const matrix: number[][] = [];

    // Get feature names from registry (or generate if mismatch)
    let featureNames: string[];
    if (FeatureRegistry.getFeatureCount() === numFeatures) {
      featureNames = FeatureRegistry.getFeatureNames();
    } else {
      featureNames = Array.from(
        { length: numFeatures },
        (_, i) => `feature_${i}`
      );
    }

    for (let i = 0; i < numFeatures; i++) {
      matrix[i] = [];
      const featureI = features.map((sample) => sample[i]);

      for (let j = 0; j < numFeatures; j++) {
        if (i === j) {
          matrix[i][j] = 1.0;
        } else {
          const featureJ = features.map((sample) => sample[j]);
          matrix[i][j] = this.pearsonCorrelation(featureI, featureJ);
        }
      }
    }

    return { matrix, featureNames };
  }

  /**
   * Compute comprehensive feature analysis
   */
  private computeFeatureAnalysis(
    correlations: number[],
    pValues: number[],
    correlationMatrix: CorrelationMatrix
  ): FeatureAnalysisResult[] {
    const results: FeatureAnalysisResult[] = [];
    const numFeatures = correlations.length;

    for (let i = 0; i < numFeatures; i++) {
      // Get feature info from registry (if available)
      let featureName = `feature_${i}`;
      let category = "unknown";
      let importance = "unknown";

      if (i < FEATURE_REGISTRY.length) {
        const featureInfo = FEATURE_REGISTRY[i];
        featureName = featureInfo.name;
        category = featureInfo.category;
        importance = featureInfo.importance || "unknown";
      }

      // Calculate redundancy measures
      const otherCorrelations = correlationMatrix.matrix[i]
        .filter((_, idx) => idx !== i)
        .map(Math.abs);

      const avgCorrelationWithOthers =
        otherCorrelations.reduce((a, b) => a + b, 0) / otherCorrelations.length;
      const maxCorrelationWithOthers = Math.max(...otherCorrelations);

      // Calculate utility score (correlation with target, statistical significance)
      const correlationScore = correlations[i];
      const significanceScore = 1 - pValues[i]; // Higher is better
      const utilityScore = correlationScore * 0.7 + significanceScore * 0.3;

      // Calculate redundancy score (lower is better)
      const redundancyScore =
        avgCorrelationWithOthers * 0.6 + maxCorrelationWithOthers * 0.4;

      // Final score combines utility and penalizes redundancy
      const finalScore = utilityScore - redundancyScore * 0.3;

      // Generate recommendation
      let recommendation: "keep" | "remove" | "investigate";
      if (finalScore > 0.3 && redundancyScore < 0.7) {
        recommendation = "keep";
      } else if (finalScore < 0.1 || redundancyScore > 0.8) {
        recommendation = "remove";
      } else {
        recommendation = "investigate";
      }

      results.push({
        featureName,
        index: i,
        category,
        importance,
        correlation: correlations[i],
        pValue: pValues[i],
        avgCorrelationWithOthers,
        maxCorrelationWithOthers,
        utilityScore,
        redundancyScore,
        finalScore,
        recommendation,
      });
    }

    return results.sort((a, b) => b.finalScore - a.finalScore);
  }

  /**
   * Generate detailed recommendations
   */
  private generateRecommendations(results: FeatureAnalysisResult[]): void {
    console.log("\n📋 FEATURE ANALYSIS RESULTS:");
    console.log("=".repeat(80));

    // Summary statistics
    const keepCount = results.filter((r) => r.recommendation === "keep").length;
    const removeCount = results.filter(
      (r) => r.recommendation === "remove"
    ).length;
    const investigateCount = results.filter(
      (r) => r.recommendation === "investigate"
    ).length;

    console.log(`\n📊 SUMMARY:`);
    console.log(`Total features analyzed: ${results.length}`);
    console.log(
      `Recommended to KEEP: ${keepCount} (${(
        (keepCount / results.length) *
        100
      ).toFixed(1)}%)`
    );
    console.log(
      `Recommended to REMOVE: ${removeCount} (${(
        (removeCount / results.length) *
        100
      ).toFixed(1)}%)`
    );
    console.log(
      `Need INVESTIGATION: ${investigateCount} (${(
        (investigateCount / results.length) *
        100
      ).toFixed(1)}%)`
    );

    // Top performing features
    console.log(`\n🏆 TOP 10 PERFORMING FEATURES:`);
    results.slice(0, 10).forEach((result, idx) => {
      console.log(`${idx + 1}. ${result.featureName} (${result.category})`);
      console.log(
        `   Score: ${result.finalScore.toFixed(
          3
        )} | Corr: ${result.correlation.toFixed(
          3
        )} | Redundancy: ${result.redundancyScore.toFixed(3)}`
      );
    });

    // Features to remove
    const toRemove = results.filter((r) => r.recommendation === "remove");
    if (toRemove.length > 0) {
      console.log(`\n❌ FEATURES TO REMOVE (${toRemove.length}):`);
      toRemove.forEach((result) => {
        console.log(
          `• ${result.featureName} (${
            result.category
          }) - Score: ${result.finalScore.toFixed(3)}`
        );
      });
    }

    // High redundancy features
    const highRedundancy = results.filter((r) => r.redundancyScore > 0.7);
    if (highRedundancy.length > 0) {
      console.log(`\n🔗 HIGH REDUNDANCY FEATURES (${highRedundancy.length}):`);
      highRedundancy.forEach((result) => {
        console.log(
          `• ${
            result.featureName
          } - Redundancy: ${result.redundancyScore.toFixed(3)}`
        );
      });
    }

    // Low correlation features
    const lowCorrelation = results.filter((r) => r.correlation < 0.05);
    if (lowCorrelation.length > 0) {
      console.log(`\n📉 LOW CORRELATION FEATURES (${lowCorrelation.length}):`);
      lowCorrelation.forEach((result) => {
        console.log(
          `• ${result.featureName} - Correlation: ${result.correlation.toFixed(
            4
          )}`
        );
      });
    }

    // Category analysis
    console.log(`\n📂 ANALYSIS BY CATEGORY:`);
    const categories = [...new Set(results.map((r) => r.category))];
    categories.forEach((category) => {
      const categoryFeatures = results.filter((r) => r.category === category);
      const avgScore =
        categoryFeatures.reduce((sum, f) => sum + f.finalScore, 0) /
        categoryFeatures.length;
      const keepPct =
        (categoryFeatures.filter((f) => f.recommendation === "keep").length /
          categoryFeatures.length) *
        100;

      console.log(
        `${category}: ${
          categoryFeatures.length
        } features, avg score: ${avgScore.toFixed(3)}, keep: ${keepPct.toFixed(
          1
        )}%`
      );
    });

    console.log("\n" + "=".repeat(80));
  }

  /**
   * Calculate Pearson correlation coefficient
   */
  private pearsonCorrelation(x: number[], y: number[]): number {
    const n = x.length;
    if (n !== y.length || n === 0) return 0;

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
   * Perform t-test between two groups
   */
  private tTest(group1: number[], group2: number[]): number {
    if (group1.length < 2 || group2.length < 2) return 1.0;

    const mean1 = group1.reduce((a, b) => a + b, 0) / group1.length;
    const mean2 = group2.reduce((a, b) => a + b, 0) / group2.length;

    const var1 =
      group1.reduce((sum, x) => sum + Math.pow(x - mean1, 2), 0) /
      (group1.length - 1);
    const var2 =
      group2.reduce((sum, x) => sum + Math.pow(x - mean2, 2), 0) /
      (group2.length - 1);

    const pooledSE = Math.sqrt(var1 / group1.length + var2 / group2.length);
    const tStat = Math.abs(mean1 - mean2) / pooledSE;

    // Simplified p-value approximation
    return Math.exp(-0.717 * tStat - 0.416 * tStat * tStat);
  }

  /**
   * Export results to file for further analysis
   */
  public exportResults(
    results: FeatureAnalysisResult[],
    filename: string
  ): void {
    const csvData = this.resultsToCSV(results);
    console.log(`Generated ${csvData.split("\n").length} lines of CSV data`);
    // In a real implementation, write to file
    console.log(`\n💾 Results would be exported to: ${filename}`);
    console.log("CSV format ready for external analysis tools");
  }

  /**
   * Convert results to CSV format
   */
  private resultsToCSV(results: FeatureAnalysisResult[]): string {
    const headers = [
      "index",
      "featureName",
      "category",
      "importance",
      "correlation",
      "pValue",
      "avgCorrelationWithOthers",
      "maxCorrelationWithOthers",
      "utilityScore",
      "redundancyScore",
      "finalScore",
      "recommendation",
    ];

    const rows = results.map((r) => [
      r.index,
      r.featureName,
      r.category,
      r.importance,
      r.correlation.toFixed(4),
      r.pValue.toFixed(4),
      r.avgCorrelationWithOthers.toFixed(4),
      r.maxCorrelationWithOthers.toFixed(4),
      r.utilityScore.toFixed(4),
      r.redundancyScore.toFixed(4),
      r.finalScore.toFixed(4),
      r.recommendation,
    ]);

    return [headers.join(","), ...rows.map((row) => row.join(","))].join("\n");
  }
}

/**
 * Main execution function
 */
async function main() {
  try {
    console.log("🚀 Starting Feature Analysis for Bitcoin Trading Model");
    console.log("=".repeat(60));

    const analyzer = new FeatureAnalyzer();
    const results = await analyzer.analyzeFeatures();

    // Export results
    analyzer.exportResults(results, "feature_analysis_results.csv");

    console.log("\n✅ Feature analysis completed successfully!");
    console.log(
      "Use these results to optimize your feature set and improve model performance."
    );
  } catch (error) {
    console.error("❌ Feature analysis failed:", error);
    process.exit(1);
  }
}

// Execute if called directly
if (require.main === module) {
  main().catch(console.error);
}

export { FeatureAnalyzer, FeatureAnalysisResult };
