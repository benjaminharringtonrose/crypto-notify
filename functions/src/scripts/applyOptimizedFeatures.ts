#!/usr/bin/env ts-node
/**
 * Apply Optimized Features - Update FeatureCalculator with gradual optimization results
 *
 * This script applies the results from gradual feature optimization to update
 * the FeatureCalculator with the optimal 25-feature set.
 */

import { FeatureDetector } from "../bitcoin/FeatureDetector";
import { FEATURE_REGISTRY } from "../bitcoin/FeatureRegistry";

// Results from gradual optimization
const OPTIMIZATION_RESULTS = {
  originalFeatureCount: 36,
  finalFeatureCount: 25,
  removedFeatures: [
    "feature_23",
    "feature_6",
    "feature_4",
    "feature_9",
    "feature_20",
    "feature_11",
    "feature_19",
    "feature_18",
    "feature_1",
    "feature_16",
    "feature_30",
  ],
  keptFeatures: [
    "feature_24",
    "feature_12",
    "feature_13",
    "feature_10",
    "feature_21",
    "feature_3",
    "feature_0",
  ],
  performanceImprovement: "40.54%",
};

// Map feature indices to actual feature names based on current FeatureRegistry
const FEATURE_INDEX_MAP = {
  0: "priceChangePct",
  1: "volumeChangePct",
  2: "highLowRange",
  3: "priceVolatility",
  4: "volumeMA5Ratio",
  5: "pricePosition",
  6: "volumeMA20Ratio",
  7: "relativeVolume",
  8: "rsi",
  9: "macdLine",
  10: "signalLine",
  11: "bollingerPosition",
  12: "vwapRatio",
  13: "atr",
  14: "obv",
  15: "momentum",
  16: "prevRsi",
  17: "macdHistogram",
  18: "rsiNormalized",
  19: "volumeRatio",
  20: "priceSMA7Ratio",
  21: "priceSMA21Ratio",
  22: "priceSMA50Ratio",
  23: "bollingerBandPosition",
  24: "trendRegime",
  25: "volatilityRegime",
  26: "ichimokuTenkanSen",
  27: "ichimokuKijunSen",
  28: "ichimokuCloudPosition",
  29: "williamsR",
  30: "stochasticK",
  31: "vpt",
  32: "volumeMA20",
  33: "volumeOscillator",
  34: "bollingerSqueeze",
  35: "rsiDivergence",
};

class OptimizedFeatureApplier {
  /**
   * Apply the optimized feature set
   */
  public async applyOptimizedFeatures(): Promise<void> {
    console.log("🎯 Applying Optimized Feature Set");
    console.log("=".repeat(60));

    // Step 1: Initialize feature detection
    console.log("🔧 Initializing feature detection...");
    await FeatureDetector.detectFeatureCount();

    const currentFeatureCount = FeatureDetector.getFeatureCount();
    console.log(`📊 Current feature count: ${currentFeatureCount}`);

    // Step 2: Map feature indices to names
    console.log("🔍 Mapping feature indices to names...");
    const removedFeatureNames = this.mapFeatureIndicesToNames(
      OPTIMIZATION_RESULTS.removedFeatures
    );
    const keptFeatureNames = this.mapFeatureIndicesToNames(
      OPTIMIZATION_RESULTS.keptFeatures
    );

    // Step 3: Generate optimized feature array
    console.log("📝 Generating optimized feature array...");
    const optimizedFeatureArray = this.generateOptimizedFeatureArray();

    // Step 4: Generate updated FeatureRegistry
    console.log("📋 Generating updated FeatureRegistry...");
    const updatedRegistry = this.generateUpdatedRegistry(optimizedFeatureArray);

    // Step 5: Present results
    this.presentResults(
      removedFeatureNames,
      keptFeatureNames,
      optimizedFeatureArray,
      updatedRegistry
    );
  }

  /**
   * Map feature indices to actual feature names
   */
  private mapFeatureIndicesToNames(featureIndices: string[]): string[] {
    return featureIndices.map((featureIndex) => {
      const index = parseInt(featureIndex.replace("feature_", ""));
      return (
        FEATURE_INDEX_MAP[index as keyof typeof FEATURE_INDEX_MAP] ||
        featureIndex
      );
    });
  }

  /**
   * Generate the optimized feature array for FeatureCalculator
   */
  private generateOptimizedFeatureArray(): string[] {
    // Features to keep (25 features) - based on gradual optimization results
    const keptIndices = [
      0, 2, 3, 5, 7, 8, 10, 12, 13, 14, 15, 17, 20, 21, 22, 24, 25, 26, 27, 28,
      29, 31, 32, 33, 34, 35,
    ];

    return keptIndices.map((index) => {
      const featureName =
        FEATURE_INDEX_MAP[index as keyof typeof FEATURE_INDEX_MAP];
      return featureName || `feature_${index}`;
    });
  }

  /**
   * Generate updated FeatureRegistry
   */
  private generateUpdatedRegistry(featureArray: string[]): any[] {
    const registry: any[] = [];

    featureArray.forEach((featureName, index) => {
      const originalFeature = FEATURE_REGISTRY.find(
        (f) => f.name === featureName
      );

      if (originalFeature) {
        registry.push({
          name: originalFeature.name,
          description: originalFeature.description,
          category: originalFeature.category,
          experimentAdded: originalFeature.experimentAdded,
          importance: originalFeature.importance,
        });
      } else {
        // Fallback for unknown features
        registry.push({
          name: featureName,
          description: `${featureName} technical indicator`,
          category: "technical" as const,
          importance: "medium" as const,
        });
      }
    });

    return registry;
  }

  /**
   * Present results and implementation instructions
   */
  private presentResults(
    removedFeatureNames: string[],
    keptFeatureNames: string[],
    optimizedFeatureArray: string[],
    updatedRegistry: any[]
  ): void {
    console.log("\n" + "=".repeat(70));
    console.log("🎯 OPTIMIZED FEATURE SET IMPLEMENTATION");
    console.log("=".repeat(70));

    console.log(`\n📊 OPTIMIZATION SUMMARY:`);
    console.log(
      `Original features: ${OPTIMIZATION_RESULTS.originalFeatureCount}`
    );
    console.log(
      `Optimized features: ${OPTIMIZATION_RESULTS.finalFeatureCount}`
    );
    console.log(`Features removed: ${removedFeatureNames.length}`);
    console.log(
      `Performance improvement: ${OPTIMIZATION_RESULTS.performanceImprovement}`
    );

    console.log(`\n✅ FEATURES REMOVED (${removedFeatureNames.length}):`);
    removedFeatureNames.forEach((feature, idx) => {
      console.log(`  ${idx + 1}. ${feature}`);
    });

    console.log(`\n❌ FEATURES KEPT (${keptFeatureNames.length}):`);
    keptFeatureNames.forEach((feature, idx) => {
      console.log(`  ${idx + 1}. ${feature}`);
    });

    console.log(
      `\n📝 OPTIMIZED FEATURE ARRAY (${optimizedFeatureArray.length} features):`
    );
    console.log("```typescript");
    console.log(
      "// OPTIMIZED FEATURE SET: 25 Most Important Features (Based on Gradual Optimization)"
    );
    console.log(
      "// Selected features that provide maximum trading performance with minimal redundancy"
    );
    console.log("const optimizedFeatures = [");

    optimizedFeatureArray.forEach((feature, idx) => {
      const originalFeature = FEATURE_REGISTRY.find((f) => f.name === feature);
      const comment = originalFeature
        ? ` // ${originalFeature.description}`
        : "";
      console.log(`  indicators.${feature},${comment}`);
    });

    console.log("];");
    console.log("```");

    console.log(
      `\n📋 UPDATED FEATURE REGISTRY (${updatedRegistry.length} features):`
    );
    console.log("```typescript");
    console.log("export const FEATURE_REGISTRY: FeatureDefinition[] = [");

    updatedRegistry.forEach((feature, idx) => {
      console.log(`  {`);
      console.log(`    name: "${feature.name}",`);
      console.log(`    description: "${feature.description}",`);
      console.log(`    category: "${feature.category}",`);
      if (feature.experimentAdded) {
        console.log(`    experimentAdded: "${feature.experimentAdded}",`);
      }
      console.log(`    importance: "${feature.importance}",`);
      console.log(`  },`);
    });

    console.log("];");
    console.log("```");

    console.log("\n" + "=".repeat(70));
    console.log("💡 IMPLEMENTATION STEPS:");
    console.log(
      "1. Update FeatureCalculator.ts with the optimized feature array above"
    );
    console.log("2. Update FeatureRegistry.ts with the updated registry above");
    console.log("3. Update FeatureDetector.ts to expect 25 features");
    console.log("4. Retrain model with optimized feature set");
    console.log("5. Run backtests to validate performance improvement");
    console.log("=".repeat(70));

    console.log(`\n🎯 EXPECTED BENEFITS:`);
    console.log(`• 40.54% improvement in combined F1 score`);
    console.log(`• Better balanced buy/sell predictions`);
    console.log(`• Faster training (25 vs 36 features)`);
    console.log(`• Reduced model complexity`);
    console.log(`• Maintained accuracy with fewer features`);
    console.log("=".repeat(70));
  }
}

/**
 * Main execution function
 */
async function main() {
  try {
    console.log("🎯 Applying Optimized Feature Set from Gradual Optimization");
    console.log("=".repeat(60));

    const applier = new OptimizedFeatureApplier();
    await applier.applyOptimizedFeatures();

    console.log("\n✅ Feature optimization application completed!");
    console.log("Follow the implementation steps above to update your code.");
  } catch (error) {
    console.error("❌ Feature optimization application failed:", error);
    process.exit(1);
  }
}

// Execute if called directly
if (require.main === module) {
  main().catch(console.error);
}

export { OptimizedFeatureApplier };
