#!/usr/bin/env ts-node
/**
 * Gradual Feature Optimization - Test individual feature removal
 *
 * This script tests the removal of features to see if they hurt model performance.
 *
 * Usage:
 *   npm run features:gradual                    # Test all features sequentially
 *   npm run features:gradual -- --feature "featureName"  # Test specific feature
 *
 * Examples:
 *   npm run features:gradual
 *   npm run features:gradual -- --feature "priceChangePct"
 *   npm run features:gradual -- --feature "rsi"
 */

import * as tf from "@tensorflow/tfjs-node";
import { DataProcessor } from "../bitcoin/DataProcessor";
import { FeatureDetector } from "../bitcoin/FeatureDetector";
import { TradeModelTrainer } from "../bitcoin/TradeModelTrainer";
import { MODEL_CONFIG, TRAINING_CONFIG } from "../constants";

interface PerformanceMetrics {
  validationAccuracy: number;
  buyF1: number;
  sellF1: number;
  combinedF1: number;
  balancedAccuracy: number;
  matthewsCorrelation: number;
  epochsTrained: number;
}

interface OptimizationStep {
  step: number;
  featureName: string;
  featureIndex: number;
  originalPerformance: PerformanceMetrics;
  modifiedPerformance: PerformanceMetrics;
  decision: "REMOVE" | "KEEP" | "MINIMAL_IMPACT";
  reason: string;
}

class GradualFeatureOptimizer {
  private tolerance = 0.02; // 2% performance tolerance
  private optimizationSteps: OptimizationStep[] = [];

  // Exact feature names from FeatureCalculator.ts optimizedFeatures array
  private readonly featureNames = [
    "priceChangePct",
    "highLowRange",
    "priceVolatility",
    "pricePosition",
    "relativeVolume",
    "rsi",
    "signalLine",
    "vwapRatio",
    "atr",
    "obv",
    "momentum",
    "macdHistogram",
    "priceSMA7Ratio",
    "priceSMA21Ratio",
    "priceSMA50Ratio",

    // 16-20: Market Regime Features (5 features)
    "trendRegime",
    "volatilityRegime",
    "ichimokuTenkanSen",
    "ichimokuKijunSen",
    "ichimokuCloudPosition",
    "williamsR",
    "volumeMA20",
    "volumeOscillator",
    "bollingerSqueeze",
    "rsiDivergence",
    "cci",
    "mfi",
    "aroonOscillator",
    "donchianPosition",
    "parabolicSAR",
    "adx",
    "ichimokuPosition",
    "fibonacciPosition",
    "stochasticK",
    "priceAcceleration",
    "proc",
    "stochRsi",
    "vwma",
    "centerOfGravityOscillator",
    "tsi",
    "pmo",
    "bollingerBandWidth",
    "historicalVolatility",
    "camarillaPivots",
    "acceleratorOscillator",
    "chaikinOscillator",
    "elderForceIndex",
    "klingerVolumeOscillator",
    "massIndex",
    "priceChannel",
  ];

  public async runOptimization(): Promise<void> {
    console.log("🚀 Starting Gradual Feature Optimization");
    console.log(
      `📊 Testing ${this.featureNames.length} features one at a time`
    );
    console.log(`🎯 Performance tolerance: ${this.tolerance * 100}%`);
    console.log("=".repeat(80));

    // Initialize feature detection
    console.log("\n🔧 Initializing feature detection...");
    await FeatureDetector.detectFeatureCount();

    // Establish baseline performance
    console.log(
      "\n📈 Step 0: Establishing Baseline Performance (All Features)"
    );
    const baselinePerformance = await this.trainAndEvaluateModel(
      this.featureNames
    );
    console.log(
      `✅ Baseline: ${baselinePerformance.validationAccuracy.toFixed(
        4
      )} accuracy, ${baselinePerformance.combinedF1.toFixed(4)} combined F1`
    );

    // Test each feature individually
    for (let i = 0; i < this.featureNames.length; i++) {
      const featureName = this.featureNames[i];
      console.log(
        `\n🔍 Step ${i + 1}: Testing removal of "${featureName}" (index ${i})`
      );

      // Create feature set without this feature
      const modifiedFeatures = this.featureNames.filter(
        (_, index) => index !== i
      );

      // Train and evaluate model without this feature
      const modifiedPerformance = await this.trainAndEvaluateModel(
        modifiedFeatures
      );

      // Compare performance
      const performanceChange = this.calculatePerformanceChange(
        baselinePerformance,
        modifiedPerformance
      );
      const decision = this.makeDecision(performanceChange);

      // Record step
      const step: OptimizationStep = {
        step: i + 1,
        featureName,
        featureIndex: i,
        originalPerformance: baselinePerformance,
        modifiedPerformance,
        decision,
        reason: this.getDecisionReason(performanceChange, decision),
      };

      this.optimizationSteps.push(step);

      // Log results
      this.logStepResults(step, performanceChange);

      // If performance improved significantly, update baseline
      if (performanceChange.combinedF1 > this.tolerance) {
        console.log(
          `🔄 Updating baseline - performance improved by ${performanceChange.combinedF1.toFixed(
            4
          )}`
        );
        baselinePerformance.validationAccuracy =
          modifiedPerformance.validationAccuracy;
        baselinePerformance.buyF1 = modifiedPerformance.buyF1;
        baselinePerformance.sellF1 = modifiedPerformance.sellF1;
        baselinePerformance.combinedF1 = modifiedPerformance.combinedF1;
        baselinePerformance.balancedAccuracy =
          modifiedPerformance.balancedAccuracy;
        baselinePerformance.matthewsCorrelation =
          modifiedPerformance.matthewsCorrelation;
      }
    }

    // Generate final report
    this.generateFinalReport();
  }

  public async testFeatureRemoval(featureName: string): Promise<void> {
    console.log("🚀 Testing Individual Feature Removal");
    console.log(`🎯 Testing removal of: "${featureName}"`);
    console.log(`🎯 Performance tolerance: ${this.tolerance * 100}%`);
    console.log("=".repeat(80));

    // Validate feature name
    if (!this.featureNames.includes(featureName)) {
      console.error(
        `❌ Error: Feature "${featureName}" not found in feature list`
      );
      console.log("\n📋 Available features:");
      this.featureNames.forEach((name, index) => {
        console.log(`   ${index + 1}. ${name}`);
      });
      process.exit(1);
    }

    // Initialize feature detection
    console.log("\n🔧 Initializing feature detection...");
    await FeatureDetector.detectFeatureCount();

    // Establish baseline performance
    console.log(
      "\n📈 Step 1: Establishing Baseline Performance (All Features)"
    );
    const baselinePerformance = await this.trainAndEvaluateModel(
      this.featureNames
    );
    console.log(
      `✅ Baseline: ${baselinePerformance.validationAccuracy.toFixed(
        4
      )} accuracy, ${baselinePerformance.combinedF1.toFixed(4)} combined F1`
    );

    // Test without the specified feature
    console.log(`\n🔍 Step 2: Testing Performance Without "${featureName}"`);
    const modifiedFeatures = this.featureNames.filter(
      (name) => name !== featureName
    );
    const modifiedPerformance = await this.trainAndEvaluateModel(
      modifiedFeatures
    );

    // Compare performance
    const performanceChange = this.calculatePerformanceChange(
      baselinePerformance,
      modifiedPerformance
    );
    const decision = this.makeDecision(performanceChange);

    // Log results
    this.logResults(
      featureName,
      baselinePerformance,
      modifiedPerformance,
      performanceChange,
      decision
    );
  }

  private async trainAndEvaluateModel(
    featureArray: string[]
  ): Promise<PerformanceMetrics> {
    // Set deterministic seed for consistent training across experiments
    const FIXED_SEED = 42; // Use same seed for all experiments

    // CRITICAL: Set seed at the very beginning and ensure it's used consistently
    tf.setBackend("tensorflow");
    tf.randomUniform([1, 1], 0, 1, "float32", FIXED_SEED);

    // Also set global random seed for any other random operations
    const originalRandom = Math.random;
    let randomCounter = 0;
    Math.random = () => {
      // Deterministic random function that cycles through predictable values
      randomCounter++;
      return (Math.sin(randomCounter * FIXED_SEED) + 1) / 2; // Deterministic but varied
    };

    console.log(`🎲 Using fixed seed: ${FIXED_SEED} for consistent training`);

    // Create a custom data processor that uses only specified features
    const dataProcessor = new DataProcessor(
      {
        timesteps: MODEL_CONFIG.TIMESTEPS,
        epochs: TRAINING_CONFIG.EPOCHS,
        batchSize: TRAINING_CONFIG.BATCH_SIZE,
        initialLearningRate: TRAINING_CONFIG.INITIAL_LEARNING_RATE,
      },
      600
    );

    const { X, y } = await dataProcessor.prepareData();

    // Filter features to only include the specified ones
    const filteredX = X.map((sequence) =>
      sequence.map((timestep) => {
        const filteredTimestep: number[] = [];
        featureArray.forEach((featureName) => {
          const index = this.featureNames.indexOf(featureName);
          if (index >= 0 && index < timestep.length) {
            filteredTimestep.push(timestep[index]);
          }
        });
        return filteredTimestep;
      })
    );

    try {
      // Create a custom trainer with the filtered data and consistent seed
      const trainer = new TradeModelTrainer(FIXED_SEED);

      // Override the data processor to use filtered features
      trainer["dataProcessor"].prepareData = async () => {
        return { X: filteredX, y };
      };

      // Override the feature detection to match our filtered feature count
      const originalDetectFeatureCount = FeatureDetector.detectFeatureCount;
      const originalGetFeatureCount = FeatureDetector.getFeatureCount;

      FeatureDetector.detectFeatureCount = async () => {
        return featureArray.length;
      };
      FeatureDetector.getFeatureCount = () => {
        return featureArray.length;
      };

      // Train the model using the same process as the main trainer
      await trainer.train();

      // Restore original feature detection
      FeatureDetector.detectFeatureCount = originalDetectFeatureCount;
      FeatureDetector.getFeatureCount = originalGetFeatureCount;

      // Restore original Math.random function
      Math.random = originalRandom;

      // Get the final metrics from the trainer
      return {
        validationAccuracy: trainer.getBalancedAccuracy(),
        buyF1: trainer.getBuyF1(),
        sellF1: trainer.getSellF1(),
        combinedF1: trainer.getCombinedF1(),
        balancedAccuracy: trainer.getBalancedAccuracy(),
        matthewsCorrelation: trainer.getMatthewsCorrelation(),
        epochsTrained: trainer.getFinalMetrics().finalEpoch,
      };
    } catch (error) {
      console.error(
        `❌ Error training model with ${featureArray.length} features:`,
        error
      );
      throw error;
    }
  }

  private calculatePerformanceChange(
    original: PerformanceMetrics,
    modified: PerformanceMetrics
  ) {
    return {
      validationAccuracy:
        (modified.validationAccuracy - original.validationAccuracy) /
        original.validationAccuracy,
      buyF1: (modified.buyF1 - original.buyF1) / original.buyF1,
      sellF1: (modified.sellF1 - original.sellF1) / original.sellF1,
      combinedF1:
        (modified.combinedF1 - original.combinedF1) / original.combinedF1,
      balancedAccuracy:
        (modified.balancedAccuracy - original.balancedAccuracy) /
        original.balancedAccuracy,
      matthewsCorrelation:
        (modified.matthewsCorrelation - original.matthewsCorrelation) /
        original.matthewsCorrelation,
    };
  }

  private makeDecision(
    performanceChange: any
  ): "REMOVE" | "KEEP" | "MINIMAL_IMPACT" {
    const combinedF1Change = performanceChange.combinedF1;

    if (combinedF1Change > this.tolerance) {
      return "REMOVE"; // Performance improved
    } else if (combinedF1Change > -this.tolerance) {
      return "MINIMAL_IMPACT"; // Within tolerance
    } else {
      return "KEEP"; // Performance degraded
    }
  }

  private getDecisionReason(performanceChange: any, decision: string): string {
    const combinedF1Change = performanceChange.combinedF1;

    switch (decision) {
      case "REMOVE":
        return `Performance improved by ${(combinedF1Change * 100).toFixed(
          2
        )}%`;
      case "KEEP":
        return `Performance degraded by ${(
          Math.abs(combinedF1Change) * 100
        ).toFixed(2)}%`;
      case "MINIMAL_IMPACT":
        return `Performance change within ${(this.tolerance * 100).toFixed(
          1
        )}% tolerance`;
      default:
        return "Unknown";
    }
  }

  private logStepResults(step: OptimizationStep, performanceChange: any): void {
    const { decision, modifiedPerformance } = step;
    const { combinedF1 } = performanceChange;

    console.log(`   📊 Modified Performance:`);
    console.log(
      `      Accuracy: ${modifiedPerformance.validationAccuracy.toFixed(4)}`
    );
    console.log(`      Buy F1: ${modifiedPerformance.buyF1.toFixed(4)}`);
    console.log(`      Sell F1: ${modifiedPerformance.sellF1.toFixed(4)}`);
    console.log(
      `      Combined F1: ${modifiedPerformance.combinedF1.toFixed(4)}`
    );
    console.log(`   📈 Performance Change: ${(combinedF1 * 100).toFixed(2)}%`);
    console.log(`   🎯 Decision: ${decision}`);
    console.log(`   💡 Reason: ${step.reason}`);
  }

  private generateFinalReport(): void {
    console.log("\n" + "=".repeat(80));
    console.log("📋 GRADUAL FEATURE OPTIMIZATION FINAL REPORT");
    console.log("=".repeat(80));

    const removedFeatures = this.optimizationSteps.filter(
      (s) => s.decision === "REMOVE"
    );
    const keptFeatures = this.optimizationSteps.filter(
      (s) => s.decision === "KEEP"
    );
    const minimalImpactFeatures = this.optimizationSteps.filter(
      (s) => s.decision === "MINIMAL_IMPACT"
    );

    console.log(`\n📊 SUMMARY:`);
    console.log(`   Original Features: ${this.featureNames.length}`);
    console.log(`   Features to Remove: ${removedFeatures.length}`);
    console.log(`   Features to Keep: ${keptFeatures.length}`);
    console.log(`   Minimal Impact: ${minimalImpactFeatures.length}`);
    console.log(
      `   Final Feature Count: ${
        this.featureNames.length - removedFeatures.length
      }`
    );

    if (removedFeatures.length > 0) {
      console.log(`\n✅ FEATURES TO REMOVE (${removedFeatures.length}):`);
      removedFeatures.forEach((step) => {
        console.log(
          `   - ${step.featureName} (index ${step.featureIndex}): ${step.reason}`
        );
      });
    }

    if (keptFeatures.length > 0) {
      console.log(`\n🔒 FEATURES TO KEEP (${keptFeatures.length}):`);
      keptFeatures.forEach((step) => {
        console.log(
          `   - ${step.featureName} (index ${step.featureIndex}): ${step.reason}`
        );
      });
    }

    if (minimalImpactFeatures.length > 0) {
      console.log(
        `\n⚠️  MINIMAL IMPACT FEATURES (${minimalImpactFeatures.length}):`
      );
      minimalImpactFeatures.forEach((step) => {
        console.log(
          `   - ${step.featureName} (index ${step.featureIndex}): ${step.reason}`
        );
      });
    }

    // Generate optimized feature array
    const optimizedFeatureNames = this.featureNames.filter((_, index) => {
      const step = this.optimizationSteps.find((s) => s.featureIndex === index);
      return step && step.decision !== "REMOVE";
    });

    console.log(`\n🎯 OPTIMIZED FEATURE ARRAY:`);
    console.log(`const optimizedFeatures = [`);
    optimizedFeatureNames.forEach((name, index) => {
      console.log(`  "${name}", // ${index + 1}`);
    });
    console.log(`];`);

    console.log(`\n📈 PERFORMANCE IMPACT:`);
    const baselineStep = this.optimizationSteps[0];
    const finalStep = this.optimizationSteps[this.optimizationSteps.length - 1];
    if (baselineStep && finalStep) {
      const improvement =
        finalStep.modifiedPerformance.combinedF1 -
        baselineStep.originalPerformance.combinedF1;
      console.log(
        `   Combined F1 Improvement: ${(improvement * 100).toFixed(2)}%`
      );
      console.log(
        `   Final Combined F1: ${finalStep.modifiedPerformance.combinedF1.toFixed(
          4
        )}`
      );
    }
  }

  private logResults(
    featureName: string,
    baselinePerformance: PerformanceMetrics,
    modifiedPerformance: PerformanceMetrics,
    performanceChange: any,
    decision: string
  ): void {
    console.log("\n" + "=".repeat(80));
    console.log("📋 FEATURE REMOVAL TEST RESULTS");
    console.log("=".repeat(80));

    console.log(`\n🎯 Feature Tested: "${featureName}"`);
    console.log(
      `📊 Feature Index: ${this.featureNames.indexOf(featureName) + 1}/${
        this.featureNames.length
      }`
    );

    console.log(`\n📈 PERFORMANCE COMPARISON:`);
    console.log(`   Metric              | Baseline | Modified | Change`);
    console.log(`   --------------------|----------|----------|--------`);
    console.log(
      `   Validation Accuracy | ${baselinePerformance.validationAccuracy.toFixed(
        4
      )} | ${modifiedPerformance.validationAccuracy.toFixed(4)} | ${(
        performanceChange.validationAccuracy * 100
      ).toFixed(2)}%`
    );
    console.log(
      `   Buy F1 Score        | ${baselinePerformance.buyF1.toFixed(
        4
      )} | ${modifiedPerformance.buyF1.toFixed(4)} | ${(
        performanceChange.buyF1 * 100
      ).toFixed(2)}%`
    );
    console.log(
      `   Sell F1 Score       | ${baselinePerformance.sellF1.toFixed(
        4
      )} | ${modifiedPerformance.sellF1.toFixed(4)} | ${(
        performanceChange.sellF1 * 100
      ).toFixed(2)}%`
    );
    console.log(
      `   Combined F1 Score   | ${baselinePerformance.combinedF1.toFixed(
        4
      )} | ${modifiedPerformance.combinedF1.toFixed(4)} | ${(
        performanceChange.combinedF1 * 100
      ).toFixed(2)}%`
    );
    console.log(
      `   Balanced Accuracy   | ${baselinePerformance.balancedAccuracy.toFixed(
        4
      )} | ${modifiedPerformance.balancedAccuracy.toFixed(4)} | ${(
        performanceChange.balancedAccuracy * 100
      ).toFixed(2)}%`
    );
    console.log(
      `   Matthews Correlation| ${baselinePerformance.matthewsCorrelation.toFixed(
        4
      )} | ${modifiedPerformance.matthewsCorrelation.toFixed(4)} | ${(
        performanceChange.matthewsCorrelation * 100
      ).toFixed(2)}%`
    );

    console.log(`\n🎯 DECISION: ${decision}`);

    const combinedF1Change = performanceChange.combinedF1;
    if (decision === "REMOVE") {
      console.log(
        `✅ RECOMMENDATION: Remove "${featureName}" - Performance improved by ${(
          combinedF1Change * 100
        ).toFixed(2)}%`
      );
    } else if (decision === "KEEP") {
      console.log(
        `🔒 RECOMMENDATION: Keep "${featureName}" - Performance degraded by ${(
          Math.abs(combinedF1Change) * 100
        ).toFixed(2)}%`
      );
    } else {
      console.log(
        `⚠️  RECOMMENDATION: "${featureName}" has minimal impact (within ${(
          this.tolerance * 100
        ).toFixed(1)}% tolerance)`
      );
    }

    console.log(`\n📊 FEATURE COUNT:`);
    console.log(`   Original: ${this.featureNames.length} features`);
    console.log(`   Modified: ${this.featureNames.length - 1} features`);
    console.log(
      `   Reduction: 1 feature (${(
        (1 / this.featureNames.length) *
        100
      ).toFixed(1)}%)`
    );
  }
}

// Parse command line arguments
function parseArguments(): { feature?: string } {
  const args = process.argv.slice(2);
  const featureIndex = args.indexOf("--feature");

  if (featureIndex !== -1) {
    if (featureIndex + 1 >= args.length) {
      console.error("❌ Error: Please specify a feature name after --feature");
      console.log("\n📋 Usage:");
      console.log(
        "   npm run features:gradual                    # Test all features"
      );
      console.log(
        '   npm run features:gradual -- --feature "featureName"  # Test specific feature'
      );
      console.log("\n📋 Examples:");
      console.log('   npm run features:gradual -- --feature "priceChangePct"');
      console.log('   npm run features:gradual -- --feature "rsi"');
      console.log('   npm run features:gradual -- --feature "macdHistogram"');
      process.exit(1);
    }
    return { feature: args[featureIndex + 1] };
  }

  return {}; // No feature specified, will test all features
}

// Run optimization
async function main() {
  try {
    const args = parseArguments();
    const optimizer = new GradualFeatureOptimizer();

    if (args.feature) {
      // Test specific feature
      await optimizer.testFeatureRemoval(args.feature);
    } else {
      // Test all features sequentially
      await optimizer.runOptimization();
    }
  } catch (error) {
    console.error("❌ Optimization failed:", error);
    process.exit(1);
  }
}

if (require.main === module) {
  main();
}
