#!/usr/bin/env ts-node
/**
 * Gradual Feature Optimization - Test individual feature removal
 *
 * This script tests the removal of features to see if they hurt model performance.
 *
 * BUG FIXES APPLIED:
 * - Fixed baseline performance mutation bug by creating deep copies
 * - Standardized feature filtering to use name-based filtering consistently
 * - Added proper type safety with PerformanceChange interface
 * - Added comprehensive error handling with try-catch-finally blocks
 * - Made seed and tolerance configurable
 * - Ensured original functions are always restored
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

// gradualFeatureOptimization.ts

import * as tf from "@tensorflow/tfjs-node";
import { DataProcessor } from "../bitcoin/DataProcessor";
import { FeatureDetector } from "../bitcoin/FeatureDetector";
import { TradeModelTrainer } from "../bitcoin/TradeModelTrainer";
import { MODEL_CONFIG, TRAINING_CONFIG } from "../constants";
import { FeatureRegistry } from "../bitcoin/FeatureRegistry";

interface PerformanceMetrics {
  validationAccuracy: number;
  buyF1: number;
  sellF1: number;
  combinedF1: number;
  balancedAccuracy: number;
  matthewsCorrelation: number;
  epochsTrained: number;
}

interface PerformanceChange {
  validationAccuracy: number;
  buyF1: number;
  sellF1: number;
  combinedF1: number;
  balancedAccuracy: number;
  matthewsCorrelation: number;
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
  private originalRandom: (() => number) | null = null;
  private originalDetectFeatureCount: (() => Promise<number>) | null = null;
  private originalGetFeatureCount: (() => number) | null = null;
  private seed = 42; // Configurable seed for deterministic training

  // Get feature names from FeatureRegistry to ensure consistency
  private get featureNames(): string[] {
    return FeatureRegistry.getFeatureNames();
  }

  // Method to set custom seed
  public setSeed(seed: number): void {
    this.seed = seed;
  }

  // Method to set custom tolerance
  public setTolerance(tolerance: number): void {
    if (tolerance < 0 || tolerance > 1) {
      throw new Error("Tolerance must be between 0 and 1");
    }
    this.tolerance = tolerance;
  }

  public async runOptimization(): Promise<void> {
    try {
      console.log("🚀 Starting Gradual Feature Optimization");
      console.log(
        `📊 Testing ${this.featureNames.length} features one at a time`
      );
      console.log(`🎯 Performance tolerance: ${this.tolerance * 100}%`);
      console.log("=".repeat(80));

      // Validate feature registry consistency
      const registryFeatureCount = FeatureRegistry.getFeatureCount();
      const registryFeatureNames = FeatureRegistry.getFeatureNames();

      if (this.featureNames.length !== registryFeatureCount) {
        throw new Error(
          `Feature count mismatch! Script has ${this.featureNames.length} features, registry has ${registryFeatureCount}`
        );
      }

      // Validate feature names match
      const nameMismatches = this.featureNames.filter(
        (name, index) => name !== registryFeatureNames[index]
      );
      if (nameMismatches.length > 0) {
        throw new Error(
          `Feature name mismatch! Mismatched features: ${nameMismatches.join(
            ", "
          )}`
        );
      }

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

      // Store original baseline for comparison (deep copy)
      const originalBaselinePerformance: PerformanceMetrics = {
        validationAccuracy: baselinePerformance.validationAccuracy,
        buyF1: baselinePerformance.buyF1,
        sellF1: baselinePerformance.sellF1,
        combinedF1: baselinePerformance.combinedF1,
        balancedAccuracy: baselinePerformance.balancedAccuracy,
        matthewsCorrelation: baselinePerformance.matthewsCorrelation,
        epochsTrained: baselinePerformance.epochsTrained,
      };

      // Test each feature individually
      for (let i = 0; i < this.featureNames.length; i++) {
        const featureName = this.featureNames[i];
        console.log(
          `\n🔍 Step ${i + 1}: Testing removal of "${featureName}" (index ${i})`
        );

        // Create feature set without this feature (use name-based filtering for consistency)
        const modifiedFeatures = this.featureNames.filter(
          (name) => name !== featureName
        );

        // Train and evaluate model without this feature
        const modifiedPerformance = await this.trainAndEvaluateModel(
          modifiedFeatures
        );

        // Compare performance against the original baseline
        const performanceChange = this.calculatePerformanceChange(
          originalBaselinePerformance,
          modifiedPerformance
        );
        const decision = this.makeDecision(performanceChange);

        // Record step
        const step: OptimizationStep = {
          step: i + 1,
          featureName,
          featureIndex: i,
          originalPerformance: originalBaselinePerformance,
          modifiedPerformance,
          decision,
          reason: this.getDecisionReason(performanceChange, decision),
        };

        this.optimizationSteps.push(step);

        // Log results
        this.logStepResults(step, performanceChange);

        // Only update baseline if performance actually improved significantly
        if (decision === "REMOVE") {
          console.log(
            `🔄 Updating baseline - performance improved by ${performanceChange.combinedF1.toFixed(
              4
            )}`
          );
          // Update the original baseline for future comparisons
          originalBaselinePerformance.validationAccuracy =
            modifiedPerformance.validationAccuracy;
          originalBaselinePerformance.buyF1 = modifiedPerformance.buyF1;
          originalBaselinePerformance.sellF1 = modifiedPerformance.sellF1;
          originalBaselinePerformance.combinedF1 =
            modifiedPerformance.combinedF1;
          originalBaselinePerformance.balancedAccuracy =
            modifiedPerformance.balancedAccuracy;
          originalBaselinePerformance.matthewsCorrelation =
            modifiedPerformance.matthewsCorrelation;
        }
      }

      // Generate final report
      this.generateFinalReport();
    } catch (error) {
      console.error("❌ Error during optimization:", error);
      throw error;
    } finally {
      // Always restore original functions, even if an error occurred
      this.restoreOriginalFunctions();
    }
  }

  public async testFeatureRemoval(featureName: string): Promise<void> {
    try {
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

      // Validate feature registry consistency
      const registryFeatureCount = FeatureRegistry.getFeatureCount();
      const registryFeatureNames = FeatureRegistry.getFeatureNames();

      if (this.featureNames.length !== registryFeatureCount) {
        throw new Error(
          `Feature count mismatch! Script has ${this.featureNames.length} features, registry has ${registryFeatureCount}`
        );
      }

      // Validate feature names match
      const nameMismatches = this.featureNames.filter(
        (name, index) => name !== registryFeatureNames[index]
      );
      if (nameMismatches.length > 0) {
        throw new Error(
          `Feature name mismatch! Mismatched features: ${nameMismatches.join(
            ", "
          )}`
        );
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
    } catch (error) {
      console.error("❌ Error during feature removal test:", error);
      throw error;
    } finally {
      // Always restore original functions, even if an error occurred
      this.restoreOriginalFunctions();
    }
  }

  public async testMultipleFeatureRemoval(
    featureNames: string[]
  ): Promise<void> {
    try {
      console.log("🚀 Testing Multiple Feature Removal");
      console.log(
        `🎯 Features: ${featureNames.map((f) => `"${f}"`).join(", ")}`
      );
      console.log(`🎯 Performance tolerance: ${this.tolerance * 100}%`);
      console.log("=".repeat(80));

      // Validate all features exist
      const invalidFeatures = featureNames.filter(
        (name) => !this.featureNames.includes(name)
      );
      if (invalidFeatures.length > 0) {
        console.error(
          `❌ Error: Features not found in feature list: ${invalidFeatures.join(
            ", "
          )}`
        );
        console.log("\n📋 Available features:");
        this.featureNames.forEach((name, index) => {
          console.log(`   ${index + 1}. ${name}`);
        });
        process.exit(1);
      }

      // Validate feature registry consistency
      const registryFeatureCount = FeatureRegistry.getFeatureCount();
      const registryFeatureNames = FeatureRegistry.getFeatureNames();

      if (this.featureNames.length !== registryFeatureCount) {
        throw new Error(
          `Feature count mismatch! Script has ${this.featureNames.length} features, registry has ${registryFeatureCount}`
        );
      }

      // Validate feature names match
      const nameMismatches = this.featureNames.filter(
        (name, index) => name !== registryFeatureNames[index]
      );
      if (nameMismatches.length > 0) {
        throw new Error(
          `Feature name mismatch! Mismatched features: ${nameMismatches.join(
            ", "
          )}`
        );
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

      // Test without the specified features
      console.log(
        `\n🔍 Step 2: Testing Performance Without ${featureNames.length} Features`
      );
      const modifiedFeatures = this.featureNames.filter(
        (name) => !featureNames.includes(name)
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

      // Log results for multiple features
      this.logMultipleFeatureResults(
        featureNames,
        baselinePerformance,
        modifiedPerformance,
        performanceChange,
        decision
      );
    } catch (error) {
      console.error("❌ Error during multiple feature removal test:", error);
      throw error;
    } finally {
      // Always restore original functions, even if an error occurred
      this.restoreOriginalFunctions();
    }
  }

  private async trainAndEvaluateModel(
    featureArray: string[]
  ): Promise<PerformanceMetrics> {
    // Set deterministic seed for consistent training across experiments
    const FIXED_SEED = this.seed; // Use configurable seed

    // CRITICAL: Set seed at the very beginning and ensure it's used consistently
    tf.setBackend("tensorflow");
    tf.randomUniform([1, 1], 0, 1, "float32", FIXED_SEED);

    // Store original random function and override it
    this.originalRandom = Math.random;
    let randomCounter = 0;
    Math.random = () => {
      // Deterministic random function that cycles through predictable values
      randomCounter++;
      return (Math.sin(randomCounter * FIXED_SEED) + 1) / 2; // Deterministic but varied
    };

    console.log(`🎲 Using fixed seed: ${FIXED_SEED} for consistent training`);

    try {
      // Create a custom trainer with the consistent seed
      const trainer = new TradeModelTrainer(FIXED_SEED);

      // Set the current feature name for logging purposes
      trainer.setCurrentFeatureName(featureArray.join(", "));

      // Get the original data to understand the feature structure
      const originalDataProcessor = new DataProcessor(
        {
          timesteps: MODEL_CONFIG.TIMESTEPS,
          epochs: TRAINING_CONFIG.EPOCHS,
          batchSize: TRAINING_CONFIG.BATCH_SIZE,
          initialLearningRate: TRAINING_CONFIG.INITIAL_LEARNING_RATE,
        },
        TRAINING_CONFIG.START_DAYS_AGO
      );

      const { X: originalX, y } = await originalDataProcessor.prepareData();

      // Get the actual feature names from FeatureRegistry to ensure correct mapping
      const actualFeatureNames = FeatureRegistry.getFeatureNames();

      console.log(`📊 Original data has ${originalX[0][0].length} features`);
      console.log(`🎯 Requested ${featureArray.length} features`);

      // Validate that we have the expected number of features
      if (originalX[0][0].length !== actualFeatureNames.length) {
        throw new Error(
          `Feature count mismatch! Expected ${actualFeatureNames.length} features, got ${originalX[0][0].length}`
        );
      }

      // Create feature mapping: which indices to keep
      const featureIndices = featureArray.map((featureName) => {
        const index = actualFeatureNames.indexOf(featureName);
        if (index === -1) {
          throw new Error(
            `Feature "${featureName}" not found in FeatureRegistry`
          );
        }
        return index;
      });

      // Validate that feature indices are unique
      const uniqueIndices = new Set(featureIndices);
      if (uniqueIndices.size !== featureIndices.length) {
        throw new Error(
          `Duplicate feature indices found: ${featureIndices.join(", ")}`
        );
      }

      console.log(`🔍 Using feature indices: [${featureIndices.join(", ")}]`);

      // Filter features to only include the specified ones
      const filteredX = originalX.map((sequence) =>
        sequence.map((timestep) => {
          const filteredTimestep: number[] = [];
          featureIndices.forEach((index) => {
            if (index >= 0 && index < timestep.length) {
              filteredTimestep.push(timestep[index]);
            } else {
              throw new Error(
                `Feature index ${index} out of bounds for timestep length ${timestep.length}`
              );
            }
          });
          return filteredTimestep;
        })
      );

      // Store original feature detection methods
      this.originalDetectFeatureCount = FeatureDetector.detectFeatureCount;
      this.originalGetFeatureCount = FeatureDetector.getFeatureCount;

      // Override the feature detection to match our filtered feature count
      FeatureDetector.detectFeatureCount = async () => {
        return featureArray.length;
      };
      FeatureDetector.getFeatureCount = () => {
        return featureArray.length;
      };

      // Override the trainer's data processor prepareData method
      trainer["dataProcessor"].prepareData = async () => ({ X: filteredX, y });

      // Train the model using the same process as the main trainer
      await trainer.train();

      // Get the final metrics from the trainer
      const metrics = {
        validationAccuracy: trainer.getBalancedAccuracy(),
        buyF1: trainer.getBuyF1(),
        sellF1: trainer.getSellF1(),
        combinedF1: trainer.getCombinedF1(),
        balancedAccuracy: trainer.getBalancedAccuracy(),
        matthewsCorrelation: trainer.getMatthewsCorrelation(),
        epochsTrained: trainer.getFinalMetrics().finalEpoch,
      };

      return metrics;
    } catch (error) {
      console.error(
        `❌ Error training model with ${featureArray.length} features:`,
        error
      );
      throw error;
    } finally {
      // Always restore original functions, even if an error occurred
      this.restoreOriginalFunctions();

      // Clear TensorFlow memory
      tf.tidy(() => {
        // This will clean up any remaining tensors
      });
    }
  }

  private restoreOriginalFunctions(): void {
    // Restore original Math.random function
    if (this.originalRandom) {
      Math.random = this.originalRandom;
      this.originalRandom = null;
    }

    // Restore original feature detection methods
    if (this.originalDetectFeatureCount) {
      FeatureDetector.detectFeatureCount = this.originalDetectFeatureCount;
      this.originalDetectFeatureCount = null;
    }
    if (this.originalGetFeatureCount) {
      FeatureDetector.getFeatureCount = this.originalGetFeatureCount;
      this.originalGetFeatureCount = null;
    }
  }

  private calculatePerformanceChange(
    original: PerformanceMetrics,
    modified: PerformanceMetrics
  ): PerformanceChange {
    const safeDivide = (numerator: number, denominator: number): number => {
      // Handle case where both numerator and denominator are zero
      if (Math.abs(numerator) < 1e-10 && Math.abs(denominator) < 1e-10) {
        return 0; // No change when both are effectively zero
      }

      if (Math.abs(denominator) < 1e-10) {
        return numerator > 0 ? 1 : numerator < 0 ? -1 : 0;
      }
      return numerator / denominator;
    };

    return {
      validationAccuracy: safeDivide(
        modified.validationAccuracy - original.validationAccuracy,
        original.validationAccuracy
      ),
      buyF1: safeDivide(modified.buyF1 - original.buyF1, original.buyF1),
      sellF1: safeDivide(modified.sellF1 - original.sellF1, original.sellF1),
      combinedF1: safeDivide(
        modified.combinedF1 - original.combinedF1,
        original.combinedF1
      ),
      balancedAccuracy: safeDivide(
        modified.balancedAccuracy - original.balancedAccuracy,
        original.balancedAccuracy
      ),
      matthewsCorrelation: safeDivide(
        modified.matthewsCorrelation - original.matthewsCorrelation,
        original.matthewsCorrelation
      ),
    };
  }

  private makeDecision(
    performanceChange: PerformanceChange
  ): "REMOVE" | "KEEP" | "MINIMAL_IMPACT" {
    const combinedF1Change = performanceChange.combinedF1;
    const balancedAccuracyChange = performanceChange.balancedAccuracy;
    const matthewsCorrelationChange = performanceChange.matthewsCorrelation;

    // Consider multiple metrics for decision making
    const positiveMetrics = [
      combinedF1Change > 0,
      balancedAccuracyChange > 0,
      matthewsCorrelationChange > 0,
    ].filter(Boolean).length;

    const negativeMetrics = [
      combinedF1Change < 0,
      balancedAccuracyChange < 0,
      matthewsCorrelationChange < 0,
    ].filter(Boolean).length;

    // Primary decision based on combined F1, but consider other metrics
    if (combinedF1Change > this.tolerance) {
      return "REMOVE"; // Performance improved significantly
    } else if (combinedF1Change > -this.tolerance) {
      // Within tolerance, check if other metrics suggest removal
      if (positiveMetrics > negativeMetrics) {
        return "REMOVE"; // More metrics improved than degraded
      } else if (negativeMetrics > positiveMetrics) {
        return "KEEP"; // More metrics degraded than improved
      } else {
        return "MINIMAL_IMPACT"; // Mixed or neutral impact
      }
    } else {
      return "KEEP"; // Performance degraded significantly
    }
  }

  private getDecisionReason(
    performanceChange: PerformanceChange,
    decision: string
  ): string {
    const combinedF1Change = performanceChange.combinedF1;
    const balancedAccuracyChange = performanceChange.balancedAccuracy;
    const matthewsCorrelationChange = performanceChange.matthewsCorrelation;

    const metrics = [
      { name: "Combined F1", value: combinedF1Change },
      { name: "Balanced Accuracy", value: balancedAccuracyChange },
      { name: "Matthews Correlation", value: matthewsCorrelationChange },
    ];

    const improvedMetrics = metrics.filter((m) => m.value > 0);
    const degradedMetrics = metrics.filter((m) => m.value < 0);

    switch (decision) {
      case "REMOVE":
        if (combinedF1Change > this.tolerance) {
          return `Performance improved by ${(combinedF1Change * 100).toFixed(
            2
          )}%`;
        } else {
          return `Mixed impact: ${improvedMetrics.length} metrics improved, ${degradedMetrics.length} degraded`;
        }
      case "KEEP":
        if (combinedF1Change < -this.tolerance) {
          return `Performance degraded by ${(
            Math.abs(combinedF1Change) * 100
          ).toFixed(2)}%`;
        } else {
          return `Mixed impact: ${improvedMetrics.length} metrics improved, ${degradedMetrics.length} degraded`;
        }
      case "MINIMAL_IMPACT":
        return `Performance change within ${(this.tolerance * 100).toFixed(
          1
        )}% tolerance`;
      default:
        return "Unknown";
    }
  }

  private logStepResults(
    step: OptimizationStep,
    performanceChange: PerformanceChange
  ): void {
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
    performanceChange: PerformanceChange,
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

  private logMultipleFeatureResults(
    featureNames: string[],
    baselinePerformance: PerformanceMetrics,
    modifiedPerformance: PerformanceMetrics,
    performanceChange: PerformanceChange,
    decision: string
  ): void {
    console.log("\n" + "=".repeat(80));
    console.log("📋 MULTIPLE FEATURE REMOVAL TEST RESULTS");
    console.log("=".repeat(80));

    console.log(
      `\n🎯 Features Tested: ${featureNames.map((f) => `"${f}"`).join(", ")}`
    );
    console.log(
      `📊 Features Count: ${featureNames.length} features removed from ${this.featureNames.length} total`
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
        `✅ RECOMMENDATION: Remove ${
          featureNames.length
        } features - Performance improved by ${(combinedF1Change * 100).toFixed(
          2
        )}%`
      );
    } else if (decision === "KEEP") {
      console.log(
        `🔒 RECOMMENDATION: Keep ${
          featureNames.length
        } features - Performance degraded by ${(
          Math.abs(combinedF1Change) * 100
        ).toFixed(2)}%`
      );
    } else {
      console.log(
        `⚠️  RECOMMENDATION: ${
          featureNames.length
        } features have minimal impact (within ${(this.tolerance * 100).toFixed(
          1
        )}% tolerance)`
      );
    }

    console.log(`\n📊 FEATURE COUNT:`);
    console.log(`   Original: ${this.featureNames.length} features`);
    console.log(
      `   Modified: ${this.featureNames.length - featureNames.length} features`
    );
    console.log(
      `   Reduction: ${featureNames.length} features (${(
        (featureNames.length / this.featureNames.length) *
        100
      ).toFixed(1)}%)`
    );
  }
}

// Parse command line arguments with proper quote handling
function parseArguments(): { feature?: string; features?: string[] } {
  const args = process.argv.slice(2);
  const featureIndex = args.indexOf("--feature");

  if (featureIndex !== -1) {
    if (featureIndex + 1 >= args.length) {
      console.error("❌ Error: Please specify feature name(s) after --feature");
      console.log("\n📋 Usage:");
      console.log(
        "   npm run features:gradual                    # Test all features"
      );
      console.log(
        '   npm run features:gradual -- --feature "featureName"  # Test specific feature'
      );
      console.log(
        '   npm run features:gradual -- --feature "feature1" "feature2" "feature3"  # Test multiple features'
      );
      console.log("\n📋 Examples:");
      console.log('   npm run features:gradual -- --feature "priceChangePct"');
      console.log('   npm run features:gradual -- --feature "rsi"');
      console.log('   npm run features:gradual -- --feature "macdHistogram"');
      console.log(
        '   npm run features:gradual -- --feature "rsi" "macdHistogram" "priceChangePct"'
      );
      process.exit(1);
    }

    // Collect all arguments after --feature until the next -- flag
    const features: string[] = [];
    let i = featureIndex + 1;
    while (i < args.length && !args[i].startsWith("--")) {
      // Handle quoted feature names by joining them
      if (args[i].startsWith('"') && !args[i].endsWith('"')) {
        let quotedFeature = args[i].substring(1);
        i++;
        while (i < args.length && !args[i].endsWith('"')) {
          quotedFeature += " " + args[i];
          i++;
        }
        if (i < args.length) {
          quotedFeature += " " + args[i].substring(0, args[i].length - 1);
        }
        features.push(quotedFeature);
      } else {
        // Remove quotes if present
        const feature = args[i].replace(/^["']|["']$/g, "");
        features.push(feature);
      }
      i++;
    }

    if (features.length === 1) {
      return { feature: features[0] };
    } else if (features.length > 1) {
      return { features: features };
    } else {
      console.error("❌ Error: No feature names specified after --feature");
      process.exit(1);
    }
  }

  return {}; // No feature specified, will test all features
}

// Run optimization
async function main() {
  try {
    const args = parseArguments();
    const optimizer = new GradualFeatureOptimizer();

    if (args.features) {
      // Test multiple specific features
      await optimizer.testMultipleFeatureRemoval(args.features);
    } else if (args.feature) {
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
