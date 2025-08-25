#!/usr/bin/env ts-node

/**
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
  private tolerance = 0.02;
  private optimizationSteps: OptimizationStep[] = [];
  private originalRandom: (() => number) | null = null;
  private originalDetectFeatureCount: (() => Promise<number>) | null = null;
  private originalGetFeatureCount: (() => number) | null = null;
  private seed = 42;
  private randomCounter = 0;
  private maxOptimizationSteps = 1000;
  private lcgState = 0;

  constructor() {
    this.storeOriginalFunctions();
  }

  private get featureNames(): string[] {
    try {
      return FeatureRegistry.getFeatureNames();
    } catch (error) {
      throw new Error(`Failed to get feature names from registry: ${error}`);
    }
  }

  public setSeed(seed: number): void {
    if (!Number.isInteger(seed) || seed < 0) {
      throw new Error("Seed must be a non-negative integer");
    }
    this.seed = seed;
    this.randomCounter = 0;
  }

  // Method to set custom tolerance
  public setTolerance(tolerance: number): void {
    if (tolerance < 0 || tolerance > 1) {
      throw new Error("Tolerance must be between 0 and 1");
    }
    this.tolerance = tolerance;
  }

  public setMaxOptimizationSteps(maxSteps: number): void {
    if (!Number.isInteger(maxSteps) || maxSteps < 1) {
      throw new Error("Max optimization steps must be a positive integer");
    }
    this.maxOptimizationSteps = maxSteps;
  }

  private deepCopyPerformanceMetrics(
    metrics: PerformanceMetrics
  ): PerformanceMetrics {
    return {
      validationAccuracy: metrics.validationAccuracy,
      buyF1: metrics.buyF1,
      sellF1: metrics.sellF1,
      combinedF1: metrics.combinedF1,
      balancedAccuracy: metrics.balancedAccuracy,
      matthewsCorrelation: metrics.matthewsCorrelation,
      epochsTrained: metrics.epochsTrained,
    };
  }

  private filterFeaturesEfficiently(
    originalX: number[][][],
    featureIndices: number[]
  ): number[][][] {
    const filteredX: number[][][] = [];

    for (let i = 0; i < originalX.length; i++) {
      const sequence = originalX[i];
      const filteredSequence: number[][] = [];

      for (let j = 0; j < sequence.length; j++) {
        const timestep = sequence[j];
        const filteredTimestep = new Array(featureIndices.length);

        for (let k = 0; k < featureIndices.length; k++) {
          filteredTimestep[k] = timestep[featureIndices[k]];
        }

        filteredSequence.push(filteredTimestep);
      }

      filteredX.push(filteredSequence);
    }

    return filteredX;
  }

  private storeOriginalFunctions(): void {
    try {
      if (!this.originalDetectFeatureCount) {
        this.originalDetectFeatureCount = FeatureDetector.detectFeatureCount;
      }
      if (!this.originalGetFeatureCount) {
        this.originalGetFeatureCount = FeatureDetector.getFeatureCount;
      }
    } catch (error) {
      console.warn(
        "⚠️  Failed to store original FeatureDetector functions:",
        error
      );
    }
  }

  private initializeDeterministicRandom(): void {
    this.originalRandom = Math.random;
    this.randomCounter = 0;

    if (this.seed === 0) {
      this.seed = 42;
      console.warn("⚠️  Seed was 0, using default seed 42");
    }

    const a = 1664525;
    const c = 1013904223;
    const m = Math.pow(2, 32);
    this.lcgState = (this.seed || 42) >>> 0;
    Math.random = () => {
      this.randomCounter++;
      this.lcgState = (a * this.lcgState + c) % m;
      return this.lcgState / m;
    };

    try {
      tf.setBackend("tensorflow");
      tf.tidy(() => {
        tf.randomUniform([1, 1], 0, 1, "float32", this.seed);
      });
    } catch (error) {
      console.warn("⚠️  Failed to set TensorFlow seed:", error);
    }
  }

  public async runOptimization(): Promise<void> {
    try {
      console.log("🚀 Starting Gradual Feature Optimization");
      console.log(
        `📊 Testing ${this.featureNames.length} features one at a time`
      );
      console.log(`🎯 Performance tolerance: ${this.tolerance * 100}%`);
      console.log(`🎲 Using seed: ${this.seed}`);
      console.log("=".repeat(80));

      // Initialize deterministic random behavior
      this.initializeDeterministicRandom();

      // Validate feature registry consistency
      await this.validateFeatureRegistry();

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

      const originalBaselinePerformance =
        this.deepCopyPerformanceMetrics(baselinePerformance);

      for (let i = 0; i < this.featureNames.length; i++) {
        const featureName = this.featureNames[i];
        console.log(
          `\n🔍 Step ${i + 1}: Testing removal of "${featureName}" (index ${i})`
        );

        const modifiedFeatures = this.featureNames.filter(
          (name) => name !== featureName
        );

        const modifiedPerformance = await this.trainAndEvaluateModel(
          modifiedFeatures
        );

        const performanceChange = this.calculatePerformanceChange(
          originalBaselinePerformance,
          modifiedPerformance
        );
        const decision = this.makeDecision(performanceChange);

        const step: OptimizationStep = {
          step: i + 1,
          featureName,
          featureIndex: i,
          originalPerformance: this.deepCopyPerformanceMetrics(
            originalBaselinePerformance
          ),
          modifiedPerformance:
            this.deepCopyPerformanceMetrics(modifiedPerformance),
          decision,
          reason: this.getDecisionReason(performanceChange, decision),
        };

        if (this.optimizationSteps.length >= this.maxOptimizationSteps) {
          console.warn(
            `⚠️  Reached maximum optimization steps (${this.maxOptimizationSteps}), clearing older steps`
          );
          this.optimizationSteps = this.optimizationSteps.slice(
            -this.maxOptimizationSteps / 2
          );
        }

        this.optimizationSteps.push(step);

        this.logStepResults(step, performanceChange);

        // This ensures consistent comparison across all feature tests
      }

      this.generateFinalReport();
    } catch (error) {
      console.error("❌ Error during optimization:", error);
      throw error;
    } finally {
      this.restoreOriginalFunctions();
      // Clean up TensorFlow memory
      this.cleanupTensorFlowMemory();
    }
  }

  public async testFeatureRemoval(featureName: string): Promise<void> {
    try {
      console.log("🚀 Testing Individual Feature Removal");
      console.log(`🎯 Testing removal of: "${featureName}"`);
      console.log(`🎯 Performance tolerance: ${this.tolerance * 100}%`);
      console.log(`🎲 Using seed: ${this.seed}`);
      console.log("=".repeat(80));

      // Validate feature name
      if (!this.featureNames.includes(featureName)) {
        const error = `Feature "${featureName}" not found in feature list`;
        console.error(`❌ Error: ${error}`);
        console.log("\n📋 Available features:");
        this.featureNames.forEach((name, index) => {
          console.log(`   ${index + 1}. ${name}`);
        });
        throw new Error(error); // FIXED: Use throw instead of process.exit()
      }

      // Initialize deterministic random behavior
      this.initializeDeterministicRandom();

      // Validate feature registry consistency
      await this.validateFeatureRegistry();

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
      // Clean up TensorFlow memory
      this.cleanupTensorFlowMemory();
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
      console.log(`🎲 Using seed: ${this.seed}`);
      console.log("=".repeat(80));

      // Validate all features exist
      const invalidFeatures = featureNames.filter(
        (name) => !this.featureNames.includes(name)
      );
      if (invalidFeatures.length > 0) {
        const error = `Features not found in feature list: ${invalidFeatures.join(
          ", "
        )}`;
        console.error(`❌ Error: ${error}`);
        console.log("\n📋 Available features:");
        this.featureNames.forEach((name, index) => {
          console.log(`   ${index + 1}. ${name}`);
        });
        throw new Error(error); // FIXED: Use throw instead of process.exit()
      }

      // Initialize deterministic random behavior
      this.initializeDeterministicRandom();

      // Validate feature registry consistency
      await this.validateFeatureRegistry();

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
      this.restoreOriginalFunctions();
      this.cleanupTensorFlowMemory();
    }
  }

  private async validateFeatureRegistry(): Promise<void> {
    try {
      const registryFeatureCount = FeatureRegistry.getFeatureCount();
      const registryFeatureNames = FeatureRegistry.getFeatureNames();

      if (this.featureNames.length !== registryFeatureCount) {
        throw new Error(
          `Feature count mismatch! Script has ${this.featureNames.length} features, registry has ${registryFeatureCount}`
        );
      }

      const scriptFeatureSet = new Set(this.featureNames);
      const registryFeatureSet = new Set(registryFeatureNames);

      if (scriptFeatureSet.size !== registryFeatureSet.size) {
        throw new Error(
          `Feature count mismatch in sets! Script has ${scriptFeatureSet.size} unique features, registry has ${registryFeatureSet.size}`
        );
      }

      // Check for any differences between the sets
      const missingInRegistry = this.featureNames.filter(
        (name) => !registryFeatureSet.has(name)
      );
      const missingInScript = registryFeatureNames.filter(
        (name) => !scriptFeatureSet.has(name)
      );

      if (missingInRegistry.length > 0 || missingInScript.length > 0) {
        const differences = [];
        if (missingInRegistry.length > 0) {
          differences.push(
            `Missing in registry: ${missingInRegistry.join(", ")}`
          );
        }
        if (missingInScript.length > 0) {
          differences.push(`Missing in script: ${missingInScript.join(", ")}`);
        }
        throw new Error(`Feature name mismatch! ${differences.join("; ")}`);
      }

      for (let i = 0; i < registryFeatureNames.length; i++) {
        if (this.featureNames[i] !== registryFeatureNames[i]) {
          throw new Error(
            `Feature order mismatch at index ${i}: script has "${this.featureNames[i]}", registry has "${registryFeatureNames[i]}"`
          );
        }
      }
    } catch (error) {
      throw new Error(`Feature registry validation failed: ${error}`);
    }
  }

  private async trainAndEvaluateModel(
    featureArray: string[]
  ): Promise<PerformanceMetrics> {
    const FIXED_SEED = this.seed;

    console.log(`🎲 Using fixed seed: ${FIXED_SEED} for consistent training`);

    try {
      const trainer = new TradeModelTrainer(FIXED_SEED);

      trainer.setCurrentFeatureName(featureArray.join(", "));

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

      const actualFeatureNames = FeatureRegistry.getFeatureNames();

      console.log(`📊 Original data has ${originalX[0][0].length} features`);
      console.log(`🎯 Requested ${featureArray.length} features`);

      if (originalX[0][0].length !== actualFeatureNames.length) {
        throw new Error(
          `Feature count mismatch! Expected ${actualFeatureNames.length} features, got ${originalX[0][0].length}`
        );
      }

      const featureIndices = featureArray.map((featureName) => {
        const index = actualFeatureNames.indexOf(featureName);
        if (index === -1) {
          throw new Error(
            `Feature "${featureName}" not found in FeatureRegistry`
          );
        }
        return index;
      });

      const uniqueIndices = new Set(featureIndices);
      if (uniqueIndices.size !== featureIndices.length) {
        throw new Error(
          `Duplicate feature indices found: ${featureIndices.join(", ")}`
        );
      }

      const maxIndex = Math.max(...featureIndices);
      if (maxIndex >= originalX[0][0].length) {
        throw new Error(
          `Feature index ${maxIndex} out of bounds for timestep length ${originalX[0][0].length}`
        );
      }

      console.log(`🔍 Using feature indices: [${featureIndices.join(", ")}]`);

      const filteredX = this.filterFeaturesEfficiently(
        originalX,
        featureIndices
      );

      try {
        FeatureDetector.detectFeatureCount = async () => {
          return featureArray.length;
        };
        FeatureDetector.getFeatureCount = () => {
          return featureArray.length;
        };
      } catch (error) {
        console.warn(
          "⚠️  Failed to override feature detection methods:",
          error
        );
      }

      trainer["dataProcessor"].prepareData = async () => ({ X: filteredX, y });

      await trainer.train();

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
    }
  }

  private restoreOriginalFunctions(): void {
    try {
      if (this.originalRandom) {
        Math.random = this.originalRandom;
        this.originalRandom = null;
      }

      if (this.originalDetectFeatureCount) {
        try {
          FeatureDetector.detectFeatureCount = this.originalDetectFeatureCount;
        } catch (error) {
          console.warn("⚠️  Failed to restore detectFeatureCount:", error);
        }
        this.originalDetectFeatureCount = null;
      }

      if (this.originalGetFeatureCount) {
        try {
          FeatureDetector.getFeatureCount = this.originalGetFeatureCount;
        } catch (error) {
          console.warn("⚠️  Failed to restore getFeatureCount:", error);
        }
        this.originalGetFeatureCount = null;
      }
    } catch (error) {
      console.error("❌ Error restoring original functions:", error);
    }
  }

  private cleanupTensorFlowMemory(): void {
    try {
      tf.tidy(() => {
        const dummyTensor = tf.zeros([1, 1]);
        dummyTensor.dispose();
      });

      tf.disposeVariables();

      if (global.gc) {
        global.gc();
      }
    } catch (error) {
      console.error("❌ Error cleaning up TensorFlow memory:", error);
    }
  }

  private calculatePerformanceChange(
    original: PerformanceMetrics,
    modified: PerformanceMetrics
  ): PerformanceChange {
    const safeDivide = (numerator: number, denominator: number): number => {
      if (
        isNaN(numerator) ||
        isNaN(denominator) ||
        !isFinite(numerator) ||
        !isFinite(denominator)
      ) {
        return 0;
      }

      if (Math.abs(denominator) < 1e-8) {
        return 0;
      }

      if (Math.abs(numerator) < 1e-8 && Math.abs(denominator) < 1e-8) {
        return 0;
      }

      const result = numerator / denominator;
      return isFinite(result) ? result : 0;
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

    if (isNaN(combinedF1Change) || !isFinite(combinedF1Change)) {
      console.warn(
        "⚠️  Invalid combined F1 change detected, defaulting to MINIMAL_IMPACT"
      );
      return "MINIMAL_IMPACT";
    }

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

    if (combinedF1Change > this.tolerance) {
      return "REMOVE";
    } else if (combinedF1Change > -this.tolerance) {
      if (positiveMetrics > negativeMetrics) {
        return "REMOVE";
      } else if (negativeMetrics > positiveMetrics) {
        return "KEEP";
      } else {
        return "MINIMAL_IMPACT";
      }
    } else {
      return "KEEP";
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

// Parse command line arguments with proper quote handling and validation
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
      throw new Error("Missing feature name(s) after --feature");
    }

    // Collect all arguments after --feature until the next -- flag
    const features: string[] = [];
    let i = featureIndex + 1;

    try {
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
          // Remove quotes if present and validate
          const feature = args[i].replace(/^["']|["']$/g, "");
          if (feature.trim() === "") {
            throw new Error("Empty feature name detected");
          }
          features.push(feature);
        }
        i++;
      }

      // Validate feature names
      if (features.length === 0) {
        throw new Error("No valid feature names specified after --feature");
      }

      // Check for duplicate features
      const uniqueFeatures = new Set(features);
      if (uniqueFeatures.size !== features.length) {
        const duplicates = features.filter(
          (f, index) => features.indexOf(f) !== index
        );
        throw new Error(
          `Duplicate feature names detected: ${duplicates.join(", ")}`
        );
      }

      if (features.length === 1) {
        return { feature: features[0] };
      } else if (features.length > 1) {
        return { features: features };
      } else {
        throw new Error("No feature names specified after --feature");
      }
    } catch (error) {
      console.error(`❌ Error parsing feature arguments: ${error}`);
      throw error;
    }
  }

  return {};
}

async function main() {
  try {
    const args = parseArguments();
    const optimizer = new GradualFeatureOptimizer();

    if (args.features) {
      await optimizer.testMultipleFeatureRemoval(args.features);
    } else if (args.feature) {
      await optimizer.testFeatureRemoval(args.feature);
    } else {
      await optimizer.runOptimization();
    }
  } catch (error) {
    console.error("❌ Optimization failed:", error);
    throw error;
  }
}

if (require.main === module) {
  main();
}
