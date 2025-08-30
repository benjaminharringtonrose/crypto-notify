#!/usr/bin/env ts-node

/**
 *
 * Usage:
 *   npm run features:gradual                                    # Test all features sequentially
 *   npm run features:gradual -- --feature "featureName"        # Test specific feature
 *
 * Examples:
 *   npm run features:gradual
 *   npm run features:gradual -- --feature "priceChangePct"
 *   npm run features:gradual -- --feature "rsi"
 */

import * as tf from "@tensorflow/tfjs-node";
import * as fs from "fs";
import * as path from "path";
import { DataProcessor } from "../bitcoin/ml/DataProcessor";
import { FeatureDetector } from "../bitcoin/shared/FeatureDetector";
import { TradeModelTrainer } from "../bitcoin/ml/TradeModelTrainer";
import { MODEL_CONFIG, TRAINING_CONFIG } from "../constants";
import { FeatureRegistry } from "../bitcoin/shared/FeatureRegistry";

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
  private outputFilePath: string;

  constructor() {
    this.storeOriginalFunctions();
    this.outputFilePath = path.join(
      process.cwd(),
      "gradual-feature-optimization-results.txt"
    );
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

  private writeToFile(content: string): void {
    try {
      fs.appendFileSync(this.outputFilePath, content + "\n");
    } catch (error) {
      console.warn(
        `⚠️  Failed to write to file ${this.outputFilePath}:`,
        error
      );
    }
  }

  private clearOutputFile(): void {
    try {
      fs.writeFileSync(this.outputFilePath, "");
      console.log(`📄 Results will be saved to: ${this.outputFilePath}`);
    } catch (error) {
      console.warn(
        `⚠️  Failed to clear output file ${this.outputFilePath}:`,
        error
      );
    }
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
    console.log(
      `🔧 Filtering ${originalX.length} sequences with ${featureIndices.length} features`
    );

    if (!originalX || originalX.length === 0) {
      throw new Error("Original data is empty or undefined");
    }

    const filteredX: number[][][] = [];

    for (let i = 0; i < originalX.length; i++) {
      const sequence = originalX[i];
      if (!sequence || !Array.isArray(sequence)) {
        throw new Error(`Invalid sequence at index ${i}: ${sequence}`);
      }

      const filteredSequence: number[][] = [];

      for (let j = 0; j < sequence.length; j++) {
        const timestep = sequence[j];
        if (!timestep || !Array.isArray(timestep)) {
          throw new Error(
            `Invalid timestep at sequence ${i}, timestep ${j}: ${timestep}`
          );
        }

        const filteredTimestep = new Array(featureIndices.length);

        for (let k = 0; k < featureIndices.length; k++) {
          const featureIndex = featureIndices[k];
          if (featureIndex < 0 || featureIndex >= timestep.length) {
            throw new Error(
              `Feature index ${featureIndex} out of bounds for timestep length ${timestep.length}`
            );
          }
          filteredTimestep[k] = timestep[featureIndex];
        }

        filteredSequence.push(filteredTimestep);
      }

      filteredX.push(filteredSequence);
    }

    console.log(`✅ Filtered data has ${filteredX.length} sequences`);
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
      this.clearOutputFile();

      const header = [
        "🚀 Starting Gradual Feature Optimization",
        `📊 Testing ${this.featureNames.length} features one at a time`,
        `🎯 Performance tolerance: ${this.tolerance * 100}%`,
        `🎲 Using seed: ${this.seed}`,
        "=".repeat(80),
      ].join("\n");

      console.log(header);
      this.writeToFile(header);

      // Initialize deterministic random behavior
      this.initializeDeterministicRandom();

      // Validate feature registry consistency
      await this.validateFeatureRegistry();

      // Initialize feature detection
      const initMsg = "\n🔧 Initializing feature detection...";
      console.log(initMsg);
      this.writeToFile(initMsg);
      await FeatureDetector.detectFeatureCount();

      // Establish baseline performance
      const baselineMsg =
        "\n📈 Step 0: Establishing Baseline Performance (All Features)";
      console.log(baselineMsg);
      this.writeToFile(baselineMsg);
      const baselinePerformance = await this.trainAndEvaluateModel(
        this.featureNames,
        0
      );
      const baselineResult = `✅ Baseline: ${baselinePerformance.validationAccuracy.toFixed(
        4
      )} accuracy, ${baselinePerformance.combinedF1.toFixed(4)} combined F1`;
      console.log(baselineResult);
      this.writeToFile(baselineResult);

      const originalBaselinePerformance =
        this.deepCopyPerformanceMetrics(baselinePerformance);

      console.log(`🔍 Testing ${this.featureNames.length} features`);

      for (let i = 0; i < this.featureNames.length; i++) {
        const featureName = this.featureNames[i];
        const stepMsg = `\n🔍 Step ${
          i + 1
        }: Testing removal of "${featureName}" (index ${i})`;
        console.log(stepMsg);
        this.writeToFile(stepMsg);

        const modifiedFeatures = this.featureNames.filter(
          (name) => name !== featureName
        );

        const modifiedPerformance = await this.trainAndEvaluateModel(
          modifiedFeatures,
          i + 1
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
      this.clearOutputFile();

      const header = [
        "🚀 Testing Individual Feature Removal",
        `🎯 Testing removal of: "${featureName}"`,
        `🎯 Performance tolerance: ${this.tolerance * 100}%`,
        `🎲 Using seed: ${this.seed}`,
        "=".repeat(80),
      ].join("\n");

      console.log(header);
      this.writeToFile(header);

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
      const initMsg = "\n🔧 Initializing feature detection...";
      console.log(initMsg);
      this.writeToFile(initMsg);
      await FeatureDetector.detectFeatureCount();

      // Establish baseline performance
      const baselineMsg =
        "\n📈 Step 1: Establishing Baseline Performance (All Features)";
      console.log(baselineMsg);
      this.writeToFile(baselineMsg);
      const baselinePerformance = await this.trainAndEvaluateModel(
        this.featureNames,
        1
      );
      const baselineResult = `✅ Baseline: ${baselinePerformance.validationAccuracy.toFixed(
        4
      )} accuracy, ${baselinePerformance.combinedF1.toFixed(4)} combined F1`;
      console.log(baselineResult);
      this.writeToFile(baselineResult);

      // Test without the specified feature
      const testMsg = `\n🔍 Step 2: Testing Performance Without "${featureName}"`;
      console.log(testMsg);
      this.writeToFile(testMsg);
      const modifiedFeatures = this.featureNames.filter(
        (name) => name !== featureName
      );
      const modifiedPerformance = await this.trainAndEvaluateModel(
        modifiedFeatures,
        2
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
      this.clearOutputFile();

      const header = [
        "🚀 Testing Multiple Feature Removal",
        `🎯 Features: ${featureNames.map((f) => `"${f}"`).join(", ")}`,
        `🎯 Performance tolerance: ${this.tolerance * 100}%`,
        `🎲 Using seed: ${this.seed}`,
        "=".repeat(80),
      ].join("\n");

      console.log(header);
      this.writeToFile(header);

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
        this.featureNames,
        1
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
        modifiedFeatures,
        2
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
    featureArray: string[],
    step: number
  ): Promise<PerformanceMetrics> {
    const FIXED_SEED = this.seed;

    console.log(`🎲 Using fixed seed: ${FIXED_SEED} for consistent training`);

    try {
      // Reset training state before each training session
      this.resetTrainingState();

      // Create a completely fresh trainer instance to avoid state contamination
      let trainer: TradeModelTrainer;
      try {
        trainer = new TradeModelTrainer(FIXED_SEED);
        console.log("🔄 Created fresh TradeModelTrainer instance");
      } catch (error) {
        console.error("❌ Failed to create trainer:", error);
        throw new Error(`Failed to create trainer: ${error}`);
      }

      trainer.setCurrentFeatureName(featureArray.join(", "));

      // Create a fresh DataProcessor for each test to avoid state contamination
      const originalDataProcessor = new DataProcessor(
        {
          timesteps: MODEL_CONFIG.TIMESTEPS,
          epochs: TRAINING_CONFIG.EPOCHS,
          batchSize: TRAINING_CONFIG.BATCH_SIZE,
          initialLearningRate: TRAINING_CONFIG.INITIAL_LEARNING_RATE,
        },
        TRAINING_CONFIG.START_DAYS_AGO
      );

      console.log("🔄 Created fresh DataProcessor instance");

      // Force a complete reset of the feature calculation system
      try {
        // Clear any cached feature data using type assertion
        const processor = originalDataProcessor as any;
        if (processor.featureCache) {
          processor.featureCache = null;
        }
        if (processor.processedData) {
          processor.processedData = null;
        }
        console.log("🧹 Cleared DataProcessor cache");
      } catch (error) {
        console.warn("⚠️  Could not clear DataProcessor cache:", error);
      }

      const { X: originalX, y } = await originalDataProcessor.prepareData();

      // Validate that data was loaded successfully
      if (!originalX || !Array.isArray(originalX) || originalX.length === 0) {
        throw new Error(
          `Failed to load training data: originalX is ${
            originalX ? "empty" : "undefined"
          }`
        );
      }

      if (
        !originalX[0] ||
        !Array.isArray(originalX[0]) ||
        originalX[0].length === 0
      ) {
        throw new Error(
          `Failed to load training data: originalX[0] is ${
            originalX[0] ? "empty" : "undefined"
          }`
        );
      }

      if (!originalX[0][0] || !Array.isArray(originalX[0][0])) {
        throw new Error(
          `Failed to load training data: originalX[0][0] is ${
            originalX[0][0] ? "not an array" : "undefined"
          }`
        );
      }

      // Additional validation to ensure data integrity
      console.log(
        `📊 Data validation: ${originalX.length} sequences, ${originalX[0].length} timesteps, ${originalX[0][0].length} features`
      );

      // Check if y data is also valid
      if (!y || !Array.isArray(y) || y.length === 0) {
        throw new Error(
          `Failed to load training data: y is ${y ? "empty" : "undefined"}`
        );
      }

      if (originalX.length !== y.length) {
        throw new Error(
          `Data mismatch: X has ${originalX.length} samples, y has ${y.length} samples`
        );
      }

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

      console.log(`🔍 Feature indices: [${featureIndices.join(", ")}]`);
      console.log(`📊 Original feature count: ${originalX[0][0].length}`);

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

      // Validate that all indices are valid
      for (const index of featureIndices) {
        if (index < 0 || index >= originalX[0][0].length) {
          throw new Error(
            `Invalid feature index: ${index} (valid range: 0-${
              originalX[0][0].length - 1
            })`
          );
        }
      }

      console.log(`🔍 Using feature indices: [${featureIndices.join(", ")}]`);

      const filteredX = this.filterFeaturesEfficiently(
        originalX,
        featureIndices
      );

      console.log(`✅ Filtered data has ${filteredX[0][0].length} features`);

      // Override the FeatureDetector to return the correct feature count
      const originalGetFeatureCount = FeatureDetector.getFeatureCount;
      FeatureDetector.getFeatureCount = () => featureArray.length;

      // Override the trainer's data processor prepareData method
      const originalPrepareData = trainer["dataProcessor"].prepareData;
      trainer["dataProcessor"].prepareData = async () => ({ X: filteredX, y });

      console.log(
        `✅ Overrode data processor with ${featureArray.length} features`
      );
      console.log(
        `📊 Training data: ${filteredX.length} sequences, ${filteredX[0].length} timesteps, ${filteredX[0][0].length} features`
      );

      // Train the model using the same process as the main trainer
      await trainer.train();

      // Restore original methods
      trainer["dataProcessor"].prepareData = originalPrepareData;
      FeatureDetector.getFeatureCount = originalGetFeatureCount;

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
      console.log("🧹 Cleaning up TensorFlow memory...");

      // Dispose all tensors in memory
      tf.tidy(() => {
        // Create and immediately dispose a dummy tensor to trigger cleanup
        const dummyTensor = tf.zeros([1, 1]);
        dummyTensor.dispose();
      });

      // Dispose all variables
      tf.disposeVariables();

      // Clear the backend memory
      const backend = tf.getBackend();
      if (backend) {
        try {
          // Force garbage collection on the backend
          tf.engine().startScope();
          tf.engine().endScope();
        } catch (error) {
          console.warn("⚠️  Backend cleanup warning:", error);
        }
      }

      // Force garbage collection if available
      if (global.gc) {
        global.gc();
      }

      console.log("✅ TensorFlow memory cleanup completed");
    } catch (error) {
      console.error("❌ Error cleaning up TensorFlow memory:", error);
    }
  }

  private resetTrainingState(): void {
    try {
      console.log("🔄 Resetting training state...");

      // Clean up TensorFlow memory
      this.cleanupTensorFlowMemory();

      // Reset the random state to ensure deterministic behavior
      this.randomCounter = 0;
      this.lcgState = (this.seed || 42) >>> 0;

      // Re-initialize deterministic random behavior
      this.initializeDeterministicRandom();

      // Force garbage collection if available
      if (global.gc) {
        global.gc();
      }

      // Additional TensorFlow backend reset
      try {
        const backend = tf.getBackend();
        if (backend === "tensorflow") {
          // Force backend reset
          tf.engine().reset();
          console.log("🔄 Reset TensorFlow backend");
        }
      } catch (error) {
        console.warn("⚠️  Could not reset TensorFlow backend:", error);
      }

      console.log("✅ Training state reset completed");
    } catch (error) {
      console.error("❌ Error resetting training state:", error);
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

    const stepResults = [
      `   📊 Modified Performance:`,
      `      Accuracy: ${modifiedPerformance.validationAccuracy.toFixed(4)}`,
      `      Buy F1: ${modifiedPerformance.buyF1.toFixed(4)}`,
      `      Sell F1: ${modifiedPerformance.sellF1.toFixed(4)}`,
      `      Combined F1: ${modifiedPerformance.combinedF1.toFixed(4)}`,
      `   📈 Performance Change: ${(combinedF1 * 100).toFixed(2)}%`,
      `   🎯 Decision: ${decision}`,
      `   💡 Reason: ${step.reason}`,
    ].join("\n");

    console.log(stepResults);
    this.writeToFile(stepResults);
  }

  private generateFinalReport(): void {
    const reportHeader =
      "\n" +
      "=".repeat(80) +
      "\n" +
      "📋 GRADUAL FEATURE OPTIMIZATION FINAL REPORT" +
      "\n" +
      "=".repeat(80);
    console.log(reportHeader);
    this.writeToFile(reportHeader);

    const removedFeatures = this.optimizationSteps.filter(
      (s) => s.decision === "REMOVE"
    );
    const keptFeatures = this.optimizationSteps.filter(
      (s) => s.decision === "KEEP"
    );
    const minimalImpactFeatures = this.optimizationSteps.filter(
      (s) => s.decision === "MINIMAL_IMPACT"
    );

    const summary = [
      `\n📊 SUMMARY:`,
      `   Original Features: ${this.featureNames.length}`,
      `   Features to Remove: ${removedFeatures.length}`,
      `   Features to Keep: ${keptFeatures.length}`,
      `   Minimal Impact: ${minimalImpactFeatures.length}`,
      `   Final Feature Count: ${
        this.featureNames.length - removedFeatures.length
      }`,
    ].join("\n");
    console.log(summary);
    this.writeToFile(summary);

    if (removedFeatures.length > 0) {
      const removeSection = [
        `\n✅ FEATURES TO REMOVE (${removedFeatures.length}):`,
      ];
      removedFeatures.forEach((step) => {
        removeSection.push(
          `   - ${step.featureName} (index ${step.featureIndex}): ${step.reason}`
        );
      });
      const removeText = removeSection.join("\n");
      console.log(removeText);
      this.writeToFile(removeText);
    }

    if (keptFeatures.length > 0) {
      const keepSection = [`\n🔒 FEATURES TO KEEP (${keptFeatures.length}):`];
      keptFeatures.forEach((step) => {
        keepSection.push(
          `   - ${step.featureName} (index ${step.featureIndex}): ${step.reason}`
        );
      });
      const keepText = keepSection.join("\n");
      console.log(keepText);
      this.writeToFile(keepText);
    }

    if (minimalImpactFeatures.length > 0) {
      const minimalSection = [
        `\n⚠️  MINIMAL IMPACT FEATURES (${minimalImpactFeatures.length}):`,
      ];
      minimalImpactFeatures.forEach((step) => {
        minimalSection.push(
          `   - ${step.featureName} (index ${step.featureIndex}): ${step.reason}`
        );
      });
      const minimalText = minimalSection.join("\n");
      console.log(minimalText);
      this.writeToFile(minimalText);
    }

    // Generate optimized feature array
    const optimizedFeatureNames = this.featureNames.filter((_, index) => {
      const step = this.optimizationSteps.find((s) => s.featureIndex === index);
      return step && step.decision !== "REMOVE";
    });

    const optimizedArray = [
      `\n🎯 OPTIMIZED FEATURE ARRAY:`,
      `const optimizedFeatures = [`,
    ];
    optimizedFeatureNames.forEach((name, index) => {
      optimizedArray.push(`  "${name}", // ${index + 1}`);
    });
    optimizedArray.push(`];`);
    const optimizedText = optimizedArray.join("\n");
    console.log(optimizedText);
    this.writeToFile(optimizedText);

    const performanceImpact = [`\n📈 PERFORMANCE IMPACT:`];
    const baselineStep = this.optimizationSteps[0];
    const finalStep = this.optimizationSteps[this.optimizationSteps.length - 1];
    if (baselineStep && finalStep) {
      const improvement =
        finalStep.modifiedPerformance.combinedF1 -
        baselineStep.originalPerformance.combinedF1;
      performanceImpact.push(
        `   Combined F1 Improvement: ${(improvement * 100).toFixed(2)}%`
      );
      performanceImpact.push(
        `   Final Combined F1: ${finalStep.modifiedPerformance.combinedF1.toFixed(
          4
        )}`
      );
    }
    const performanceText = performanceImpact.join("\n");
    console.log(performanceText);
    this.writeToFile(performanceText);
  }

  private logResults(
    featureName: string,
    baselinePerformance: PerformanceMetrics,
    modifiedPerformance: PerformanceMetrics,
    performanceChange: PerformanceChange,
    decision: string
  ): void {
    const resultsHeader =
      "\n" +
      "=".repeat(80) +
      "\n" +
      "📋 FEATURE REMOVAL TEST RESULTS" +
      "\n" +
      "=".repeat(80);
    console.log(resultsHeader);
    this.writeToFile(resultsHeader);

    const featureInfo = [
      `\n🎯 Feature Tested: "${featureName}"`,
      `📊 Feature Index: ${this.featureNames.indexOf(featureName) + 1}/${
        this.featureNames.length
      }`,
    ].join("\n");
    console.log(featureInfo);
    this.writeToFile(featureInfo);

    const performanceComparison = [
      `\n📈 PERFORMANCE COMPARISON:`,
      `   Metric              | Baseline | Modified | Change`,
      `   --------------------|----------|----------|--------`,
      `   Validation Accuracy | ${baselinePerformance.validationAccuracy.toFixed(
        4
      )} | ${modifiedPerformance.validationAccuracy.toFixed(4)} | ${(
        performanceChange.validationAccuracy * 100
      ).toFixed(2)}%`,
      `   Buy F1 Score        | ${baselinePerformance.buyF1.toFixed(
        4
      )} | ${modifiedPerformance.buyF1.toFixed(4)} | ${(
        performanceChange.buyF1 * 100
      ).toFixed(2)}%`,
      `   Sell F1 Score       | ${baselinePerformance.sellF1.toFixed(
        4
      )} | ${modifiedPerformance.sellF1.toFixed(4)} | ${(
        performanceChange.sellF1 * 100
      ).toFixed(2)}%`,
      `   Combined F1 Score   | ${baselinePerformance.combinedF1.toFixed(
        4
      )} | ${modifiedPerformance.combinedF1.toFixed(4)} | ${(
        performanceChange.combinedF1 * 100
      ).toFixed(2)}%`,
      `   Balanced Accuracy   | ${baselinePerformance.balancedAccuracy.toFixed(
        4
      )} | ${modifiedPerformance.balancedAccuracy.toFixed(4)} | ${(
        performanceChange.balancedAccuracy * 100
      ).toFixed(2)}%`,
      `   Matthews Correlation| ${baselinePerformance.matthewsCorrelation.toFixed(
        4
      )} | ${modifiedPerformance.matthewsCorrelation.toFixed(4)} | ${(
        performanceChange.matthewsCorrelation * 100
      ).toFixed(2)}%`,
    ].join("\n");
    console.log(performanceComparison);
    this.writeToFile(performanceComparison);

    const decisionText = `\n🎯 DECISION: ${decision}`;
    console.log(decisionText);
    this.writeToFile(decisionText);

    const combinedF1Change = performanceChange.combinedF1;
    let recommendation = "";
    if (decision === "REMOVE") {
      recommendation = `✅ RECOMMENDATION: Remove "${featureName}" - Performance improved by ${(
        combinedF1Change * 100
      ).toFixed(2)}%`;
    } else if (decision === "KEEP") {
      recommendation = `🔒 RECOMMENDATION: Keep "${featureName}" - Performance degraded by ${(
        Math.abs(combinedF1Change) * 100
      ).toFixed(2)}%`;
    } else {
      recommendation = `⚠️  RECOMMENDATION: "${featureName}" has minimal impact (within ${(
        this.tolerance * 100
      ).toFixed(1)}% tolerance)`;
    }
    console.log(recommendation);
    this.writeToFile(recommendation);

    const featureCount = [
      `\n📊 FEATURE COUNT:`,
      `   Original: ${this.featureNames.length} features`,
      `   Modified: ${this.featureNames.length - 1} features`,
      `   Reduction: 1 feature (${(
        (1 / this.featureNames.length) *
        100
      ).toFixed(1)}%)`,
    ].join("\n");
    console.log(featureCount);
    this.writeToFile(featureCount);
  }

  private logMultipleFeatureResults(
    featureNames: string[],
    baselinePerformance: PerformanceMetrics,
    modifiedPerformance: PerformanceMetrics,
    performanceChange: PerformanceChange,
    decision: string
  ): void {
    const resultsHeader =
      "\n" +
      "=".repeat(80) +
      "\n" +
      "📋 MULTIPLE FEATURE REMOVAL TEST RESULTS" +
      "\n" +
      "=".repeat(80);
    console.log(resultsHeader);
    this.writeToFile(resultsHeader);

    const featureInfo = [
      `\n🎯 Features Tested: ${featureNames.map((f) => `"${f}"`).join(", ")}`,
      `📊 Features Count: ${featureNames.length} features removed from ${this.featureNames.length} total`,
    ].join("\n");
    console.log(featureInfo);
    this.writeToFile(featureInfo);

    const performanceComparison = [
      `\n📈 PERFORMANCE COMPARISON:`,
      `   Metric              | Baseline | Modified | Change`,
      `   --------------------|----------|----------|--------`,
      `   Validation Accuracy | ${baselinePerformance.validationAccuracy.toFixed(
        4
      )} | ${modifiedPerformance.validationAccuracy.toFixed(4)} | ${(
        performanceChange.validationAccuracy * 100
      ).toFixed(2)}%`,
      `   Buy F1 Score        | ${baselinePerformance.buyF1.toFixed(
        4
      )} | ${modifiedPerformance.buyF1.toFixed(4)} | ${(
        performanceChange.buyF1 * 100
      ).toFixed(2)}%`,
      `   Sell F1 Score       | ${baselinePerformance.sellF1.toFixed(
        4
      )} | ${modifiedPerformance.sellF1.toFixed(4)} | ${(
        performanceChange.sellF1 * 100
      ).toFixed(2)}%`,
      `   Combined F1 Score   | ${baselinePerformance.combinedF1.toFixed(
        4
      )} | ${modifiedPerformance.combinedF1.toFixed(4)} | ${(
        performanceChange.combinedF1 * 100
      ).toFixed(2)}%`,
      `   Balanced Accuracy   | ${baselinePerformance.balancedAccuracy.toFixed(
        4
      )} | ${modifiedPerformance.balancedAccuracy.toFixed(4)} | ${(
        performanceChange.balancedAccuracy * 100
      ).toFixed(2)}%`,
      `   Matthews Correlation| ${baselinePerformance.matthewsCorrelation.toFixed(
        4
      )} | ${modifiedPerformance.matthewsCorrelation.toFixed(4)} | ${(
        performanceChange.matthewsCorrelation * 100
      ).toFixed(2)}%`,
    ].join("\n");
    console.log(performanceComparison);
    this.writeToFile(performanceComparison);

    const decisionText = `\n🎯 DECISION: ${decision}`;
    console.log(decisionText);
    this.writeToFile(decisionText);

    const combinedF1Change = performanceChange.combinedF1;
    let recommendation = "";
    if (decision === "REMOVE") {
      recommendation = `✅ RECOMMENDATION: Remove ${
        featureNames.length
      } features - Performance improved by ${(combinedF1Change * 100).toFixed(
        2
      )}%`;
    } else if (decision === "KEEP") {
      recommendation = `🔒 RECOMMENDATION: Keep ${
        featureNames.length
      } features - Performance degraded by ${(
        Math.abs(combinedF1Change) * 100
      ).toFixed(2)}%`;
    } else {
      recommendation = `⚠️  RECOMMENDATION: ${
        featureNames.length
      } features have minimal impact (within ${(this.tolerance * 100).toFixed(
        1
      )}% tolerance)`;
    }
    console.log(recommendation);
    this.writeToFile(recommendation);

    const featureCount = [
      `\n📊 FEATURE COUNT:`,
      `   Original: ${this.featureNames.length} features`,
      `   Modified: ${this.featureNames.length - featureNames.length} features`,
      `   Reduction: ${featureNames.length} features (${(
        (featureNames.length / this.featureNames.length) *
        100
      ).toFixed(1)}%)`,
    ].join("\n");
    console.log(featureCount);
    this.writeToFile(featureCount);
  }
}

// Parse command line arguments with proper quote handling and validation
function parseArguments(): {
  feature?: string;
  features?: string[];
} {
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
        return {
          feature: features[0],
        };
      } else if (features.length > 1) {
        return {
          features,
        };
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
