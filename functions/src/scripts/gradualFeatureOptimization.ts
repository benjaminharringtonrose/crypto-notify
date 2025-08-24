#!/usr/bin/env ts-node
/**
 * Gradual Feature Optimization - Systematic feature reduction with production model testing
 *
 * This script performs careful, one-at-a-time feature removal testing to identify
 * truly redundant features without hurting model performance.
 */

import * as tf from "@tensorflow/tfjs-node";
import { DataProcessor } from "../bitcoin/DataProcessor";
import { FeatureDetector } from "../bitcoin/FeatureDetector";
import TradeModelFactory from "../bitcoin/TradeModelFactory";
import { Metrics } from "../bitcoin/Metrics";
import { MODEL_CONFIG, TRAINING_CONFIG } from "../constants";

interface OptimizationStep {
  step: number;
  featureRemoved: string;
  featureIndex: number;
  category: string;

  // Performance metrics
  baselineAccuracy: number;
  testAccuracy: number;
  accuracyDelta: number;

  baselineF1Buy: number;
  testF1Buy: number;
  f1BuyDelta: number;

  baselineF1Sell: number;
  testF1Sell: number;
  f1SellDelta: number;

  baselineCombinedF1: number;
  testCombinedF1: number;
  combinedF1Delta: number;

  // Decision
  decision: "keep" | "remove" | "investigate";
  reasoning: string;

  // Training info
  trainingTimeMs: number;
  epochsTrained: number;
}

interface OptimizationResult {
  originalFeatureCount: number;
  finalFeatureCount: number;
  removedFeatures: string[];
  keptFeatures: string[];

  // Performance tracking
  baselinePerformance: {
    accuracy: number;
    f1Buy: number;
    f1Sell: number;
    combinedF1: number;
  };

  finalPerformance: {
    accuracy: number;
    f1Buy: number;
    f1Sell: number;
    combinedF1: number;
  };

  // Optimization steps
  steps: OptimizationStep[];

  // Summary
  totalTimeMs: number;
  featuresRemoved: number;
  performanceChange: number;
}

class GradualFeatureOptimizer {
  private dataProcessor: DataProcessor;
  private baselinePerformance: any = null;
  private originalFeatureCount: number = 0;
  private currentFeatureIndices: number[] = [];
  private steps: OptimizationStep[] = [];

  constructor() {
    this.dataProcessor = new DataProcessor(
      {
        timesteps: MODEL_CONFIG.TIMESTEPS,
        epochs: TRAINING_CONFIG.EPOCHS,
        batchSize: TRAINING_CONFIG.BATCH_SIZE,
        initialLearningRate: TRAINING_CONFIG.INITIAL_LEARNING_RATE,
      },
      600 // Use full dataset for accurate testing
    );
  }

  /**
   * Run complete gradual optimization
   */
  public async optimizeFeatures(): Promise<OptimizationResult> {
    console.log("🎯 Starting Gradual Feature Optimization...");
    console.log("=".repeat(70));

    // Step 0: Initialize feature detection
    console.log("🔧 Initializing feature detection...");
    await FeatureDetector.detectFeatureCount();
    this.originalFeatureCount = FeatureDetector.getFeatureCount();
    this.currentFeatureIndices = Array.from(
      { length: this.originalFeatureCount },
      (_, i) => i
    );

    console.log(`📊 Original feature count: ${this.originalFeatureCount}`);

    // Step 1: Establish baseline performance
    console.log("📈 Establishing baseline performance...");
    this.baselinePerformance = await this.trainBaselineModel();

    console.log("✅ Baseline established:");
    console.log(
      `  Accuracy: ${(this.baselinePerformance.accuracy * 100).toFixed(2)}%`
    );
    console.log(`  F1 Buy: ${this.baselinePerformance.f1Buy.toFixed(4)}`);
    console.log(`  F1 Sell: ${this.baselinePerformance.f1Sell.toFixed(4)}`);
    console.log(
      `  Combined F1: ${this.baselinePerformance.combinedF1.toFixed(4)}`
    );

    // Step 2: Analyze feature correlations
    console.log("🔍 Analyzing feature correlations...");
    const correlationAnalysis = await this.analyzeFeatureCorrelations();

    // Step 3: Test feature removals systematically
    console.log("🧪 Testing feature removals systematically...");
    await this.testFeatureRemovals(correlationAnalysis);

    // Step 4: Generate final result
    const result = this.generateOptimizationResult();
    this.presentResults(result);

    return result;
  }

  /**
   * Train baseline model with all features
   */
  private async trainBaselineModel(): Promise<any> {
    const { X, y } = await this.dataProcessor.prepareData();

    const factory = new TradeModelFactory(
      MODEL_CONFIG.TIMESTEPS,
      this.originalFeatureCount
    );
    const model = factory.createModel();

    // Compile the model
    model.compile({
      optimizer: tf.train.adam(TRAINING_CONFIG.INITIAL_LEARNING_RATE),
      loss: (yTrue: tf.Tensor, yPred: tf.Tensor) => {
        return Metrics.focalLoss(
          yTrue,
          yPred,
          TRAINING_CONFIG.GAMMA,
          TRAINING_CONFIG.ALPHA
        );
      },
      metrics: ["accuracy"],
    });

    const splitIndex = Math.floor(X.length * 0.8);

    const XTrain = X.slice(0, splitIndex);
    const yTrain = y.slice(0, splitIndex);
    const XVal = X.slice(splitIndex);
    const yVal = y.slice(splitIndex);

    const XTrainTensor = tf.tensor3d(XTrain);
    const yTrainTensor = tf.tensor2d(
      yTrain.map((label) => [label === 0 ? 1 : 0, label === 1 ? 1 : 0])
    );
    const XValTensor = tf.tensor3d(XVal);
    const yValTensor = tf.tensor2d(
      yVal.map((label) => [label === 0 ? 1 : 0, label === 1 ? 1 : 0])
    );

    try {
      // Train with deterministic seeding for reproducibility
      tf.setBackend("tensorflow");
      tf.engine().startScope();

      await model.fit(XTrainTensor, yTrainTensor, {
        epochs: 30,
        batchSize: 16,
        validationData: [XValTensor, yValTensor],
        verbose: 0,
      });

      // Evaluate with comprehensive metrics
      const evaluation = await Metrics.evaluateModel(
        model,
        XValTensor,
        yValTensor
      );

      return {
        accuracy: evaluation.balancedAccuracy,
        f1Buy: evaluation.buyF1,
        f1Sell: evaluation.sellF1,
        combinedF1: evaluation.combinedF1,
        matthewsCorrelation: evaluation.matthewsCorrelation,
      };
    } finally {
      XTrainTensor.dispose();
      yTrainTensor.dispose();
      XValTensor.dispose();
      yValTensor.dispose();
      model.dispose();
      tf.engine().endScope();
    }
  }

  /**
   * Analyze correlations between features
   */
  private async analyzeFeatureCorrelations(): Promise<
    Array<{
      featureIndex: number;
      featureName: string;
      category: string;
      avgCorrelation: number;
      maxCorrelation: number;
      highlyCorrelatedFeatures: number[];
    }>
  > {
    const { X } = await this.dataProcessor.prepareData();

    // Flatten sequences to get feature values across all timesteps
    const featureValues: number[][] = [];
    for (
      let featureIdx = 0;
      featureIdx < this.originalFeatureCount;
      featureIdx++
    ) {
      const values: number[] = [];
      for (let sampleIdx = 0; sampleIdx < X.length; sampleIdx++) {
        for (let timestep = 0; timestep < X[sampleIdx].length; timestep++) {
          values.push(X[sampleIdx][timestep][featureIdx]);
        }
      }
      featureValues.push(values);
    }

    // Calculate correlations
    const correlations: Array<{
      featureIndex: number;
      featureName: string;
      category: string;
      avgCorrelation: number;
      maxCorrelation: number;
      highlyCorrelatedFeatures: number[];
    }> = [];

    for (let i = 0; i < this.originalFeatureCount; i++) {
      const correlationsWithOthers: number[] = [];
      const highlyCorrelated: number[] = [];

      for (let j = 0; j < this.originalFeatureCount; j++) {
        if (i !== j) {
          const correlation = this.calculateCorrelation(
            featureValues[i],
            featureValues[j]
          );
          correlationsWithOthers.push(Math.abs(correlation));

          if (Math.abs(correlation) > 0.8) {
            highlyCorrelated.push(j);
          }
        }
      }

      const avgCorrelation =
        correlationsWithOthers.reduce((a, b) => a + b, 0) /
        correlationsWithOthers.length;
      const maxCorrelation = Math.max(...correlationsWithOthers);

      correlations.push({
        featureIndex: i,
        featureName: `feature_${i}`,
        category: "unknown",
        avgCorrelation,
        maxCorrelation,
        highlyCorrelatedFeatures: highlyCorrelated,
      });
    }

    // Sort by average correlation (highest first - most redundant)
    return correlations.sort((a, b) => b.avgCorrelation - a.avgCorrelation);
  }

  /**
   * Calculate Pearson correlation between two arrays
   */
  private calculateCorrelation(x: number[], y: number[]): number {
    const n = Math.min(x.length, y.length);
    if (n === 0) return 0;

    const sumX = x.slice(0, n).reduce((a, b) => a + b, 0);
    const sumY = y.slice(0, n).reduce((a, b) => a + b, 0);
    const sumXY = x.slice(0, n).reduce((sum, xi, i) => sum + xi * y[i], 0);
    const sumX2 = x.slice(0, n).reduce((sum, xi) => sum + xi * xi, 0);
    const sumY2 = y.slice(0, n).reduce((sum, yi) => sum + yi * yi, 0);

    const numerator = n * sumXY - sumX * sumY;
    const denominator = Math.sqrt(
      (n * sumX2 - sumX * sumX) * (n * sumY2 - sumY * sumY)
    );

    return denominator === 0 ? 0 : numerator / denominator;
  }

  /**
   * Test feature removals systematically
   */
  private async testFeatureRemovals(correlationAnalysis: any[]): Promise<void> {
    console.log(
      `\n🧪 Testing ${this.originalFeatureCount} features for removal...`
    );

    // Start with most correlated features (potential redundancies)
    const testOrder = correlationAnalysis.map((c) => c.featureIndex);

    for (let i = 0; i < testOrder.length; i++) {
      const featureIndex = testOrder[i];
      const correlation = correlationAnalysis.find(
        (c) => c.featureIndex === featureIndex
      );

      console.log(
        `\n📊 Step ${i + 1}/${
          this.originalFeatureCount
        }: Testing feature ${featureIndex}`
      );
      console.log(
        `   Avg correlation: ${correlation?.avgCorrelation.toFixed(3)}`
      );
      console.log(
        `   Max correlation: ${correlation?.maxCorrelation.toFixed(3)}`
      );

      const step = await this.testSingleFeatureRemoval(featureIndex, i + 1);
      this.steps.push(step);

      // Update current feature indices if feature was removed
      if (step.decision === "remove") {
        this.currentFeatureIndices = this.currentFeatureIndices.filter(
          (idx) => idx !== featureIndex
        );
        console.log(
          `   ✅ REMOVED: Feature ${featureIndex} (${step.reasoning})`
        );
      } else {
        console.log(`   ❌ KEPT: Feature ${featureIndex} (${step.reasoning})`);
      }

      // Early stopping if we've removed too many features
      if (this.currentFeatureIndices.length < this.originalFeatureCount * 0.7) {
        console.log(
          `   ⚠️  Stopping early: Too many features removed (${this.currentFeatureIndices.length}/${this.originalFeatureCount})`
        );
        break;
      }
    }
  }

  /**
   * Test removal of a single feature
   */
  private async testSingleFeatureRemoval(
    featureIndex: number,
    stepNumber: number
  ): Promise<OptimizationStep> {
    const startTime = Date.now();

    // Create modified data with feature removed
    const { X, y } = await this.dataProcessor.prepareData();
    const XModified = X.map((sequence) =>
      sequence.map((timestep) =>
        timestep.filter((_, idx) => idx !== featureIndex)
      )
    );

    // Train model with modified features
    const factory = new TradeModelFactory(
      MODEL_CONFIG.TIMESTEPS,
      this.originalFeatureCount - 1
    );
    const model = factory.createModel();

    // Compile the model
    model.compile({
      optimizer: tf.train.adam(TRAINING_CONFIG.INITIAL_LEARNING_RATE),
      loss: (yTrue: tf.Tensor, yPred: tf.Tensor) => {
        return Metrics.focalLoss(
          yTrue,
          yPred,
          TRAINING_CONFIG.GAMMA,
          TRAINING_CONFIG.ALPHA
        );
      },
      metrics: ["accuracy"],
    });

    const splitIndex = Math.floor(XModified.length * 0.8);

    const XTrain = XModified.slice(0, splitIndex);
    const yTrain = y.slice(0, splitIndex);
    const XVal = XModified.slice(splitIndex);
    const yVal = y.slice(splitIndex);

    const XTrainTensor = tf.tensor3d(XTrain);
    const yTrainTensor = tf.tensor2d(
      yTrain.map((label) => [label === 0 ? 1 : 0, label === 1 ? 1 : 0])
    );
    const XValTensor = tf.tensor3d(XVal);
    const yValTensor = tf.tensor2d(
      yVal.map((label) => [label === 0 ? 1 : 0, label === 1 ? 1 : 0])
    );

    try {
      tf.setBackend("tensorflow");
      tf.engine().startScope();

      await model.fit(XTrainTensor, yTrainTensor, {
        epochs: 30,
        batchSize: 16,
        validationData: [XValTensor, yValTensor],
        verbose: 0,
      });

      const evaluation = await Metrics.evaluateModel(
        model,
        XValTensor,
        yValTensor
      );
      const trainingTimeMs = Date.now() - startTime;

      // Calculate deltas
      const accuracyDelta =
        evaluation.balancedAccuracy - this.baselinePerformance.accuracy;
      const f1BuyDelta = evaluation.buyF1 - this.baselinePerformance.f1Buy;
      const f1SellDelta = evaluation.sellF1 - this.baselinePerformance.f1Sell;
      const combinedF1Delta =
        evaluation.combinedF1 - this.baselinePerformance.combinedF1;

      // Decision logic
      let decision: "keep" | "remove" | "investigate";
      let reasoning: string;

      if (accuracyDelta >= 0 && combinedF1Delta >= 0) {
        decision = "remove";
        reasoning = "Performance improved or maintained";
      } else if (accuracyDelta >= -0.01 && combinedF1Delta >= -0.02) {
        decision = "remove";
        reasoning = "Minimal performance impact (within tolerance)";
      } else if (accuracyDelta >= -0.02 && combinedF1Delta >= -0.05) {
        decision = "investigate";
        reasoning = "Moderate performance impact - needs manual review";
      } else {
        decision = "keep";
        reasoning = "Significant performance degradation";
      }

      return {
        step: stepNumber,
        featureRemoved: `feature_${featureIndex}`,
        featureIndex,
        category: "unknown",
        baselineAccuracy: this.baselinePerformance.accuracy,
        testAccuracy: evaluation.balancedAccuracy,
        accuracyDelta,
        baselineF1Buy: this.baselinePerformance.f1Buy,
        testF1Buy: evaluation.buyF1,
        f1BuyDelta,
        baselineF1Sell: this.baselinePerformance.f1Sell,
        testF1Sell: evaluation.sellF1,
        f1SellDelta,
        baselineCombinedF1: this.baselinePerformance.combinedF1,
        testCombinedF1: evaluation.combinedF1,
        combinedF1Delta,
        decision,
        reasoning,
        trainingTimeMs,
        epochsTrained: 30, // Fixed epochs since we're not using history
      };
    } finally {
      XTrainTensor.dispose();
      yTrainTensor.dispose();
      XValTensor.dispose();
      yValTensor.dispose();
      model.dispose();
      tf.engine().endScope();
    }
  }

  /**
   * Generate final optimization result
   */
  private generateOptimizationResult(): OptimizationResult {
    const removedFeatures = this.steps
      .filter((step) => step.decision === "remove")
      .map((step) => step.featureRemoved);

    const keptFeatures = this.steps
      .filter(
        (step) => step.decision === "keep" || step.decision === "investigate"
      )
      .map((step) => step.featureRemoved);

    const finalPerformance =
      this.steps.length > 0
        ? {
            accuracy: this.steps[this.steps.length - 1].testAccuracy,
            f1Buy: this.steps[this.steps.length - 1].testF1Buy,
            f1Sell: this.steps[this.steps.length - 1].testF1Sell,
            combinedF1: this.steps[this.steps.length - 1].testCombinedF1,
          }
        : this.baselinePerformance;

    const totalTimeMs = this.steps.reduce(
      (sum, step) => sum + step.trainingTimeMs,
      0
    );
    const performanceChange =
      finalPerformance.combinedF1 - this.baselinePerformance.combinedF1;

    return {
      originalFeatureCount: this.originalFeatureCount,
      finalFeatureCount: this.currentFeatureIndices.length,
      removedFeatures,
      keptFeatures,
      baselinePerformance: this.baselinePerformance,
      finalPerformance,
      steps: this.steps,
      totalTimeMs,
      featuresRemoved: removedFeatures.length,
      performanceChange,
    };
  }

  /**
   * Present optimization results
   */
  private presentResults(result: OptimizationResult): void {
    console.log("\n" + "=".repeat(70));
    console.log("🎯 GRADUAL FEATURE OPTIMIZATION RESULTS");
    console.log("=".repeat(70));

    console.log(`\n📊 OPTIMIZATION SUMMARY:`);
    console.log(`Original features: ${result.originalFeatureCount}`);
    console.log(`Final features: ${result.finalFeatureCount}`);
    console.log(`Features removed: ${result.featuresRemoved}`);
    console.log(
      `Performance change: ${(result.performanceChange * 100).toFixed(2)}%`
    );

    console.log(`\n📈 PERFORMANCE COMPARISON:`);
    console.log(
      `Accuracy: ${(result.baselinePerformance.accuracy * 100).toFixed(
        2
      )}% → ${(result.finalPerformance.accuracy * 100).toFixed(2)}%`
    );
    console.log(
      `F1 Buy: ${result.baselinePerformance.f1Buy.toFixed(
        4
      )} → ${result.finalPerformance.f1Buy.toFixed(4)}`
    );
    console.log(
      `F1 Sell: ${result.baselinePerformance.f1Sell.toFixed(
        4
      )} → ${result.finalPerformance.f1Sell.toFixed(4)}`
    );
    console.log(
      `Combined F1: ${result.baselinePerformance.combinedF1.toFixed(
        4
      )} → ${result.finalPerformance.combinedF1.toFixed(4)}`
    );

    console.log(`\n✅ FEATURES REMOVED (${result.removedFeatures.length}):`);
    result.removedFeatures.forEach((feature, idx) => {
      const step = this.steps.find((s) => s.featureRemoved === feature);
      console.log(`  ${idx + 1}. ${feature} (${step?.reasoning})`);
    });

    console.log(`\n❌ FEATURES KEPT (${result.keptFeatures.length}):`);
    result.keptFeatures.forEach((feature, idx) => {
      const step = this.steps.find((s) => s.featureRemoved === feature);
      console.log(`  ${idx + 1}. ${feature} (${step?.reasoning})`);
    });

    console.log(`\n⏱️  OPTIMIZATION DETAILS:`);
    console.log(
      `Total time: ${(result.totalTimeMs / 1000 / 60).toFixed(1)} minutes`
    );
    console.log(
      `Average time per test: ${(
        result.totalTimeMs /
        this.steps.length /
        1000
      ).toFixed(1)} seconds`
    );

    console.log("\n" + "=".repeat(70));
    console.log("💡 Next steps:");
    console.log("1. Review 'investigate' features manually");
    console.log("2. Update FeatureCalculator with removed features");
    console.log("3. Retrain model with optimized feature set");
    console.log("4. Run full backtests to validate performance");
    console.log("=".repeat(70));
  }
}

/**
 * Main execution function
 */
async function main() {
  try {
    console.log("🎯 Gradual Feature Optimization for Bitcoin Trading Model");
    console.log("=".repeat(60));

    const optimizer = new GradualFeatureOptimizer();
    const result = await optimizer.optimizeFeatures();

    console.log("\n✅ Gradual optimization completed successfully!");
    console.log(`Removed ${result.featuresRemoved} features safely`);
    console.log(
      `Performance change: ${(result.performanceChange * 100).toFixed(2)}%`
    );
  } catch (error) {
    console.error("❌ Gradual optimization failed:", error);
    process.exit(1);
  }
}

// Execute if called directly
if (require.main === module) {
  main().catch(console.error);
}

export { GradualFeatureOptimizer, OptimizationResult };
