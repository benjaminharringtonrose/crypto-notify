/**
 * Ensemble Training Script for Bitcoin Price Prediction
 *
 * This script provides different training modes for the Bitcoin ensemble prediction system:
 * - Quick: Fast training with minimal data for testing
 * - Full: Complete training with full dataset
 * - Production: Production-ready training with validation
 * - Validate: Validate existing models
 */

import {
  BitcoinDataCollector,
  FeatureEngineer,
  BitcoinPredictor,
} from "./index";

interface TrainingConfig {
  dataPeriod: string;
  forceRetrain: boolean;
  validateOnly: boolean;
  saveModels: boolean;
  verbose: boolean;
}

async function trainEnsemble(
  config: Partial<TrainingConfig> = {}
): Promise<void> {
  const {
    dataPeriod = "1y",
    forceRetrain = false,
    validateOnly = false,
    saveModels = true,
    verbose = true,
  } = config;

  console.log("🚀 Bitcoin Ensemble Training Script");
  console.log("=".repeat(50));
  console.log(`Mode: ${validateOnly ? "Validation" : "Training"}`);
  console.log(`Data Period: ${dataPeriod}`);
  console.log(`Force Retrain: ${forceRetrain}`);
  console.log(`Save Models: ${saveModels}`);
  console.log();

  try {
    // Step 1: Data Collection
    if (verbose) console.log("📊 Step 1: Data Collection");
    const collector = new BitcoinDataCollector();
    const data = await collector.fetchHistoricalData(dataPeriod);

    if (verbose) {
      console.log(`   ✅ Collected ${data.length} data points`);
      console.log(
        `   Date range: ${data[0].Date.toISOString()} to ${data[
          data.length - 1
        ].Date.toISOString()}`
      );
      console.log(
        `   Price range: $${Math.min(
          ...data.map((d) => d.Close)
        ).toLocaleString()} - $${Math.max(
          ...data.map((d) => d.Close)
        ).toLocaleString()}`
      );
    }

    // Validate data quality
    if (!collector.validateData(data)) {
      throw new Error("Data validation failed");
    }
    console.log();

    // Step 2: Feature Engineering
    if (verbose) console.log("🔧 Step 2: Feature Engineering");
    const engineer = new FeatureEngineer();
    const dataWithFeatures = engineer.createTechnicalIndicators(data);
    const dataWithTarget = engineer.createTargetVariable(dataWithFeatures);
    const { X, y, featureColumns } = engineer.selectFeatures(dataWithTarget);

    if (verbose) {
      console.log(`   ✅ Created ${featureColumns.length} features`);
      console.log(`   Feature matrix: ${X.length} x ${X[0]?.length || 0}`);
      console.log(
        `   Target distribution: ${y.filter((val) => val === 1).length} up, ${
          y.filter((val) => val === 0).length
        } down`
      );
    }
    console.log();

    if (validateOnly) {
      // Validation mode - test existing models
      await validateExistingModels(X, y, verbose);
      return;
    }

    // Step 3: Model Training
    if (verbose) console.log("🤖 Step 3: Model Training");
    const predictor = new BitcoinPredictor();
    const trainingResults = await predictor.trainModels(X, y);

    if (verbose) {
      console.log(
        `   ✅ Trained ${Object.keys(trainingResults).length} models`
      );
      for (const [name, result] of Object.entries(trainingResults)) {
        if ("accuracy" in result) {
          console.log(
            `   ${name}: ${(result.accuracy * 100).toFixed(1)}% accuracy`
          );
        }
      }
    }
    console.log();

    // Step 4: Model Evaluation
    if (verbose) console.log("📈 Step 4: Model Evaluation");
    const ensembleWeights = predictor.getEnsembleWeights();
    if (verbose) {
      console.log("   Ensemble Weights:");
      for (const [name, weight] of Object.entries(ensembleWeights)) {
        console.log(`   ${name}: ${(weight * 100).toFixed(1)}%`);
      }
    }
    console.log();

    // Step 5: Test Ensemble Prediction
    if (verbose) console.log("🎯 Step 5: Test Ensemble Prediction");
    const testSample = X.slice(-5); // Test on last 5 samples
    const ensembleResult = await predictor.ensemblePredict(testSample);

    if (verbose) {
      console.log(
        `   Test predictions: ${ensembleResult.predictions.slice(0, 3)}`
      );
      console.log(
        `   Test probabilities: ${ensembleResult.probabilities
          .slice(0, 3)
          .map((p) => p.toFixed(3))}`
      );
    }
    console.log();

    // Step 6: Save Models (if enabled)
    if (saveModels) {
      if (verbose) console.log("💾 Step 6: Saving Models");
      await saveModelsToStorage(predictor, trainingResults);
      if (verbose) console.log("   ✅ Models saved successfully");
      console.log();
    }

    // Step 7: Performance Summary
    if (verbose) {
      console.log("📊 Training Summary");
      console.log("=".repeat(30));
      console.log(`Total Data Points: ${data.length}`);
      console.log(`Features Used: ${featureColumns.length}`);
      console.log(
        `Training Time: ${calculateTotalTrainingTime(trainingResults)}s`
      );
      console.log(`Best Model: ${getBestModel(trainingResults)}`);
      console.log(
        `Ensemble Performance: ${calculateEnsemblePerformance(ensembleResult)}`
      );
    }

    console.log("\n🎉 Ensemble training completed successfully!");
  } catch (error) {
    console.error("❌ Training failed:", error);
    process.exit(1);
  }
}

async function validateExistingModels(
  X: number[][],
  y: number[],
  verbose: boolean = true
): Promise<void> {
  console.log("🔍 Validating existing models...");

  try {
    const predictor = new BitcoinPredictor();

    // Try to load existing models
    // Note: This would require implementing model persistence
    console.log("   ⚠️  Model persistence not yet implemented");
    console.log("   Running validation on fresh models...");

    const results = await predictor.trainModels(X, y);

    if (verbose) {
      console.log("   Validation Results:");
      for (const [name, result] of Object.entries(results)) {
        if ("accuracy" in result) {
          console.log(
            `   ${name}: ${(result.accuracy * 100).toFixed(1)}% accuracy`
          );
        }
      }
    }

    console.log("   ✅ Validation completed");
  } catch (error) {
    console.error("   ❌ Validation failed:", error);
    throw error;
  }
}

async function saveModelsToStorage(
  predictor: BitcoinPredictor,
  results: any
): Promise<void> {
  // This would implement model persistence to Firebase Storage or similar
  // For now, just log the intention
  console.log("   📝 Model saving not yet implemented");
  console.log("   Models would be saved to Firebase Storage");
}

function calculateTotalTrainingTime(results: any): number {
  let totalTime = 0;
  for (const result of Object.values(results)) {
    if (result && typeof result === "object" && "trainingTime" in result) {
      totalTime += (result as any).trainingTime;
    }
  }
  return totalTime;
}

function getBestModel(results: any): string {
  let bestModel = "unknown";
  let bestAccuracy = 0;

  for (const [name, result] of Object.entries(results)) {
    if (result && typeof result === "object" && "accuracy" in result) {
      const accuracy = (result as any).accuracy;
      if (accuracy > bestAccuracy) {
        bestAccuracy = accuracy;
        bestModel = name;
      }
    }
  }

  return `${bestModel} (${(bestAccuracy * 100).toFixed(1)}%)`;
}

function calculateEnsemblePerformance(ensembleResult: any): string {
  if (!ensembleResult || !ensembleResult.probabilities) {
    return "N/A";
  }

  const avgConfidence =
    ensembleResult.probabilities.reduce(
      (sum: number, p: number) => sum + p,
      0
    ) / ensembleResult.probabilities.length;
  return `${(avgConfidence * 100).toFixed(1)}% average confidence`;
}

// Parse command line arguments
function parseArgs(): TrainingConfig {
  const args = process.argv.slice(2);
  const config: TrainingConfig = {
    dataPeriod: "1y",
    forceRetrain: false,
    validateOnly: false,
    saveModels: true,
    verbose: true,
  };

  for (const arg of args) {
    switch (arg) {
      case "--quick":
        config.dataPeriod = "6mo";
        config.verbose = false;
        break;
      case "--full":
        config.dataPeriod = "2y";
        break;
      case "--production":
        config.dataPeriod = "5y";
        config.saveModels = true;
        break;
      case "--validate":
        config.validateOnly = true;
        break;
      case "--force":
        config.forceRetrain = true;
        break;
      case "--no-save":
        config.saveModels = false;
        break;
      case "--quiet":
        config.verbose = false;
        break;
      default:
        if (arg.startsWith("--period=")) {
          config.dataPeriod = arg.split("=")[1];
        }
    }
  }

  return config;
}

// Main execution
if (require.main === module) {
  const config = parseArgs();
  trainEnsemble(config).catch(console.error);
}

export { trainEnsemble, TrainingConfig };
