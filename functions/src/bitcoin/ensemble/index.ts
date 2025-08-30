/**
 * Bitcoin Ensemble Prediction System
 *
 * This module provides a complete TypeScript implementation of a Bitcoin price prediction
 * system using ensemble machine learning methods. It recreates the Python architecture
 * with the following components:
 *
 * - Data Collection: Fetches and manages Bitcoin price data
 * - Feature Engineering: Creates technical indicators and features
 * - Machine Learning Models: Implements ensemble prediction algorithms
 * - Prediction Engine: Handles real-time predictions and model management
 */

// Export all modules from the ensemble directory
export { BitcoinDataCollector, testDataCollection } from "./DataCollector";
export { FeatureEngineer, testFeatureEngineering } from "./FeatureEngineer";
export { BitcoinPredictor, testModels } from "./Models";
export {
  BitcoinPredictionEngine,
  testPredictionEngine,
} from "./PredictionEngine";

// Re-export the main prediction engine as EnsemblePredictor for convenience
export { BitcoinPredictionEngine as EnsemblePredictor } from "./PredictionEngine";

// Export types
export type {
  BitcoinDataPoint,
  BitcoinDataCollectorConfig,
} from "./DataCollector";

export type {
  FeatureDataPoint,
  FeatureEngineeringResult,
} from "./FeatureEngineer";

export type {
  ModelPrediction,
  ModelMetrics,
  TrainingResult,
  EnsembleWeights,
  ModelConfig,
} from "./Models";

export type {
  PredictionResult,
  PredictionSummary,
  ValidationResult,
  ModelPerformance,
  PredictionEngineConfig,
} from "./PredictionEngine";

// Import test functions
import { testDataCollection } from "./DataCollector";
import { testFeatureEngineering } from "./FeatureEngineer";
import { testModels } from "./Models";
import { testPredictionEngine } from "./PredictionEngine";

// Test functions
export async function testEnsemble(): Promise<void> {
  console.log("Testing Bitcoin Ensemble Prediction System...");

  try {
    // Test data collection
    console.log("\n1. Testing Data Collection...");
    await testDataCollection();

    // Test feature engineering
    console.log("\n2. Testing Feature Engineering...");
    await testFeatureEngineering();

    // Test models
    console.log("\n3. Testing Models...");
    await testModels();

    // Test prediction engine
    console.log("\n4. Testing Prediction Engine...");
    await testPredictionEngine();

    console.log("\n✅ All ensemble tests completed successfully!");
  } catch (error) {
    console.error("❌ Ensemble test failed:", error);
    throw error;
  }
}

export {
  runAllTests,
  testIndividualComponents,
  testEnsemblePerformance,
} from "./test-ensemble";
export { basicExample, advancedExample, realTimeExample } from "./example";
export { testFixes } from "./test-fixes";
