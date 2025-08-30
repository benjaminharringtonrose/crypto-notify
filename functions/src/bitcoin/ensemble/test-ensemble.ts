/**
 * Test script for the Bitcoin Ensemble Prediction System
 * This script tests all components of the ensemble architecture
 */

import {
  BitcoinDataCollector,
  FeatureEngineer,
  BitcoinPredictor,
  BitcoinPredictionEngine,
  testDataCollection,
  testFeatureEngineering,
  testModels,
  testPredictionEngine,
} from "./index";

async function runAllTests() {
  console.log("🚀 Starting Bitcoin Ensemble Prediction System Tests\n");

  try {
    // Test 1: Data Collection
    console.log("📊 Test 1: Data Collection");
    console.log("=".repeat(50));
    await testDataCollection();
    console.log("✅ Data Collection Test Passed\n");

    // Test 2: Feature Engineering
    console.log("🔧 Test 2: Feature Engineering");
    console.log("=".repeat(50));
    await testFeatureEngineering();
    console.log("✅ Feature Engineering Test Passed\n");

    // Test 3: Models
    console.log("🤖 Test 3: Machine Learning Models");
    console.log("=".repeat(50));
    await testModels();
    console.log("✅ Models Test Passed\n");

    // Test 4: Complete Pipeline
    console.log("🎯 Test 4: Complete Prediction Pipeline");
    console.log("=".repeat(50));
    await testPredictionEngine();
    console.log("✅ Complete Pipeline Test Passed\n");

    console.log("🎉 All Tests Passed Successfully!");
  } catch (error) {
    console.error("❌ Test Failed:", error);
    process.exit(1);
  }
}

async function testIndividualComponents() {
  console.log("🔍 Testing Individual Components\n");

  try {
    // Test Data Collector
    console.log("Testing Data Collector...");
    const collector = new BitcoinDataCollector();
    const data = await collector.getLatestData(30);
    console.log(`✅ Collected ${data.length} data points`);
    console.log(
      `   Latest price: $${data[data.length - 1].Close.toLocaleString()}`
    );
    console.log(
      `   Date range: ${data[0].Date.toDateString()} to ${data[
        data.length - 1
      ].Date.toDateString()}\n`
    );

    // Test Feature Engineer
    console.log("Testing Feature Engineer...");
    const engineer = new FeatureEngineer();
    const dataWithFeatures = engineer.createTechnicalIndicators(data);
    const dataWithTarget = engineer.createTargetVariable(dataWithFeatures);
    const { X, y, featureColumns } = engineer.selectFeatures(dataWithTarget);
    console.log(`✅ Created ${featureColumns.length} features`);
    console.log(`   Feature matrix: ${X.length} x ${X[0]?.length || 0}`);
    console.log(
      `   Target distribution: ${y.filter((val) => val === 1).length} up, ${
        y.filter((val) => val === 0).length
      } down\n`
    );

    // Test Predictor
    console.log("Testing Predictor...");
    const predictor = new BitcoinPredictor();
    const results = await predictor.trainModels(X, y);
    console.log(`✅ Trained ${Object.keys(results).length} models`);
    for (const [name, result] of Object.entries(results)) {
      if ("accuracy" in result) {
        console.log(
          `   ${name}: ${(result.accuracy * 100).toFixed(1)}% accuracy`
        );
      }
    }
    console.log();

    // Test Prediction Engine
    console.log("Testing Prediction Engine...");
    const engine = new BitcoinPredictionEngine();
    const modelsReady = await engine.loadOrTrainModels();
    if (modelsReady) {
      const prediction = await engine.makePrediction();
      console.log(
        `✅ Made prediction: ${prediction.direction} with ${(
          prediction.confidence * 100
        ).toFixed(1)}% confidence`
      );
      console.log(
        `   Current price: $${prediction.currentPrice.toLocaleString()}`
      );
      console.log(`   Models used: ${prediction.modelsUsed.join(", ")}`);
    }
    console.log();
  } catch (error) {
    console.error("❌ Component Test Failed:", error);
    throw error;
  }
}

async function testEnsemblePerformance() {
  console.log("⚡ Testing Ensemble Performance\n");

  try {
    const engine = new BitcoinPredictionEngine();

    // Train models
    console.log("Training models...");
    await engine.loadOrTrainModels();

    // Make multiple predictions
    console.log("Making predictions...");
    const predictions = [];
    for (let i = 0; i < 5; i++) {
      const prediction = await engine.makePrediction();
      predictions.push(prediction);
      console.log(
        `   Prediction ${i + 1}: ${prediction.direction} (${(
          prediction.confidence * 100
        ).toFixed(1)}% confidence)`
      );
    }

    // Analyze predictions
    const upPredictions = predictions.filter(
      (p) => p.direction === "UP"
    ).length;
    const downPredictions = predictions.filter(
      (p) => p.direction === "DOWN"
    ).length;
    const avgConfidence =
      predictions.reduce((sum, p) => sum + p.confidence, 0) /
      predictions.length;

    console.log(`\n📈 Prediction Analysis:`);
    console.log(`   UP predictions: ${upPredictions}`);
    console.log(`   DOWN predictions: ${downPredictions}`);
    console.log(`   Average confidence: ${(avgConfidence * 100).toFixed(1)}%`);

    // Get model performance
    const performance = engine.getModelPerformance();
    console.log(`\n🤖 Model Performance:`);
    console.log(`   Total models: ${performance.totalModels}`);
    console.log(`   Feature count: ${performance.featureCount}`);

    // Get model contributions
    const contributions = await engine.getModelContributions();
    console.log(`\n⚖️ Model Contributions:`);
    for (const [model, contrib] of Object.entries(contributions)) {
      console.log(`   ${model}: ${contrib.contribution.toFixed(1)}%`);
    }

    console.log("\n✅ Performance Test Completed\n");
  } catch (error) {
    console.error("❌ Performance Test Failed:", error);
    throw error;
  }
}

// Main execution
async function main() {
  const args = process.argv.slice(2);

  if (args.includes("--all")) {
    await runAllTests();
  } else if (args.includes("--components")) {
    await testIndividualComponents();
  } else if (args.includes("--performance")) {
    await testEnsemblePerformance();
  } else {
    console.log("Bitcoin Ensemble Prediction System Test Runner\n");
    console.log("Usage:");
    console.log("  npm run test:ensemble -- --all          # Run all tests");
    console.log(
      "  npm run test:ensemble -- --components   # Test individual components"
    );
    console.log(
      "  npm run test:ensemble -- --performance  # Test ensemble performance"
    );
    console.log("\nRunning default test (components)...\n");
    await testIndividualComponents();
  }
}

// Run the test
if (require.main === module) {
  main().catch((error) => {
    console.error("Test runner failed:", error);
    process.exit(1);
  });
}

export { runAllTests, testIndividualComponents, testEnsemblePerformance };
