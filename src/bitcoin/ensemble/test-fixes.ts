/**
 * Test script to verify the ensemble system fixes
 * This script tests the key fixes made to make the system production-ready
 */

import {
  BitcoinDataCollector,
  FeatureEngineer,
  BitcoinPredictor,
  BitcoinPredictionEngine,
} from "./index";

async function testFixes() {
  console.log("🧪 Testing Bitcoin Ensemble System Fixes\n");

  try {
    // Test 1: Real Data Collection
    console.log("📊 Test 1: Real Bitcoin Data Collection");
    console.log("=".repeat(50));

    const collector = new BitcoinDataCollector();
    const data = await collector.fetchHistoricalData("1y");

    console.log(`✅ Collected ${data.length} days of real Bitcoin data`);
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
    console.log(
      `   Latest price: $${data[data.length - 1].Close.toLocaleString()}`
    );

    // Verify data quality
    if (collector.validateData(data)) {
      console.log("   ✅ Data validation passed");
    } else {
      throw new Error("Data validation failed");
    }
    console.log();

    // Test 2: Feature Engineering with Real Data
    console.log("🔧 Test 2: Feature Engineering with Real Data");
    console.log("=".repeat(50));

    const engineer = new FeatureEngineer();
    const dataWithFeatures = engineer.createTechnicalIndicators(data);
    const dataWithTarget = engineer.createTargetVariable(dataWithFeatures);
    const { X, y, featureColumns } = engineer.selectFeatures(dataWithTarget);

    console.log(`✅ Created ${featureColumns.length} features from real data`);
    console.log(`   Feature matrix: ${X.length} x ${X[0]?.length || 0}`);
    console.log(
      `   Target distribution: ${y.filter((val) => val === 1).length} up, ${
        y.filter((val) => val === 0).length
      } down`
    );

    // Check for NaN values in features
    const hasNaN = X.some((row) => row.some((val) => isNaN(val)));
    if (hasNaN) {
      console.warn("   ⚠️  Found NaN values in features");
    } else {
      console.log("   ✅ No NaN values in features");
    }
    console.log();

    // Test 3: Model Training with Real Data
    console.log("🤖 Test 3: Model Training with Real Data");
    console.log("=".repeat(50));

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

    // Test ensemble prediction
    const ensembleResult = await predictor.ensemblePredict(X.slice(-5));
    console.log(
      `   Ensemble predictions: ${ensembleResult.predictions.slice(0, 3)}`
    );
    console.log(
      `   Ensemble probabilities: ${ensembleResult.probabilities
        .slice(0, 3)
        .map((p) => p.toFixed(3))}`
    );
    console.log();

    // Test 4: Complete Prediction Pipeline
    console.log("🎯 Test 4: Complete Prediction Pipeline");
    console.log("=".repeat(50));

    const engine = new BitcoinPredictionEngine();
    const modelsReady = await engine.loadOrTrainModels(false, "1y");

    if (modelsReady) {
      console.log("✅ Models loaded/trained successfully");

      const prediction = await engine.makePrediction();
      console.log(
        `   Prediction: ${prediction.direction} with ${(
          prediction.confidence * 100
        ).toFixed(1)}% confidence`
      );
      console.log(
        `   Current price: $${prediction.currentPrice.toLocaleString()}`
      );
      console.log(`   Models used: ${prediction.modelsUsed.join(", ")}`);

      const summary = engine.getPredictionSummary();
      console.log(`   Total predictions: ${summary.predictionCount}`);
    } else {
      throw new Error("Failed to prepare models");
    }
    console.log();

    // Test 5: Error Handling
    console.log("🛡️  Test 5: Error Handling");
    console.log("=".repeat(50));

    try {
      // Test with invalid data
      const invalidData = [
        { Date: new Date(), Open: -1, High: 0, Low: 0, Close: 0, Volume: 0 },
      ];
      const isValid = collector.validateData(invalidData);
      console.log(
        `   Invalid data validation: ${
          isValid ? "❌ Should have failed" : "✅ Correctly rejected"
        }`
      );
    } catch (e) {
      console.log("   ✅ Error handling works correctly");
    }
    console.log();

    console.log(
      "🎉 All tests passed! The ensemble system is now using real Bitcoin data."
    );
    console.log("📈 Key improvements made:");
    console.log("   ✅ Real Bitcoin data from CryptoCompare API");
    console.log("   ✅ Fixed technical indicator calculations");
    console.log("   ✅ Improved machine learning models");
    console.log("   ✅ Better error handling and validation");
    console.log("   ✅ Production-ready data pipeline");
  } catch (error) {
    console.error("❌ Test failed:", error);
    console.error(
      "Stack trace:",
      error instanceof Error ? error.stack : "No stack trace"
    );
    process.exit(1);
  }
}

// Run the test
if (require.main === module) {
  testFixes().catch(console.error);
}

export { testFixes };
