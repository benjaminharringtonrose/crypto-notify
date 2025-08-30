/**
 * Example usage of the Bitcoin Ensemble Prediction System
 * This script demonstrates how to use the ensemble system for Bitcoin price prediction
 */

import {
  BitcoinDataCollector,
  FeatureEngineer,
  BitcoinPredictor,
  BitcoinPredictionEngine,
  EnsemblePredictor,
} from "./index";

async function basicExample() {
  console.log("🚀 Bitcoin Ensemble Prediction System - Basic Example\n");

  try {
    // Initialize the prediction engine
    const engine = new EnsemblePredictor();

    // Load or train models
    console.log("📚 Loading/training models...");
    const modelsReady = await engine.loadOrTrainModels();

    if (!modelsReady) {
      throw new Error("Failed to prepare models");
    }

    // Make a prediction
    console.log("🔮 Making prediction...");
    const prediction = await engine.makePrediction();

    // Display results
    console.log("\n📊 Prediction Results:");
    console.log("=".repeat(40));
    console.log(`Direction: ${prediction.direction}`);
    console.log(`Confidence: ${(prediction.confidence * 100).toFixed(1)}%`);
    console.log(`Current Price: $${prediction.currentPrice.toLocaleString()}`);
    console.log(`Prediction Window: ${prediction.predictionWindow}`);
    console.log(`Models Used: ${prediction.modelsUsed.join(", ")}`);
    console.log(`Is Confident: ${prediction.isConfident ? "Yes" : "No"}`);

    // Get additional information
    const summary = engine.getPredictionSummary();
    console.log(`\n📈 Summary:`);
    console.log(`Total Predictions Made: ${summary.predictionCount}`);

    const performance = engine.getModelPerformance();
    console.log(`Models Available: ${performance.totalModels}`);
    console.log(`Features Used: ${performance.featureCount}`);

    console.log("\n✅ Basic example completed successfully!");
  } catch (error) {
    console.error("❌ Basic example failed:", error);
  }
}

async function advancedExample() {
  console.log("\n🚀 Bitcoin Ensemble Prediction System - Advanced Example\n");

  try {
    // Step 1: Custom data collection
    console.log("📊 Step 1: Custom Data Collection");
    const collector = new BitcoinDataCollector({ dataDir: "custom_data" });
    const data = await collector.fetchHistoricalData("1y");
    console.log(`   Collected ${data.length} data points`);
    console.log(
      `   Date range: ${data[0].Date.toDateString()} to ${data[
        data.length - 1
      ].Date.toDateString()}`
    );
    console.log(
      `   Price range: $${Math.min(
        ...data.map((d) => d.Close)
      ).toLocaleString()} - $${Math.max(
        ...data.map((d) => d.Close)
      ).toLocaleString()}\n`
    );

    // Step 2: Feature engineering
    console.log("🔧 Step 2: Feature Engineering");
    const engineer = new FeatureEngineer();
    const dataWithFeatures = engineer.createTechnicalIndicators(data);
    const dataWithTarget = engineer.createTargetVariable(dataWithFeatures, 5); // 5-day prediction

    const { X, y, featureColumns } = engineer.selectFeatures(dataWithTarget);
    console.log(`   Created ${featureColumns.length} features`);
    console.log(`   Feature matrix: ${X.length} x ${X[0]?.length || 0}`);
    console.log(
      `   Target distribution: ${y.filter((val) => val === 1).length} up, ${
        y.filter((val) => val === 0).length
      } down\n`
    );

    // Step 3: Model training
    console.log("🤖 Step 3: Model Training");
    const predictor = new BitcoinPredictor();
    const results = await predictor.trainModels(X, y);

    console.log("   Training Results:");
    for (const [name, result] of Object.entries(results)) {
      if ("accuracy" in result) {
        console.log(
          `     ${name}: ${(result.accuracy * 100).toFixed(
            1
          )}% accuracy, ${result.trainingTime.toFixed(2)}s`
        );
      }
    }

    // Step 4: Ensemble prediction
    console.log("\n🎯 Step 4: Ensemble Prediction");
    const ensembleResult = await predictor.ensemblePredict(X.slice(-1));
    const prediction = ensembleResult.predictions[0];
    const probability = ensembleResult.probabilities[0];
    const confidence = ensembleResult.confidenceScores[0];

    console.log(`   Prediction: ${prediction === 1 ? "UP" : "DOWN"}`);
    console.log(`   Probability: ${(probability * 100).toFixed(1)}%`);
    console.log(`   Confidence: ${(confidence * 100).toFixed(1)}%`);

    // Step 5: Model analysis
    console.log("\n📈 Step 5: Model Analysis");
    const weights = predictor.getEnsembleWeights();
    console.log("   Ensemble Weights:");
    for (const [model, weight] of Object.entries(weights)) {
      console.log(`     ${model}: ${(weight * 100).toFixed(1)}%`);
    }

    console.log("\n✅ Advanced example completed successfully!");
  } catch (error) {
    console.error("❌ Advanced example failed:", error);
  }
}

async function realTimeExample() {
  console.log("\n🚀 Bitcoin Ensemble Prediction System - Real-time Example\n");

  try {
    const engine = new BitcoinPredictionEngine();

    // Initialize models
    console.log("📚 Initializing models...");
    await engine.loadOrTrainModels();

    // Simulate real-time predictions
    console.log("🔄 Simulating real-time predictions...\n");

    for (let i = 1; i <= 3; i++) {
      console.log(`--- Prediction ${i} ---`);

      // Get latest price
      const latestPrice = await engine.getLatestPrice();
      console.log(`Current Bitcoin Price: $${latestPrice.toLocaleString()}`);

      // Make prediction
      const prediction = await engine.makePrediction();
      console.log(
        `Prediction: ${prediction.direction} (${(
          prediction.confidence * 100
        ).toFixed(1)}% confidence)`
      );

      // Get price history
      const priceHistory = await engine.getPriceHistory(7);
      const priceChange =
        ((latestPrice - priceHistory[0].price) / priceHistory[0].price) * 100;
      console.log(
        `7-day price change: ${priceChange > 0 ? "+" : ""}${priceChange.toFixed(
          2
        )}%`
      );

      // Get model contributions
      const contributions = await engine.getModelContributions();
      const topModel = Object.entries(contributions).sort(
        (a, b) => b[1].contribution - a[1].contribution
      )[0];
      console.log(
        `Top contributing model: ${
          topModel[0]
        } (${topModel[1].contribution.toFixed(1)}%)`
      );

      console.log("");

      // Wait a bit before next prediction
      if (i < 3) {
        console.log("Waiting 2 seconds before next prediction...\n");
        await new Promise((resolve) => setTimeout(resolve, 2000));
      }
    }

    // Final analysis
    console.log("📊 Final Analysis:");
    const accuracy = await engine.getPredictionAccuracy();
    console.log(
      `Historical Accuracy: ${(accuracy.accuracy * 100).toFixed(1)}% (${
        accuracy.correct
      }/${accuracy.total})`
    );

    const confidenceDistribution = await engine.getConfidenceDistribution();
    console.log("Confidence Distribution:");
    for (const [range, count] of Object.entries(confidenceDistribution)) {
      if (count > 0) {
        console.log(`  ${range}: ${count} predictions`);
      }
    }

    console.log("\n✅ Real-time example completed successfully!");
  } catch (error) {
    console.error("❌ Real-time example failed:", error);
  }
}

// Main execution
async function main() {
  const args = process.argv.slice(2);

  console.log("Bitcoin Ensemble Prediction System Examples\n");

  if (args.includes("--basic")) {
    await basicExample();
  } else if (args.includes("--advanced")) {
    await advancedExample();
  } else if (args.includes("--realtime")) {
    await realTimeExample();
  } else if (args.includes("--all")) {
    await basicExample();
    await advancedExample();
    await realTimeExample();
  } else {
    console.log("Usage:");
    console.log(
      "  npm run ensemble:example -- --basic      # Basic usage example"
    );
    console.log(
      "  npm run ensemble:example -- --advanced   # Advanced usage example"
    );
    console.log(
      "  npm run ensemble:example -- --realtime   # Real-time prediction example"
    );
    console.log(
      "  npm run ensemble:example -- --all        # Run all examples"
    );
    console.log("\nRunning basic example...\n");
    await basicExample();
  }
}

// Run the example
if (require.main === module) {
  main().catch((error) => {
    console.error("Example runner failed:", error);
    process.exit(1);
  });
}

export { basicExample, advancedExample, realTimeExample };
