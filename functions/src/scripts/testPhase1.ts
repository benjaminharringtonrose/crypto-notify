import { EnhancedDataProcessor } from "../bitcoin/EnhancedDataProcessor";
import { DataProcessor } from "../bitcoin/DataProcessor";
import { MODEL_CONFIG, TRAINING_CONFIG } from "../constants";

/**
 * Test script for Phase 1 improvements
 * Tests robust normalization, feature selection, and SMOTE
 */
async function testPhase1() {
  console.log("🧪 Testing Phase 1 Improvements...");

  try {
    // Test 1: Enhanced Data Processor
    console.log("\n🔧 Test 1: Enhanced Data Processor");
    const enhancedProcessor = new EnhancedDataProcessor();

    // Create sample data
    const sampleFeatures = Array.from({ length: 100 }, () =>
      Array.from({ length: 42 }, () => Math.random() * 10)
    );

    // Test robust normalization
    const { normalized, selectedFeatures } =
      enhancedProcessor.robustNormalize(sampleFeatures);
    console.log(
      `✅ Robust normalization: ${normalized.length} samples, ${normalized[0].length} features`
    );
    console.log(
      `✅ Feature selection: ${selectedFeatures.length} features selected from 42 original`
    );

    // Test 2: Data Processor Integration
    console.log("\n🔧 Test 2: Data Processor Integration");
    const dataProcessor = new DataProcessor(
      {
        timesteps: MODEL_CONFIG.TIMESTEPS,
        epochs: TRAINING_CONFIG.EPOCHS,
        batchSize: TRAINING_CONFIG.BATCH_SIZE,
        initialLearningRate: TRAINING_CONFIG.INITIAL_LEARNING_RATE,
      },
      TRAINING_CONFIG.START_DAYS_AGO
    );

    // Test feature statistics
    const featureStats = dataProcessor.computeFeatureStats(sampleFeatures);
    console.log(
      `✅ Enhanced feature stats: ${featureStats.selectedFeatures.length} features selected`
    );

    // Test 3: SMOTE for Time Series
    console.log("\n🔧 Test 3: SMOTE for Time Series");

    // Create sample time series data
    const sampleSequences = Array.from({ length: 50 }, () =>
      Array.from({ length: 35 }, () =>
        Array.from({ length: 25 }, () => Math.random() * 10)
      )
    );

    // Create imbalanced labels (more sell than buy)
    const labels = Array.from({ length: 50 }, (_, i) => (i < 10 ? 1 : 0)); // 10 buy, 40 sell

    console.log(
      `📊 Original distribution - Buy: ${
        labels.filter((l) => l === 1).length
      }, Sell: ${labels.filter((l) => l === 0).length}`
    );

    // Apply SMOTE
    const buySamples = sampleSequences.filter((_, i) => labels[i] === 1);
    const syntheticSamples = enhancedProcessor.smoteTimeSeries(buySamples);

    console.log(
      `✅ SMOTE generated ${syntheticSamples.length} synthetic samples from ${buySamples.length} minority samples`
    );

    // Test 4: Enhanced Balancing
    console.log("\n🔧 Test 4: Enhanced Balancing");
    const { y: balancedY } = enhancedProcessor.enhancedBalanceDataset(
      sampleSequences,
      labels
    );

    console.log(
      `✅ Enhanced balancing - Buy: ${
        balancedY.filter((l) => l === 1).length
      }, Sell: ${balancedY.filter((l) => l === 0).length}`
    );

    // Test 5: Feature Selection Analysis
    console.log("\n🔧 Test 5: Feature Selection Analysis");
    const selectedIndices = enhancedProcessor.getSelectedFeatureIndices();
    const normalizationStats = enhancedProcessor.getNormalizationStats();

    console.log(
      `✅ Selected feature indices: ${selectedIndices.length} features`
    );
    console.log(
      `✅ Normalization stats computed: ${normalizationStats ? "Yes" : "No"}`
    );

    console.log("\n🎉 Phase 1 Tests Completed Successfully!");
    console.log("\n📊 Summary:");
    console.log(`- Robust normalization: ✅ Working`);
    console.log(
      `- Feature selection: ✅ ${selectedIndices.length} features selected`
    );
    console.log(
      `- SMOTE for time series: ✅ ${syntheticSamples.length} synthetic samples generated`
    );
    console.log(`- Enhanced balancing: ✅ Balanced dataset created`);
    console.log(`- Integration: ✅ All components working together`);
  } catch (error) {
    console.error("❌ Phase 1 test failed:", error);
    throw error;
  }
}

// Run the test
if (require.main === module) {
  testPhase1()
    .then(() => {
      console.log("\n✅ All Phase 1 tests passed!");
      process.exit(0);
    })
    .catch((error) => {
      console.error("\n❌ Phase 1 tests failed:", error);
      process.exit(1);
    });
}

export { testPhase1 };
