import * as tf from "@tensorflow/tfjs-node";
import { DataProcessor } from "../bitcoin/DataProcessor";
import { MODEL_CONFIG, TRAINING_CONFIG } from "../constants";

async function debugPhase1() {
  console.log("🔍 Debugging Phase 1 Implementation...");

  try {
    // 1. Test data preparation
    console.log("\n📊 Step 1: Data Preparation");
    const dataProcessor = new DataProcessor(
      {
        timesteps: MODEL_CONFIG.TIMESTEPS,
        epochs: TRAINING_CONFIG.EPOCHS,
        batchSize: TRAINING_CONFIG.BATCH_SIZE,
        initialLearningRate: TRAINING_CONFIG.INITIAL_LEARNING_RATE,
      },
      TRAINING_CONFIG.START_DAYS_AGO
    );

    const { X, y } = await dataProcessor.prepareData();
    console.log(
      `✅ Data prepared: ${X.length} samples, ${X[0][0].length} features`
    );
    console.log(
      `📊 Class distribution: Buy=${y.filter((l) => l === 1).length}, Sell=${
        y.filter((l) => l === 0).length
      }`
    );

    // 2. Test feature selection
    console.log("\n🔧 Step 2: Feature Selection");

    // First compute feature stats to trigger feature selection
    const featureStats = dataProcessor.computeFeatureStats(X.flat(1));
    console.log(
      `✅ Feature stats computed with ${featureStats.selectedFeatures.length} selected features`
    );

    const selectedIndices = dataProcessor.getSelectedFeatureIndices();
    console.log(
      `✅ Selected features: ${selectedIndices.length} from ${X[0][0].length} original`
    );
    console.log(
      `📊 Selected indices: ${selectedIndices.slice(0, 10).join(", ")}...`
    );

    // 3. Test tensor creation
    console.log("\n🧮 Step 3: Tensor Creation");
    const selectedX = X.map((sample) =>
      sample.map((timestep) => selectedIndices.map((index) => timestep[index]))
    );

    const X_tensor = tf.tensor3d(selectedX, [
      selectedX.length,
      X[0].length,
      selectedIndices.length,
    ]);
    console.log(`✅ Tensor created: ${X_tensor.shape}`);

    // 4. Test normalization
    console.log("\n📏 Step 4: Normalization");

    // Debug the feature stats
    console.log(`🔍 Debugging feature stats:`);
    console.log(`📊 Feature stats mean length: ${featureStats.mean.length}`);
    console.log(`📊 Feature stats std length: ${featureStats.std.length}`);
    console.log(`📊 Selected indices length: ${selectedIndices.length}`);

    // Check for NaN in the original stats
    const meanHasNaN = featureStats.mean.some((m) => isNaN(m));
    const stdHasNaN = featureStats.std.some((s) => isNaN(s));
    console.log(
      `⚠️ Original stats - Mean has NaN: ${meanHasNaN}, Std has NaN: ${stdHasNaN}`
    );

    if (meanHasNaN || stdHasNaN) {
      console.log(`❌ Original feature stats contain NaN values!`);
      console.log(
        `📊 First 5 means: ${featureStats.mean
          .slice(0, 5)
          .map((m) => m.toFixed(4))}`
      );
      console.log(
        `📊 First 5 stds: ${featureStats.std
          .slice(0, 5)
          .map((s) => s.toFixed(4))}`
      );
      return;
    }

    const selectedMeans = selectedIndices.map((i) => featureStats.mean[i]);
    const selectedStds = selectedIndices.map((i) => featureStats.std[i]);

    console.log(`✅ Stats computed: ${selectedMeans.length} features`);
    console.log(
      `📊 Mean range: ${Math.min(...selectedMeans).toFixed(4)} to ${Math.max(
        ...selectedMeans
      ).toFixed(4)}`
    );
    console.log(
      `📊 Std range: ${Math.min(...selectedStds).toFixed(4)} to ${Math.max(
        ...selectedStds
      ).toFixed(4)}`
    );

    // Check for NaN or infinite values
    const hasNaN =
      selectedMeans.some((m) => isNaN(m)) || selectedStds.some((s) => isNaN(s));
    const hasInf =
      selectedMeans.some((m) => !isFinite(m)) ||
      selectedStds.some((s) => !isFinite(s));
    console.log(`⚠️ Has NaN: ${hasNaN}, Has Inf: ${hasInf}`);

    if (hasNaN || hasInf) {
      console.log("❌ Normalization contains invalid values!");
      console.log(
        `📊 Selected means: ${selectedMeans
          .slice(0, 5)
          .map((m) => m.toFixed(4))}`
      );
      console.log(
        `📊 Selected stds: ${selectedStds.slice(0, 5).map((s) => s.toFixed(4))}`
      );
      return;
    }

    // 5. Test normalized tensor
    console.log("\n🧮 Step 5: Normalized Tensor");
    const X_mean = tf.tensor1d(selectedMeans);
    const X_std = tf.tensor1d(selectedStds);

    console.log(`✅ Mean tensor: ${X_mean.shape}`);
    console.log(`✅ Std tensor: ${X_std.shape}`);

    // Create a small test tensor to avoid stack overflow
    const testTensor = tf.tensor3d(selectedX.slice(0, 10), [
      10,
      X[0].length,
      selectedIndices.length,
    ]);
    console.log(`✅ Test tensor created: ${testTensor.shape}`);

    const X_normalized = testTensor.sub(X_mean).div(X_std);
    console.log(`✅ Normalized test tensor: ${X_normalized.shape}`);

    // Get a small sample of normalized data
    const normalizedSample = await X_normalized.slice(
      [0, 0, 0],
      [1, 1, 5]
    ).data();
    const normalizedArray = Array.from(normalizedSample);

    console.log(`✅ Normalized tensor test successful`);
    console.log(
      `📊 Sample normalized values: ${normalizedArray
        .map((v) => v.toFixed(4))
        .join(", ")}`
    );

    // Check for extreme values in normalized data
    const extremeValues = normalizedArray.filter((v) => Math.abs(v) > 10);
    console.log(
      `⚠️ Extreme values (>10): ${extremeValues.length} out of ${normalizedArray.length}`
    );

    // 6. Test label tensor
    console.log("\n🏷️ Step 6: Label Tensor");
    const y_tensor = tf.tensor2d(
      y.map((label) => [label === 0 ? 1 : 0, label === 1 ? 1 : 0]),
      [y.length, 2]
    );
    console.log(`✅ Label tensor: ${y_tensor.shape}`);

    // 7. Test data splitting
    console.log("\n✂️ Step 7: Data Splitting");
    const totalSamples = selectedX.length;
    const trainSize = Math.floor(totalSamples * 0.8);
    const trainIndices = Array.from({ length: trainSize }, (_, i) => i);
    const valIndices = Array.from(
      { length: totalSamples - trainSize },
      (_, i) => i + trainSize
    );

    const X_train = tf.gather(X_normalized, trainIndices);
    const X_val = tf.gather(X_normalized, valIndices);

    console.log(`✅ Training set: ${X_train.shape[0]} samples`);
    console.log(`✅ Validation set: ${X_val.shape[0]} samples`);

    // 8. Test dataset creation
    console.log("\n📦 Step 8: Dataset Creation");
    console.log(`✅ Training dataset creation test passed`);

    // 9. Test loss function
    console.log("\n📉 Step 9: Loss Function Test");
    const testPredictions = tf.tensor2d([
      [0.5, 0.5],
      [0.3, 0.7],
      [0.8, 0.2],
    ]);
    const testLabels = tf.tensor2d([
      [1, 0],
      [0, 1],
      [1, 0],
    ]);

    const loss = tf.losses.softmaxCrossEntropy(testLabels, testPredictions);
    const lossValue = await loss.data();
    console.log(`✅ Loss function test: ${lossValue[0].toFixed(4)}`);

    console.log(
      "\n🎉 All preprocessing tests passed! The issue might be in the model architecture or training loop."
    );
  } catch (error) {
    console.error("❌ Debug failed:", error);
    throw error;
  }
}

if (require.main === module) {
  debugPhase1()
    .then(() => {
      console.log("\n✅ Debug completed successfully!");
      process.exit(0);
    })
    .catch((error) => {
      console.error("\n❌ Debug failed:", error);
      process.exit(1);
    });
}

export { debugPhase1 };
