import { TradeModelTrainer } from "../bitcoin/TradeModelTrainer";
import { TRAINING_CONFIG } from "../constants";

async function runTraining() {
  console.log("🚀 Starting FAST-LEARNING BTC Model Training");
  console.log("=".repeat(60));
  console.log("🎯 OPTIMIZATIONS APPLIED:");
  console.log("   ✅ Simplified CNN-LSTM architecture (Conv1D+LSTM+Dense)");
  console.log("   ✅ Stratified data splitting for diverse market conditions");
  console.log("   ✅ Reduced epochs (30), increased learning rate (0.002)");
  console.log("   ✅ Reduced data size (300 days), increased batch size (32)");
  console.log(
    "   ✅ Removed complex features (attention, residual, batch norm)"
  );
  console.log("   ✅ Simple labeling with 0.05% threshold");
  console.log("=".repeat(60));

  console.log("\n📊 Training Configuration:");
  console.log(`   Epochs: ${TRAINING_CONFIG.EPOCHS} (REDUCED for speed)`);
  console.log(
    `   Batch Size: ${TRAINING_CONFIG.BATCH_SIZE} (INCREASED for stability)`
  );
  console.log(
    `   Learning Rate: ${TRAINING_CONFIG.INITIAL_LEARNING_RATE} (INCREASED for faster learning)`
  );
  console.log(
    `   Data Days: ${TRAINING_CONFIG.START_DAYS_AGO} (REDUCED for speed)`
  );
  console.log(
    `   Train Split: ${TRAINING_CONFIG.TRAIN_SPLIT} (Stratified split)`
  );
  console.log(
    `   Patience: ${TRAINING_CONFIG.PATIENCE} (REDUCED for faster stopping)`
  );
  console.log(
    `   Time-based Split: ${TRAINING_CONFIG.TIME_BASED_SPLIT} (DISABLED)`
  );

  const trainer = new TradeModelTrainer();

  try {
    console.log("\n🔄 Starting OPTIMIZED training process...");
    const startTime = Date.now();

    await trainer.train();

    const endTime = Date.now();
    const trainingTime = (endTime - startTime) / 1000;

    console.log("\n✅ FAST Training completed successfully!");
    console.log(`⏱️  Total training time: ${trainingTime.toFixed(2)} seconds`);
    console.log(
      `⚡ Speed improvement: Expected 3-5x faster than previous runs`
    );

    const bestThreshold = trainer.getBestThreshold();
    console.log(`🎯 Best validation threshold: ${bestThreshold.toFixed(4)}`);

    console.log("\n📈 Expected Improvements:");
    console.log("   🎯 Dynamic predictions (no more static 0.0232/0.9768)");
    console.log("   🚀 Much faster convergence (30 epochs vs 200)");
    console.log("   📊 Better class balance through stratified splitting");
    console.log("   💰 Improved win rate through better learning");

    console.log("\n📋 Next Steps:");
    console.log("   1. Run backtest to verify dynamic predictions");
    console.log("   2. Check for improved win rate (target: >50%)");
    console.log("   3. Monitor training convergence and learning");
  } catch (error) {
    console.error("\n❌ Training failed:", error);
    console.error(
      "Stack trace:",
      error instanceof Error ? error.stack : "No stack trace"
    );
    process.exit(1);
  }
}

// Run the enhanced training
runTraining();
