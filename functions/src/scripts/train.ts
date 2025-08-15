import { TradeModelTrainer } from "../bitcoin/TradeModelTrainer";
import { TRAINING_CONFIG } from "../constants";

async function runTraining() {
  console.log("🚀 Starting Enhanced BTC Model Training");
  console.log("=".repeat(60));

  console.log("📊 Training Configuration:");
  console.log(`   Epochs: ${TRAINING_CONFIG.EPOCHS}`);
  console.log(`   Batch Size: ${TRAINING_CONFIG.BATCH_SIZE}`);
  console.log(`   Learning Rate: ${TRAINING_CONFIG.INITIAL_LEARNING_RATE}`);
  console.log(`   Focal Loss Gamma: ${TRAINING_CONFIG.GAMMA}`);
  console.log(`   Focal Loss Alpha: [${TRAINING_CONFIG.ALPHA.join(", ")}]`);
  console.log(`   Train Split: ${TRAINING_CONFIG.TRAIN_SPLIT}`);
  console.log(`   Patience: ${TRAINING_CONFIG.PATIENCE}`);

  const trainer = new TradeModelTrainer();

  try {
    console.log("\n🔄 Starting training process...");
    const startTime = Date.now();

    await trainer.train();

    const endTime = Date.now();
    const trainingTime = (endTime - startTime) / 1000;

    console.log("\n✅ Training completed successfully!");
    console.log(`⏱️  Total training time: ${trainingTime.toFixed(2)} seconds`);

    const bestThreshold = trainer.getBestThreshold();
    console.log(`🎯 Best validation threshold: ${bestThreshold.toFixed(4)}`);

    console.log("\n📈 Next Steps:");
    console.log("   1. Run backtest to evaluate trading performance");
    console.log("   2. Analyze model predictions and confidence");
    console.log("   3. Monitor live trading performance");
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
