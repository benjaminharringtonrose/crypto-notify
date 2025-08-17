import { TradeModelTrainer } from "../bitcoin/TradeModelTrainer";

async function runTraining() {
  const trainer = new TradeModelTrainer();

  try {
    console.log("\n🚀 EXPERIMENT ACCURACY-2: MAXIMUM CAPACITY OVERHAUL");
    console.log(
      "🔬 TESTING: Dual LSTM (128+64), Massive Dense (128), Ultra-High LR (0.005)"
    );
    console.log(
      "💡 HYPOTHESIS: Need EXTREME capacity + learning rate for complex crypto patterns"
    );
    console.log(
      "📊 ACCURACY-1: ~52-56% training accuracy (STILL INSUFFICIENT!)"
    );
    console.log("🎯 TARGET: >75% training accuracy - FINAL ATTEMPT!\n");

    const startTime = Date.now();

    await trainer.train();

    const endTime = Date.now();
    const trainingTime = (endTime - startTime) / 1000;

    console.log(`⏱️  Total training time: ${trainingTime.toFixed(2)} seconds`);

    const bestThreshold = trainer.getBestThreshold();
    console.log(`🎯 Best validation threshold: ${bestThreshold.toFixed(4)}`);

    console.log("\n📈 EXPERIMENT ACCURACY-2 RESULTS:");
    console.log("ACCURACY-1: ~52-56% training accuracy (STILL FAILED!)");
    console.log("ACCURACY-2: Analyzing maximum capacity architecture...");
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
