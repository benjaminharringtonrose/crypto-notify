import { CryptoCompareService } from "../api/CryptoCompareService";
import { FirebaseService } from "../api/FirebaseService";
import { RLTradingStrategy } from "../bitcoin/RLTradingStrategy";

// Initialize Firebase
FirebaseService.getInstance();

async function testRLIntegration(): Promise<void> {
  console.log("🧪 Testing RL Integration...");

  try {
    // Load data
    const cryptoCompare = new CryptoCompareService();
    const btcData = await cryptoCompare.getHistoricalData("BTC", 100); // 100 days for quick test

    console.log(`📈 Loaded ${btcData.prices.length} days of BTC data`);

    // Create RL trading strategy
    const rlStrategy = new RLTradingStrategy(null, null, {
      useRLAgent: false, // We'll test without RL agent for now
      rlWeight: 0.6,
      ruleBasedWeight: 0.4,
      confidenceThreshold: 0.7,
      fallbackToRuleBased: true,
      adaptiveWeighting: true,
      performanceWindow: 30,
    });

    console.log("🤖 Testing RL Trading Strategy...");

    // Test decision making
    for (let i = 35; i < 45; i++) {
      const decision = await rlStrategy.makeDecision(
        btcData.prices,
        btcData.volumes,
        i
      );

      console.log(`Day ${i}:`);
      console.log(`  Action: ${decision.action}`);
      console.log(`  Confidence: ${decision.confidence.toFixed(3)}`);
      console.log(`  RL Action: ${decision.rlAction}`);
      console.log(`  RL Confidence: ${decision.rlConfidence.toFixed(3)}`);
      console.log(`  Rule-Based Action: ${decision.ruleBasedAction}`);
      console.log(
        `  Rule-Based Confidence: ${decision.ruleBasedConfidence.toFixed(3)}`
      );
      console.log(`  Final Weight: ${decision.finalWeight.toFixed(3)}`);
      console.log(`  Reasoning: ${decision.reasoning}`);
      console.log("");
    }

    // Test strategy state management
    console.log("📊 Testing Strategy State Management...");
    const state = rlStrategy.exportState();
    console.log("Strategy State:", JSON.stringify(state, null, 2));

    // Test performance tracking
    console.log("📈 Testing Performance Tracking...");
    const performance = rlStrategy.getPerformanceHistory();
    console.log(`Performance History Length: ${performance.length}`);

    // Test configuration
    console.log("⚙️ Testing Configuration...");
    const config = rlStrategy.getRLConfig();
    console.log("RL Config:", JSON.stringify(config, null, 2));

    console.log("✅ RL Integration Test Completed Successfully!");
  } catch (error) {
    console.error("❌ RL Integration Test Failed:", error);
    throw error;
  }
}

// Run test
if (require.main === module) {
  testRLIntegration().catch(console.error);
}

export { testRLIntegration };
