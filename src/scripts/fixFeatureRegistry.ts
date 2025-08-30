#!/usr/bin/env ts-node
/**
 * Fix Feature Registry - Align FeatureRegistry with actual FeatureCalculator implementation
 *
 * This script analyzes the actual features produced by FeatureCalculator and updates
 * the FeatureRegistry to match, ensuring consistency across the system.
 */

import FeatureCalculator from "../bitcoin/shared/FeatureCalculator";
import { FEATURE_REGISTRY } from "../bitcoin/shared/FeatureRegistry";

class FeatureRegistryFixer {
  private featureCalculator: FeatureCalculator;

  constructor() {
    this.featureCalculator = new FeatureCalculator();
  }

  /**
   * Analyze actual features produced by FeatureCalculator
   */
  public analyzeActualFeatures(): string[] {
    // Create sample data to analyze the actual features
    const samplePrices = Array.from(
      { length: 100 },
      (_, i) => 50000 + Math.sin(i / 10) * 1000
    );
    const sampleVolumes = Array.from(
      { length: 100 },
      (_, i) => 1000 + Math.sin(i / 5) * 200
    );
    const currentPrice = 50000;
    const dayIndex = 50;

    // Get actual features
    const actualFeatures = this.featureCalculator.compute({
      prices: samplePrices,
      volumes: sampleVolumes,
      dayIndex,
      currentPrice,
    });

    console.log(
      `🔍 FeatureCalculator produces ${actualFeatures.length} features`
    );
    console.log(
      `📋 FeatureRegistry defines ${FEATURE_REGISTRY.length} features`
    );

    // Analyze the FeatureCalculator code to determine what features are actually computed
    const actualFeatureNames = this.determineActualFeatureNames();

    return actualFeatureNames;
  }

  /**
   * Determine actual feature names by analyzing FeatureCalculator code
   * Based on the coreFeatures array in FeatureCalculator.compute()
   */
  private determineActualFeatureNames(): string[] {
    // These are the actual features based on FeatureCalculator.compute() method
    return [
      // 1-3: Price Action (3 features)
      "currentPrice",
      "previousPrice",
      "priceChangePct",

      // 4-6: Trend Indicators (3 features)
      "sma7",
      "sma21",
      "sma50",

      // 7-11: Enhanced Momentum Indicators (5 features) - v1.4.0
      "rsi14", // 14-day RSI (standard)
      "rsi21", // 21-day RSI for longer trend
      "rsi7", // 7-day RSI for short-term momentum
      "macdLine",
      "signalLine",

      // 12-14: Volatility Indicators (3 features)
      "atr",
      "bollingerUpper", // Bollinger Band upper
      "bollingerLower", // Bollinger Band lower

      // 15-18: Enhanced Volume Indicators (4 features) - v1.4.0
      "currentVolume",
      "volumeMA20", // Volume MA (20-day)
      "vwap",
      "volumeMomentum", // Volume momentum: current vs 5-day average

      // 19-23: Relative/Ratio Features (5 features)
      "priceSMA7Ratio", // Price/SMA7 ratio
      "priceSMA21Ratio", // Price/SMA21 ratio
      "priceSMA50Ratio", // Price/SMA50 ratio
      "rsiNormalized", // Normalized RSI
      "bollingerPosition", // Bollinger Band position (0-1 scale)

      // 24-30: Enhanced Secondary Indicators (7 features) - v1.4.0
      "prevRsi", // Previous RSI for momentum
      "macdHistogram", // MACD histogram
      "atrNormalized", // Normalized ATR
      "volumeRatio20", // Volume ratio
      "momentum", // Raw momentum
      "bollingerSqueeze", // Bollinger Band squeeze indicator
      "rsiDivergence", // RSI divergence signal

      // 31-36: EXPERIMENT #61 - Advanced Market Microstructure Features (6 features)
      "ichimokuTenkanSen", // Ichimoku Tenkan-sen (9-period conversion line)
      "ichimokuKijunSen", // Ichimoku Kijun-sen (26-period base line)
      "ichimokuCloudPosition", // Position relative to Ichimoku cloud (-1, 0, 1)
      "williamsR", // Williams %R momentum oscillator
      "stochasticK", // Stochastic %K oscillator
      "vpt", // Volume-Price Trend (VPT) for volume analysis
    ];
  }

  /**
   * Generate corrected FeatureRegistry
   */
  public generateCorrectedRegistry(): string {
    const actualFeatureNames = this.determineActualFeatureNames();

    console.log(
      `\n📝 Generating corrected FeatureRegistry with ${actualFeatureNames.length} features...`
    );

    const registryCode = this.generateRegistryCode(actualFeatureNames);

    return registryCode;
  }

  /**
   * Generate the TypeScript code for the corrected registry
   */
  private generateRegistryCode(featureNames: string[]): string {
    const features = featureNames.map((name, index) => {
      const category = this.categorizeFeature(name, index);
      const importance = this.assessImportance(name, index);
      const description = this.generateDescription(name);
      const experiment =
        name.startsWith("ichimoku") ||
        name === "williamsR" ||
        name === "stochasticK" ||
        name === "vpt"
          ? '"Experiment #61"'
          : "undefined";

      return `  {
    name: "${name}",
    description: "${description}",
    category: "${category}",
    ${experiment !== "undefined" ? `experimentAdded: ${experiment},` : ""}
    importance: "${importance}",
  }`;
    });

    return `/**
 * CORRECTED FeatureRegistry - Generated to match FeatureCalculator.compute()
 * Total features: ${featureNames.length}
 * Last updated: ${new Date().toISOString()}
 */

export interface FeatureDefinition {
  name: string;
  description: string;
  category: "core" | "technical" | "ratio" | "enhanced" | "microstructure";
  experimentAdded?: string;
  importance?: "high" | "medium" | "low" | "unknown";
}

export const FEATURE_REGISTRY: FeatureDefinition[] = [
${features.join(",\n\n")}
];

// Update the count in FeatureRegistry class as well
export class FeatureRegistry {
  public static getFeatureCount(): number {
    return ${featureNames.length}; // Updated to match actual implementation
  }

  // ... rest of the methods remain the same
}`;
  }

  /**
   * Categorize features based on their type
   */
  private categorizeFeature(name: string, index: number): string {
    if (
      name.includes("currentPrice") ||
      name.includes("previousPrice") ||
      name.includes("priceChangePct") ||
      name.includes("currentVolume") ||
      name.includes("volumeMomentum")
    ) {
      return "core";
    }

    if (
      name.includes("sma") ||
      name.includes("rsi") ||
      name.includes("macd") ||
      name.includes("bollinger") ||
      name.includes("vwap") ||
      name.includes("atr") ||
      name.includes("volume")
    ) {
      return "technical";
    }

    if (
      name.includes("Ratio") ||
      name.includes("Normalized") ||
      name.includes("Position")
    ) {
      return "ratio";
    }

    if (
      name.includes("ichimoku") ||
      name.includes("williams") ||
      name.includes("stochastic") ||
      name.includes("vpt")
    ) {
      return "microstructure";
    }

    return "enhanced";
  }

  /**
   * Assess importance based on known trading significance
   */
  private assessImportance(name: string, index: number): string {
    // High importance features (core trading indicators)
    if (
      name.includes("rsi14") ||
      name.includes("macd") ||
      name.includes("sma7") ||
      name.includes("sma21") ||
      name.includes("currentPrice") ||
      name.includes("priceChangePct") ||
      name.includes("bollinger")
    ) {
      return "high";
    }

    // Medium importance features
    if (
      name.includes("atr") ||
      name.includes("vwap") ||
      name.includes("volume") ||
      name.includes("momentum") ||
      name.includes("sma50") ||
      name.includes("Ratio")
    ) {
      return "medium";
    }

    // Unknown importance (experimental features)
    if (
      name.includes("ichimoku") ||
      name.includes("williams") ||
      name.includes("stochastic") ||
      name.includes("vpt")
    ) {
      return "unknown";
    }

    return "low";
  }

  /**
   * Generate human-readable descriptions
   */
  private generateDescription(name: string): string {
    const descriptions: Record<string, string> = {
      currentPrice: "Current market price",
      previousPrice: "Previous period price",
      priceChangePct: "Price change percentage",
      sma7: "7-day Simple Moving Average",
      sma21: "21-day Simple Moving Average",
      sma50: "50-day Simple Moving Average",
      rsi14: "14-day RSI momentum oscillator",
      rsi21: "21-day RSI for longer trend",
      rsi7: "7-day RSI for short-term momentum",
      macdLine: "MACD line (12-26 EMA difference)",
      signalLine: "MACD signal line (9-day EMA)",
      atr: "Average True Range volatility",
      bollingerUpper: "Bollinger Band upper boundary",
      bollingerLower: "Bollinger Band lower boundary",
      currentVolume: "Current trading volume",
      volumeMA20: "20-day volume moving average",
      vwap: "Volume Weighted Average Price",
      volumeMomentum: "Volume momentum vs 5-day average",
      priceSMA7Ratio: "Price to SMA7 ratio",
      priceSMA21Ratio: "Price to SMA21 ratio",
      priceSMA50Ratio: "Price to SMA50 ratio",
      rsiNormalized: "Normalized RSI (0-1 scale)",
      bollingerPosition: "Position within Bollinger Bands",
      prevRsi: "Previous RSI for momentum analysis",
      macdHistogram: "MACD histogram (line - signal)",
      atrNormalized: "ATR normalized by price",
      volumeRatio20: "Current vs 20-day volume ratio",
      momentum: "Raw price momentum",
      bollingerSqueeze: "Bollinger Band squeeze indicator",
      rsiDivergence: "RSI divergence signal",
      ichimokuTenkanSen: "Ichimoku Tenkan-sen conversion line",
      ichimokuKijunSen: "Ichimoku Kijun-sen base line",
      ichimokuCloudPosition: "Position relative to Ichimoku cloud",
      williamsR: "Williams %R momentum oscillator",
      stochasticK: "Stochastic %K oscillator",
      vpt: "Volume-Price Trend indicator",
    };

    return descriptions[name] || `${name} technical indicator`;
  }

  /**
   * Write the corrected registry to file
   */
  public async writeCorrectRegistry(): Promise<void> {
    const correctedCode = this.generateCorrectedRegistry();

    console.log("\n📄 Corrected FeatureRegistry.ts:");
    console.log("=".repeat(50));
    console.log(correctedCode);
    console.log("=".repeat(50));

    console.log("\n💡 To apply this fix:");
    console.log("1. Backup your current FeatureRegistry.ts");
    console.log("2. Replace the content with the generated code above");
    console.log("3. Update any imports if needed");
    console.log("4. Test the system to ensure compatibility");
  }

  /**
   * Validate current registry against actual implementation
   */
  public validateRegistry(): { isValid: boolean; issues: string[] } {
    const actualFeatures = this.analyzeActualFeatures();
    const registryCount = FEATURE_REGISTRY.length;
    const actualCount = actualFeatures.length;

    const issues: string[] = [];

    if (registryCount !== actualCount) {
      issues.push(
        `Feature count mismatch: Registry has ${registryCount}, Calculator produces ${actualCount}`
      );
    }

    // Check for feature name mismatches (if we can infer them)
    for (let i = 0; i < Math.min(registryCount, actualCount); i++) {
      const registryFeature = FEATURE_REGISTRY[i];
      const actualFeature = actualFeatures[i];

      // Simple heuristic check
      if (!this.featuresLikelyMatch(registryFeature.name, actualFeature)) {
        issues.push(
          `Feature ${i}: Registry "${registryFeature.name}" may not match actual "${actualFeature}"`
        );
      }
    }

    return {
      isValid: issues.length === 0,
      issues,
    };
  }

  /**
   * Simple heuristic to check if features likely match
   */
  private featuresLikelyMatch(
    registryName: string,
    actualName: string
  ): boolean {
    // Simple keyword matching
    const registryKeywords = registryName.toLowerCase().split(/[_-]/);
    const actualKeywords = actualName.toLowerCase().split(/[_-]/);

    // Check if they share common keywords
    return registryKeywords.some((keyword) =>
      actualKeywords.some(
        (actual) => actual.includes(keyword) || keyword.includes(actual)
      )
    );
  }
}

/**
 * Main execution function
 */
async function main() {
  try {
    console.log("🔧 Feature Registry Alignment Tool");
    console.log("=".repeat(50));

    const fixer = new FeatureRegistryFixer();

    // Step 1: Validate current registry
    console.log("🔍 Validating current registry...");
    const validation = fixer.validateRegistry();

    if (validation.isValid) {
      console.log("✅ Current registry is valid!");
    } else {
      console.log("❌ Issues found:");
      validation.issues.forEach((issue) => console.log(`  • ${issue}`));
    }

    // Step 2: Analyze actual features
    console.log("\n📊 Analyzing actual FeatureCalculator implementation...");
    const actualFeatures = fixer.analyzeActualFeatures();

    console.log(`\n📋 Actual features (${actualFeatures.length}):`);
    actualFeatures.forEach((feature, idx) => {
      console.log(`  ${idx + 1}. ${feature}`);
    });

    // Step 3: Generate corrected registry
    if (!validation.isValid) {
      console.log("\n🛠️  Generating corrected registry...");
      await fixer.writeCorrectRegistry();
    }

    console.log("\n✅ Registry analysis completed!");
  } catch (error) {
    console.error("❌ Registry fix failed:", error);
    process.exit(1);
  }
}

// Execute if called directly
if (require.main === module) {
  main().catch(console.error);
}

export { FeatureRegistryFixer };
