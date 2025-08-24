/**
 * FeatureRegistry - Central registry of all features with metadata
 * This eliminates hardcoded feature counts and enables feature analysis
 */

export interface FeatureDefinition {
  name: string;
  description: string;
  category: "core" | "technical" | "ratio" | "enhanced" | "microstructure";
  experimentAdded?: string; // e.g., "Experiment #61"
  importance?: "high" | "medium" | "low" | "unknown";
}

/**
 * OPTIMIZED feature registry matching FeatureCalculator.ts optimizedFeatures array
 * Order MUST match the exact order in FeatureCalculator
 * Optimized from 36 to 26 features based on gradual optimization results
 */
export const FEATURE_REGISTRY: FeatureDefinition[] = [
  // 1-5: Core Price Action & Volatility (5 features)
  {
    name: "priceChangePct",
    description: "Price change percentage",
    category: "core",
    importance: "high",
  },
  {
    name: "highLowRange",
    description: "High-low price range",
    category: "core",
    importance: "medium",
  },
  {
    name: "priceVolatility",
    description: "Short-term price volatility",
    category: "core",
    importance: "high",
  },
  {
    name: "pricePosition",
    description: "Price position in recent range",
    category: "core",
    importance: "medium",
  },
  {
    name: "relativeVolume",
    description: "Relative volume indicator",
    category: "core",
    importance: "medium",
  },

  // 6-10: Technical Indicators (5 features)
  {
    name: "rsi",
    description: "RSI momentum oscillator",
    category: "technical",
    importance: "high",
  },
  {
    name: "signalLine",
    description: "MACD signal line",
    category: "technical",
    importance: "high",
  },
  {
    name: "vwapRatio",
    description: "Price to VWAP ratio",
    category: "technical",
    importance: "high",
  },
  {
    name: "atr",
    description: "Average True Range",
    category: "technical",
    importance: "high",
  },
  {
    name: "obv",
    description: "On-Balance Volume",
    category: "technical",
    importance: "high",
  },

  // 11-15: Enhanced Indicators (5 features)
  {
    name: "momentum",
    description: "Raw momentum",
    category: "enhanced",
    importance: "medium",
  },
  {
    name: "macdHistogram",
    description: "MACD histogram",
    category: "enhanced",
    importance: "medium",
  },
  {
    name: "priceSMA7Ratio",
    description: "Price to SMA7 ratio",
    category: "ratio",
    importance: "medium",
  },
  {
    name: "priceSMA21Ratio",
    description: "Price to SMA21 ratio",
    category: "ratio",
    importance: "medium",
  },
  {
    name: "priceSMA50Ratio",
    description: "Price to SMA50 ratio",
    category: "ratio",
    importance: "medium",
  },

  // 16-20: Market Regime Features (5 features)
  {
    name: "trendRegime",
    description: "Trend regime score",
    category: "enhanced",
    importance: "high",
  },
  {
    name: "volatilityRegime",
    description: "Volatility regime score (normalized)",
    category: "enhanced",
    importance: "medium",
  },
  {
    name: "ichimokuTenkanSen",
    description: "Ichimoku Tenkan-sen (9-period)",
    category: "microstructure",
    experimentAdded: "Experiment #61",
    importance: "medium",
  },
  {
    name: "ichimokuKijunSen",
    description: "Ichimoku Kijun-sen (26-period)",
    category: "microstructure",
    experimentAdded: "Experiment #61",
    importance: "medium",
  },
  {
    name: "ichimokuCloudPosition",
    description: "Position relative to Ichimoku cloud",
    category: "microstructure",
    experimentAdded: "Experiment #61",
    importance: "medium",
  },

  // 21-26: Advanced Microstructure Features (6 features)
  {
    name: "williamsR",
    description: "Williams %R momentum oscillator",
    category: "microstructure",
    experimentAdded: "Experiment #61",
    importance: "medium",
  },
  {
    name: "vpt",
    description: "Volume-Price Trend (VPT)",
    category: "microstructure",
    experimentAdded: "Experiment #61",
    importance: "high",
  },
  {
    name: "volumeMA20",
    description: "20-day volume moving average",
    category: "technical",
    importance: "medium",
  },
  {
    name: "volumeOscillator",
    description: "Volume oscillator",
    category: "technical",
    importance: "medium",
  },
  {
    name: "bollingerSqueeze",
    description: "Bollinger Band squeeze indicator",
    category: "enhanced",
    importance: "low",
  },
  {
    name: "rsiDivergence",
    description: "RSI divergence signal",
    category: "enhanced",
    importance: "low",
  },
  {
    name: "cci",
    description: "Commodity Channel Index (CCI)",
    category: "microstructure",
    experimentAdded: "Experiment #2 - CCI",
    importance: "medium",
  },
  {
    name: "mfi",
    description: "Money Flow Index (MFI)",
    category: "microstructure",
    experimentAdded: "Experiment #3 - MFI",
    importance: "medium",
  },
  {
    name: "aroonOscillator",
    description: "Aroon Oscillator",
    category: "microstructure",
    experimentAdded: "Experiment #5 - Aroon",
    importance: "medium",
  },
];

/**
 * FeatureRegistry utility functions
 */
export class FeatureRegistry {
  /**
   * Get total feature count
   */
  public static getFeatureCount(): number {
    return FEATURE_REGISTRY.length;
  }

  /**
   * Get features by category
   */
  public static getFeaturesByCategory(
    category: FeatureDefinition["category"]
  ): FeatureDefinition[] {
    return FEATURE_REGISTRY.filter((f) => f.category === category);
  }

  /**
   * Get features added in a specific experiment
   */
  public static getFeaturesByExperiment(
    experimentName: string
  ): FeatureDefinition[] {
    return FEATURE_REGISTRY.filter((f) => f.experimentAdded === experimentName);
  }

  /**
   * Get features by importance level
   */
  public static getFeaturesByImportance(
    importance: FeatureDefinition["importance"]
  ): FeatureDefinition[] {
    return FEATURE_REGISTRY.filter((f) => f.importance === importance);
  }

  /**
   * Get feature definition by index
   */
  public static getFeatureByIndex(index: number): FeatureDefinition | null {
    return FEATURE_REGISTRY[index] || null;
  }

  /**
   * Get feature index by name
   */
  public static getFeatureIndex(name: string): number {
    return FEATURE_REGISTRY.findIndex((f) => f.name === name);
  }

  /**
   * Validate that feature count matches expected count
   */
  public static validateFeatureCount(actualCount: number): void {
    const expectedCount = this.getFeatureCount();
    if (actualCount !== expectedCount) {
      throw new Error(
        `Feature count mismatch! Registry defines ${expectedCount} features, got ${actualCount}. ` +
          `Check FeatureCalculator.ts coreFeatures array matches FeatureRegistry.ts`
      );
    }
  }

  /**
   * Get summary statistics
   */
  public static getSummary(): {
    total: number;
    byCategory: Record<string, number>;
    byImportance: Record<string, number>;
  } {
    const byCategory: Record<string, number> = {};
    const byImportance: Record<string, number> = {};

    FEATURE_REGISTRY.forEach((feature) => {
      byCategory[feature.category] = (byCategory[feature.category] || 0) + 1;
      byImportance[feature.importance || "unknown"] =
        (byImportance[feature.importance || "unknown"] || 0) + 1;
    });

    return {
      total: FEATURE_REGISTRY.length,
      byCategory,
      byImportance,
    };
  }

  /**
   * Get feature names in order (useful for analysis)
   */
  public static getFeatureNames(): string[] {
    return FEATURE_REGISTRY.map((f) => f.name);
  }
}
