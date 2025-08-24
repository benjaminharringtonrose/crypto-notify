import { FeatureRegistry } from "./FeatureRegistry";

/**
 * FeatureDetector - Simple feature count detection using FeatureRegistry
 * Eliminates hardcoded feature counts and prevents tensor shape mismatches
 */
export class FeatureDetector {
  private static featureCount: number | null = null;

  /**
   * Get feature count from the centralized FeatureRegistry
   * Much simpler and faster than dynamic detection
   */
  public static async detectFeatureCount(): Promise<number> {
    if (this.featureCount !== null) {
      return this.featureCount;
    }

    console.log("🔍 Getting feature count from FeatureRegistry...");

    try {
      this.featureCount = FeatureRegistry.getFeatureCount();

      console.log(`✅ Feature count from registry: ${this.featureCount}`);

      // Log feature breakdown for debugging
      const summary = FeatureRegistry.getSummary();
      console.log(`📊 Feature breakdown:`, summary.byCategory);

      return this.featureCount;
    } catch (error) {
      console.error("❌ Failed to get feature count from registry:", error);
      throw new Error(
        "Feature registry access failed. Check FeatureRegistry definition."
      );
    }
  }

  /**
   * Get the cached feature count (must call detectFeatureCount first)
   */
  public static getFeatureCount(): number {
    if (this.featureCount === null) {
      throw new Error(
        "Feature count not detected yet. Call detectFeatureCount() first."
      );
    }
    return this.featureCount;
  }

  /**
   * Validate that a data sample has the expected feature count
   */
  public static validateFeatureCount(features: number[][]): void {
    if (this.featureCount === null) {
      throw new Error(
        "Feature count not detected yet. Call detectFeatureCount() first."
      );
    }

    if (features.length === 0) {
      throw new Error("No features provided for validation");
    }

    const actualFeatureCount = features[0].length;
    FeatureRegistry.validateFeatureCount(actualFeatureCount);
  }

  /**
   * Validate tensor shape matches expected feature count
   */
  public static validateTensorShape(
    shape: number[],
    expectedTimesteps: number
  ): void {
    if (this.featureCount === null) {
      throw new Error(
        "Feature count not detected yet. Call detectFeatureCount() first."
      );
    }

    const [, timesteps, features] = shape;

    if (timesteps !== expectedTimesteps) {
      throw new Error(
        `Timesteps mismatch! Expected ${expectedTimesteps}, got ${timesteps}`
      );
    }

    if (features !== this.featureCount) {
      throw new Error(
        `Feature dimension mismatch! Expected ${this.featureCount}, got ${features}. ` +
          `Tensor shape: [${shape.join(", ")}]`
      );
    }

    console.log(`✅ Tensor shape validated: [${shape.join(", ")}]`);
  }

  /**
   * Get dynamic input shape for model creation
   */
  public static getInputShape(timesteps: number): [number, number] {
    if (this.featureCount === null) {
      throw new Error(
        "Feature count not detected yet. Call detectFeatureCount() first."
      );
    }
    return [timesteps, this.featureCount];
  }

  /**
   * Clear cached feature count (useful for testing)
   */
  public static clearCache(): void {
    this.featureCount = null;
  }

  /**
   * Get diagnostic information about detected features
   */
  public static getDiagnosticInfo(): {
    featureCount: number | null;
    isDetected: boolean;
  } {
    return {
      featureCount: this.featureCount,
      isDetected: this.featureCount !== null,
    };
  }
}
