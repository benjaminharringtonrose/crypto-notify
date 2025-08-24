import { MODEL_CONFIG } from "../constants";
import FeatureCalculator from "./FeatureCalculator";

export class FeatureSequenceGenerator {
  private timesteps: number;

  constructor(timesteps: number) {
    this.timesteps = timesteps;
  }

  public generateSequence(
    btcPrices: number[],
    btcVolumes: number[],
    startIndex: number,
    endIndex: number
  ): number[][] {
    const sequence: number[][] = [];
    const safeEndIndex = Math.min(endIndex, btcPrices.length - 1);
    const safeStartIndex = Math.max(0, safeEndIndex - this.timesteps + 1);

    for (let i = safeStartIndex; i <= safeEndIndex; i++) {
      const featureCalculator = new FeatureCalculator();

      const btcFeatures = featureCalculator.compute({
        prices: btcPrices,
        volumes: btcVolumes,
        dayIndex: i,
        currentPrice: btcPrices[i],
      });

      sequence.push(btcFeatures);
    }

    while (sequence.length < this.timesteps) {
      sequence.unshift(
        sequence[0] || Array(MODEL_CONFIG.FEATURE_COUNT).fill(0)
      );
    }
    return sequence.slice(-this.timesteps); // Ensure exact length
  }

  public generateBatchSequences(
    btcPrices: number[],
    btcVolumes: number[],
    startIndex: number,
    endIndex: number,
    stepDays: number
  ): number[][][] {
    const sequences: number[][][] = [];
    if (
      btcPrices.length < this.timesteps ||
      btcVolumes.length < this.timesteps
    ) {
      console.error("Insufficient data length for sequence generation");
      return sequences;
    }

    for (let i = startIndex; i < endIndex; i += stepDays) {
      const sequence = this.generateSequence(
        btcPrices,
        btcVolumes,
        Math.max(0, i - this.timesteps + 1),
        i
      );
      if (sequence.length === this.timesteps) {
        sequences.push(sequence);
      } else {
        console.warn(`Invalid sequence length at index ${i}`);
      }
    }

    console.log(`Generated ${sequences.length} sequences`);
    return sequences;
  }
}
