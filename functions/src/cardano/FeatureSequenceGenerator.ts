import FeatureCalculator from "./FeatureCalculator";

export class FeatureSequenceGenerator {
  private timesteps: number;

  constructor(timesteps: number) {
    this.timesteps = timesteps;
  }

  public generateSequence(
    adaPrices: number[],
    adaVolumes: number[],
    btcPrices: number[],
    btcVolumes: number[],
    startIndex: number,
    endIndex: number
  ): number[][] {
    const sequence: number[][] = [];
    // Ensure we don’t exceed array bounds
    const safeEndIndex = Math.min(endIndex, adaPrices.length - 1);
    const safeStartIndex = Math.max(0, safeEndIndex - this.timesteps + 1);

    for (let i = safeStartIndex; i <= safeEndIndex; i++) {
      const adaFeatures = new FeatureCalculator({
        prices: adaPrices,
        volumes: adaVolumes,
        dayIndex: i,
        currentPrice: adaPrices[i],
        isBTC: false,
      }).compute();
      const btcFeatures = new FeatureCalculator({
        prices: btcPrices,
        volumes: btcVolumes,
        dayIndex: i,
        currentPrice: btcPrices[i],
        isBTC: true,
      }).compute();
      sequence.push([...adaFeatures, ...btcFeatures]);
    }

    // Pad if necessary
    while (sequence.length < this.timesteps) {
      sequence.unshift(sequence[0] || Array(61).fill(0)); // Default to zeros if empty
    }
    return sequence;
  }

  public generateBatchSequences(
    adaPrices: number[],
    adaVolumes: number[],
    btcPrices: number[],
    btcVolumes: number[],
    startIndex: number,
    endIndex: number,
    stepDays: number
  ): number[][][] {
    const sequences: number[][][] = [];
    if (
      adaPrices.length < this.timesteps ||
      btcPrices.length < this.timesteps ||
      adaVolumes.length < this.timesteps ||
      btcVolumes.length < this.timesteps
    ) {
      console.error("Insufficient data length for sequence generation");
      return sequences;
    }

    for (let i = startIndex; i < endIndex; i += stepDays) {
      const sequence = this.generateSequence(
        adaPrices,
        adaVolumes,
        btcPrices,
        btcVolumes,
        Math.max(0, i - this.timesteps + 1),
        i
      );
      if (sequence.length > 0) {
        sequences.push(sequence);
      } else {
        console.warn(`No sequence generated for index ${i}`);
      }
    }

    console.log(`Generated ${sequences.length} sequences`);
    return sequences;
  }
}
