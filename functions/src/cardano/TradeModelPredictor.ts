import * as tf from "@tensorflow/tfjs-node";
import { ModelWeightManager } from "./TradeModelWeightManager";
import TradeModelFactory from "./TradeModelFactory";
import { FeatureSequenceGenerator } from "./FeatureSequenceGenerator";
import { FirebaseService } from "../api/FirebaseService";

export class TradeModelPredictor {
  private weightManager: ModelWeightManager;
  private sequenceGenerator: FeatureSequenceGenerator;
  private timesteps = 30;
  private model: tf.LayersModel;
  private isWeightsLoaded = false;
  private temperature = 0.5; // Reduced for stronger signals

  constructor() {
    this.weightManager = new ModelWeightManager();
    this.sequenceGenerator = new FeatureSequenceGenerator(this.timesteps);
    FirebaseService.getInstance();
    const factory = new TradeModelFactory(this.timesteps, 61);
    this.model = factory.createModel();
    this.loadWeightsAsync();
  }

  private async loadWeightsAsync(): Promise<void> {
    try {
      await this.weightManager.loadWeights();
      this.weightManager.setWeights(this.model);
      this.isWeightsLoaded = true;
      console.log("Weights loaded successfully");
    } catch (error) {
      console.error("Failed to load weights:", error);
      throw error;
    }
  }

  public async predict(
    adaPrices: number[],
    adaVolumes: number[],
    btcPrices: number[],
    btcVolumes: number[]
  ): Promise<{ buyProb: number; sellProb: number; confidence: number }> {
    if (!this.isWeightsLoaded) {
      console.log("Weights not yet loaded, awaiting load...");
      await this.loadWeightsAsync();
    }

    const startTime = performance.now();

    // Log input data stats for debugging
    console.log(
      `Input stats - ADA Prices: min=${Math.min(...adaPrices).toFixed(
        4
      )}, max=${Math.max(...adaPrices).toFixed(4)}, mean=${(
        adaPrices.reduce((a, b) => a + b, 0) / adaPrices.length
      ).toFixed(4)}`
    );
    console.log(
      `Input stats - BTC Prices: min=${Math.min(...btcPrices).toFixed(
        4
      )}, max=${Math.max(...btcPrices).toFixed(4)}, mean=${(
        btcPrices.reduce((a, b) => a + b, 0) / btcPrices.length
      ).toFixed(4)}`
    );

    const startIndex = Math.max(0, adaPrices.length - this.timesteps - 1);
    const endIndex = adaPrices.length - 1;
    const sequence = this.sequenceGenerator.generateSequence(
      adaPrices,
      adaVolumes,
      btcPrices,
      btcVolumes,
      startIndex,
      endIndex
    );

    const features = tf.tensor3d([sequence], [1, this.timesteps, 61]);
    const means = tf.tensor1d(this.weightManager.getFeatureMeans());
    const stds = tf.tensor1d(this.weightManager.getFeatureStds());
    const featuresNormalized = features.sub(means).div(stds.add(1e-6));

    const logits = this.model.predict(featuresNormalized) as tf.Tensor2D;
    const scaledLogits = logits.div(tf.scalar(this.temperature));
    const probs = scaledLogits.softmax();

    const probArray = await probs.data();
    const [sellProb, buyProb] = [probArray[0], probArray[1]];
    const confidence = Math.max(buyProb, sellProb);

    const moments = tf.moments(logits);
    const variance = moments.variance.dataSync()[0];

    const endTime = performance.now();
    console.log(
      `Prediction executed in ${(endTime - startTime).toFixed(2)} ms`
    );
    console.log(
      `Raw logits: [Sell: ${sellProb.toFixed(4)}, Buy: ${buyProb.toFixed(4)}]`
    );
    console.log(
      `Calibrated probs: [Sell: ${sellProb.toFixed(4)}, Buy: ${buyProb.toFixed(
        4
      )}]`
    );
    console.log(
      `Confidence: ${confidence.toFixed(4)}, Variance: ${variance.toFixed(4)}`
    );

    features.dispose();
    featuresNormalized.dispose();
    logits.dispose();
    scaledLogits.dispose();
    probs.dispose();
    means.dispose();
    stds.dispose();
    moments.mean.dispose();
    moments.variance.dispose();

    return { buyProb, sellProb, confidence };
  }
}
