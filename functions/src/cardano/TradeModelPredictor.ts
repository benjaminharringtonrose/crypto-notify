import * as tf from "@tensorflow/tfjs-node";
import { ModelWeightManager } from "./TradeModelWeightManager";
import TradeModelFactory from "./TradeModelFactory";
import { FeatureSequenceGenerator } from "./FeatureSequenceGenerator";
import { FirebaseService } from "../api/FirebaseService";

export class TradeModelPredictor {
  private weightManager: ModelWeightManager;
  private sequenceGenerator: FeatureSequenceGenerator;
  private timesteps = 30;
  private model: tf.LayersModel; // Model stored as class property

  constructor() {
    this.weightManager = new ModelWeightManager();
    this.sequenceGenerator = new FeatureSequenceGenerator(this.timesteps);
    FirebaseService.getInstance();
    // Initialize model once
    const factory = new TradeModelFactory(this.timesteps, 61);
    this.model = factory.createModel();
    // Load weights asynchronously and set them
    this.weightManager
      .loadWeights()
      .then(() => this.weightManager.setWeights(this.model));
  }

  public async predict(
    adaPrices: number[],
    adaVolumes: number[],
    btcPrices: number[],
    btcVolumes: number[]
  ): Promise<{ buyProb: number; sellProb: number; confidence: number }> {
    // Ensure weights are loaded before prediction
    if (!this.weightManager["weights"]) {
      await this.weightManager.loadWeights();
      this.weightManager.setWeights(this.model);
    }

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

    const logits = this.model.predict(featuresNormalized) as tf.Tensor2D; // Reuse model
    const scaledLogits = logits.mul(tf.scalar(2));
    const buyLogit = scaledLogits.slice([0, 1], [1, 1]).sigmoid();
    const sellLogit = scaledLogits.slice([0, 0], [1, 1]).sigmoid();

    let buyProbRaw = (await buyLogit.data())[0];
    let sellProbRaw = (await sellLogit.data())[0];
    const total = buyProbRaw + sellProbRaw;

    let buyProb = total > 0 ? buyProbRaw / total : 0.5;
    let sellProb = total > 0 ? sellProbRaw / total : 0.5;

    const atr = this.calculateATR(sequence);
    const rsi = this.calculateRSI(adaPrices.slice(-15));
    const momentum =
      adaPrices.length > 1
        ? (adaPrices[adaPrices.length - 1] - adaPrices[adaPrices.length - 2]) /
          adaPrices[adaPrices.length - 2]
        : 0;
    const volatilityWeight = Math.min(1, 1 / (atr * 100));
    const trendWeight =
      rsi > 60 || momentum < 0 ? 0.6 : rsi < 40 && momentum > 0 ? 1.5 : 1.5;

    buyProb = Math.min(0.9, buyProb * volatilityWeight * trendWeight);
    sellProb = Math.min(0.9, (sellProb * volatilityWeight) / trendWeight);
    const confidence = Math.max(buyProb, sellProb);

    features.dispose();
    featuresNormalized.dispose();
    logits.dispose();
    scaledLogits.dispose();
    buyLogit.dispose();
    sellLogit.dispose();
    means.dispose();
    stds.dispose();

    return { buyProb, sellProb, confidence };
  }

  private calculateATR(sequence: number[][]): number {
    const priceData = sequence.map((row) => row[0]);
    if (priceData.length < 14) return 0.01;
    let atr = 0;
    for (let i = 1; i < 14; i++) {
      atr += Math.abs(priceData[i] - priceData[i - 1]);
    }
    return atr / 14;
  }

  private calculateRSI(prices: number[]): number {
    if (prices.length < 14) return 50;
    let gains = 0,
      losses = 0;
    for (let i = 1; i < prices.length; i++) {
      const diff = prices[i] - prices[i - 1];
      if (diff > 0) gains += diff;
      else losses -= diff;
    }
    const avgGain = gains / 14;
    const avgLoss = losses / 14;
    const rs = avgLoss === 0 ? 100 : avgGain / avgLoss;
    return 100 - 100 / (1 + rs);
  }
}
