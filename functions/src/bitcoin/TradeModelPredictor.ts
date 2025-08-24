import * as tf from "@tensorflow/tfjs-node";
import { ModelWeightManager } from "./TradeModelWeightManager";
import TradeModelFactory from "./TradeModelFactory";
import { FeatureSequenceGenerator } from "./FeatureSequenceGenerator";
import { FirebaseService } from "../api/FirebaseService";
import { MODEL_CONFIG, PERIODS } from "../constants";
import { FeatureDetector } from "./FeatureDetector";

export class TradeModelPredictor {
  private weightManager: ModelWeightManager;
  private sequenceGenerator: FeatureSequenceGenerator;
  private timesteps = MODEL_CONFIG.TIMESTEPS;
  private model: tf.LayersModel;
  private isWeightsLoaded = false;

  constructor() {
    this.weightManager = new ModelWeightManager();
    this.sequenceGenerator = new FeatureSequenceGenerator(this.timesteps);
    FirebaseService.getInstance();

    // Initialize with a placeholder model - will be recreated with correct feature count
    const factory = new TradeModelFactory(
      MODEL_CONFIG.TIMESTEPS,
      36 // Temporary - will be updated when feature detection completes
    );
    this.model = factory.createModel();
    this.initializeAsync();
  }

  /**
   * Asynchronous initialization with dynamic feature detection
   */
  private async initializeAsync(): Promise<void> {
    try {
      // Detect feature count dynamically
      const featureCount = await FeatureDetector.detectFeatureCount();

      // Recreate model with correct feature count
      const factory = new TradeModelFactory(
        MODEL_CONFIG.TIMESTEPS,
        featureCount
      );
      this.model = factory.createModel();

      console.log(
        `🔍 TradeModelPredictor: Initialized with ${featureCount} features`
      );

      // Load weights after model is properly initialized
      await this.loadWeightsAsync();
    } catch (error) {
      console.error("Failed to initialize TradeModelPredictor:", error);
      throw error;
    }
  }

  private async loadWeightsAsync(): Promise<void> {
    try {
      await this.weightManager.loadWeights();
      this.weightManager.setWeights(this.model);
      this.isWeightsLoaded = true;
    } catch (error) {
      console.error("Failed to load weights:", error);
      throw error;
    }
  }

  public async predict(
    btcPrices: number[],
    btcVolumes: number[]
  ): Promise<{
    buyLogit: number;
    sellLogit: number;
    buyProb: number;
    sellProb: number;
    confidence: number;
    momentum: number;
    shortMomentum: number;
    trendSlope: number;
    atr: number;
    momentumDivergence: number;
    volatilityAdjustedMomentum: number;
    trendStrength: number;
    atrBreakout: number;
  }> {
    if (!this.isWeightsLoaded) {
      await this.loadWeightsAsync();
    }

    // Use the existing model instance with loaded weights
    if (!this.model) {
      throw new Error("Model not initialized");
    }

    // Debug: Log input data to see if it's changing
    console.log(
      `DEBUG: Input prices length: ${
        btcPrices.length
      }, last 3 prices: [${btcPrices
        .slice(-3)
        .map((p) => p.toFixed(2))
        .join(", ")}]`
    );
    console.log(
      `DEBUG: Input volumes length: ${
        btcVolumes.length
      }, last 3 volumes: [${btcVolumes
        .slice(-3)
        .map((v) => v.toFixed(2))
        .join(", ")}]`
    );

    // const startTime = performance.now();
    const startIndex = Math.max(0, btcPrices.length - this.timesteps - 1);
    const endIndex = btcPrices.length - 1;
    const sequence = this.sequenceGenerator
      .generateSequence(btcPrices, btcVolumes, startIndex, endIndex)
      .slice(-this.timesteps);

    // Debug: Log sequence data
    console.log(
      `DEBUG: Sequence shape: [${sequence.length}, ${sequence[0]?.length || 0}]`
    );
    console.log(
      `DEBUG: Last sequence features: [${sequence[sequence.length - 1]
        ?.slice(0, 5)
        .map((f) => f.toFixed(4))
        .join(", ")}...]`
    );

    // Dynamic feature count validation
    const expectedFeatureCount = FeatureDetector.getFeatureCount();

    if (
      sequence.length !== MODEL_CONFIG.TIMESTEPS ||
      sequence[0].length !== expectedFeatureCount
    ) {
      throw new Error(
        `Sequence shape mismatch: expected [${
          MODEL_CONFIG.TIMESTEPS
        }, ${expectedFeatureCount}], got [${sequence.length}, ${
          sequence[0]?.length || 0
        }]`
      );
    }

    const features = tf.tensor3d(
      [sequence],
      [1, MODEL_CONFIG.TIMESTEPS, expectedFeatureCount] // Dynamic feature count!
    );
    const means = tf.tensor1d(this.weightManager.getFeatureMeans());
    const stds = tf.tensor1d(this.weightManager.getFeatureStds());
    const featuresNormalized = features.sub(means).div(stds.add(1e-6));

    const logits = this.model.predict(featuresNormalized) as tf.Tensor2D;
    const logitsArray = await logits.data();
    const [sellLogit, buyLogit] = [logitsArray[0], logitsArray[1]];

    // Use natural softmax probabilities without any calibration
    const probs = tf.tensor2d([[sellLogit, buyLogit]]).softmax();
    const probArray = await probs.data();
    const [sellProb, buyProb] = [probArray[0], probArray[1]];

    const confidence = Math.max(buyProb, sellProb);

    // Manual variance calculation
    // const meanProb = tf.mean(probs).dataSync()[0];
    // const variance = tf.mean(tf.square(probs.sub(meanProb))).dataSync()[0];

    const atr = sequence[sequence.length - 1][11];
    const momentumWindowSize =
      atr > 0.05 ? 5 : atr > 0.02 ? 10 : atr > 0.01 ? 14 : 20;
    const momentumWindow = btcPrices.slice(-momentumWindowSize);
    const momentum =
      momentumWindow.length >= 2
        ? (momentumWindow[momentumWindow.length - 1] - momentumWindow[0]) /
          momentumWindow[0]
        : 0;

    const shortMomentumWindow = btcPrices.slice(-3);
    const shortMomentum =
      shortMomentumWindow.length >= 2
        ? (shortMomentumWindow[shortMomentumWindow.length - 1] -
            shortMomentumWindow[0]) /
          shortMomentumWindow[0]
        : 0;

    const trendWindow = btcPrices.slice(-momentumWindowSize);
    const trendSlope =
      trendWindow.length >= 2
        ? (trendWindow[trendWindow.length - 1] - trendWindow[0]) /
          (trendWindow.length - 1)
        : 0;

    const momentumDivergence = shortMomentum - momentum;
    const volatilityAdjustedMomentum = momentum / (atr || 0.01);
    const trendStrength = trendSlope * volatilityAdjustedMomentum;

    const atrWindow = sequence.slice(-PERIODS.ATR_BREAKOUT).map((s) => s[11]);
    const atrSma =
      atrWindow.length >= PERIODS.ATR_BREAKOUT
        ? atrWindow.reduce((sum, val) => sum + val, 0) / atrWindow.length
        : atr;
    const atrBreakout = atr / atrSma;

    // Debug logging for confidence analysis
    console.log(
      `Raw Logits: [Sell: ${sellLogit.toFixed(4)}, Buy: ${buyLogit.toFixed(4)}]`
    );
    console.log(
      `Raw Probs: [Sell: ${sellProb.toFixed(4)}, Buy: ${buyProb.toFixed(4)}]`
    );
    console.log(
      `Confidence: ${confidence.toFixed(4)}, Momentum: ${momentum.toFixed(
        4
      )}, Short Momentum: ${shortMomentum.toFixed(
        4
      )}, Trend Slope: ${trendSlope.toFixed(
        4
      )}, Momentum Divergence: ${momentumDivergence.toFixed(
        4
      )}, Vol-Adj Momentum: ${volatilityAdjustedMomentum.toFixed(
        4
      )}, Trend Strength: ${trendStrength.toFixed(
        4
      )}, ATR Breakout: ${atrBreakout.toFixed(4)}`
    );

    features.dispose();
    featuresNormalized.dispose();
    logits.dispose();
    probs.dispose();
    means.dispose();
    stds.dispose();

    return {
      buyLogit,
      sellLogit,
      buyProb,
      sellProb,
      confidence,
      momentum,
      shortMomentum,
      trendSlope,
      atr,
      momentumDivergence,
      volatilityAdjustedMomentum,
      trendStrength,
      atrBreakout,
    };
  }
}
