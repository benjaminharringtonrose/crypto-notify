import * as admin from "firebase-admin";
import * as tf from "@tensorflow/tfjs-node";
import dotenv from "dotenv";
import { TradeDecision, Indicators, Recommendation } from "../types";
import { getCurrentPrices } from "../api/getCurrentPrices";
import { getHistoricalPricesAndVolumes } from "../api/getHistoricalPricesAndVolumes";
import FeatureCalculator from "./FeatureCalculator";
import TradeModelFactory from "./TradeModelFactory";

dotenv.config();

if (!admin.apps.length) {
  admin.initializeApp({
    credential: admin.credential.cert(require("../../../serviceAccount.json")),
    storageBucket:
      process.env.STORAGE_BUCKET || "crypto-notify-ee5bc.firebasestorage.app",
  });
}
const bucket = admin.storage().bucket();

export default class TradeModelPredictor {
  private async fetchAndCalculateIndicators() {
    const { currentAdaPrice, currentBtcPrice } = await getCurrentPrices();
    const { adaPrices, adaVolumes, btcPrices, btcVolumes } =
      await getHistoricalPricesAndVolumes();
    const adaCalculator = new FeatureCalculator({
      prices: adaPrices,
      volumes: adaVolumes,
      dayIndex: adaPrices.length - 1,
      currentPrice: currentAdaPrice,
      isBTC: false,
    });
    const btcCalculator = new FeatureCalculator({
      prices: btcPrices,
      volumes: btcVolumes,
      dayIndex: btcPrices.length - 1,
      currentPrice: currentBtcPrice,
      isBTC: true,
    });
    return {
      adaIndicators: adaCalculator.calculateIndicators(),
      btcIndicators: btcCalculator.calculateIndicators(),
      adaPrices,
      adaVolumes,
      btcPrices,
      btcVolumes,
    };
  }

  private evaluateStrategy(
    name: string,
    adaIndicators: Indicators,
    btcIndicators: Indicators,
    sma50: number
  ): {
    buyProb: number;
    sellProb: number;
    holdProb: number;
    recommendation: Recommendation;
  } {
    const {
      rsi,
      sma7,
      sma21,
      prevSma7,
      prevSma21,
      macdLine,
      signalLine,
      currentPrice,
      prices,
      atr,
      atrBaseline,
      isVolumeSpike,
      prevMacdLine,
      adxProxy,
      momentum,
    } = adaIndicators;
    const {
      sma7: btcSma7,
      sma21: btcSma21,
      prevSma7: btcPrevSma7,
      prevSma21: btcPrevSma21,
    } = btcIndicators;
    const sma20 = prices.slice(-20).reduce((a, b) => a + b, 0) / 20;
    const macdHistogram = macdLine - signalLine;
    const prevMacdHistogram = prevMacdLine - signalLine;
    const rsiSafe = rsi !== undefined ? rsi : 50;
    const volatility = atr / atrBaseline;
    const recentHigh = Math.max(...prices.slice(-10));
    const recentLow = Math.min(...prices.slice(-10));
    const momentumSafe = momentum ?? 0;

    let buyProb = 0.3,
      sellProb = 0.3,
      holdProb = 0.3;

    switch (name) {
      case "TrendFollowing":
        if (
          sma7 > sma21 &&
          prevSma7 <= prevSma21 &&
          sma7 > sma50 &&
          btcSma7 > btcSma21 &&
          btcPrevSma7 <= btcPrevSma21
        )
          buyProb += 0.45;
        if (
          sma7 < sma21 &&
          prevSma7 >= prevSma21 &&
          sma7 < sma50 &&
          btcSma7 < btcSma21 &&
          btcPrevSma7 >= btcPrevSma21
        )
          sellProb += 0.45;
        if (macdHistogram > 0 && prevMacdHistogram <= 0) buyProb += 0.3;
        if (macdHistogram < 0 && prevMacdHistogram >= 0) sellProb += 0.3;
        if (adxProxy <= 0.02) holdProb += 0.15;
        if (momentumSafe < -0.15) sellProb += 0.25;
        if (momentumSafe > 0.15) buyProb += 0.25;
        break;
      case "MeanReversion":
        if (rsiSafe < 20 && currentPrice < sma20 - 2.5 * atr) buyProb += 0.45; // Tightened RSI and range
        if (rsiSafe > 80 && currentPrice > sma20 + 2.5 * atr) sellProb += 0.45;
        if (rsiSafe >= 25 && rsiSafe <= 75) holdProb += 0.15;
        break;
      case "Breakout":
        if (currentPrice > recentHigh && isVolumeSpike && volatility > 1.8)
          buyProb += 0.45;
        if (currentPrice < recentLow && isVolumeSpike && volatility > 1.8)
          sellProb += 0.45;
        if (currentPrice <= recentHigh && currentPrice >= recentLow)
          holdProb += 0.15;
        if (volatility <= 1.8) holdProb += 0.1;
        break;
    }

    const total = buyProb + sellProb + holdProb;
    buyProb /= total;
    sellProb /= total;
    holdProb /= total;

    const recommendation =
      buyProb > 0.45
        ? Recommendation.Buy
        : sellProb > 0.45
        ? Recommendation.Sell
        : Recommendation.Hold; // Raised threshold
    return { buyProb, sellProb, holdProb, recommendation };
  }

  public async predict(): Promise<TradeDecision> {
    try {
      const {
        adaIndicators: adaIndRaw,
        btcIndicators: btcIndRaw,
        adaPrices,
        adaVolumes,
        btcPrices,
        btcVolumes,
      } = await this.fetchAndCalculateIndicators();
      const adaIndicators: Indicators = { ...adaIndRaw, prices: adaPrices };
      const btcIndicators: Indicators = { ...btcIndRaw, prices: btcPrices };
      const sma50 =
        adaPrices.length >= 50
          ? adaPrices.slice(-50).reduce((a, b) => a + b, 0) / 50
          : adaIndicators.sma20;

      const file = bucket.file("tradePredictorWeights.json");
      const [weightsData] = await file.download();
      const { weights } = JSON.parse(weightsData.toString("utf8"));

      const timesteps = 20;
      const featureSequence: number[][] = [];
      for (
        let i = Math.max(0, adaPrices.length - timesteps);
        i < adaPrices.length;
        i++
      ) {
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
        const adaBtcRatio = adaPrices[i] / btcPrices[i];
        featureSequence.push([...adaFeatures, ...btcFeatures, adaBtcRatio]);
      }
      while (featureSequence.length < timesteps)
        featureSequence.unshift(featureSequence[0]);

      let buyProbBase = 0.5,
        sellProbBase = 0.5;
      await tf.tidy(() => {
        const factory = new TradeModelFactory(timesteps, 61);
        const model = factory.createModel();
        model
          .getLayer("conv1d")
          .setWeights([
            tf.tensor3d(weights.conv1Weights, [5, 61, 12]),
            tf.tensor1d(weights.conv1Bias),
          ]);
        model
          .getLayer("conv1d_2")
          .setWeights([
            tf.tensor3d(weights.conv2Weights, [3, 12, 24]),
            tf.tensor1d(weights.conv2Bias),
          ]);
        model
          .getLayer("lstm1")
          .setWeights([
            tf.tensor2d(weights.lstm1Weights, [24, 256]),
            tf.tensor2d(weights.lstm1RecurrentWeights, [64, 256]),
            tf.tensor1d(weights.lstm1Bias),
          ]);
        model
          .getLayer("lstm2")
          .setWeights([
            tf.tensor2d(weights.lstm2Weights, [64, 128]),
            tf.tensor2d(weights.lstm2RecurrentWeights, [32, 128]),
            tf.tensor1d(weights.lstm2Bias),
          ]);
        model
          .getLayer("lstm3")
          .setWeights([
            tf.tensor2d(weights.lstm3Weights, [32, 64]),
            tf.tensor2d(weights.lstm3RecurrentWeights, [16, 64]),
            tf.tensor1d(weights.lstm3Bias),
          ]);
        model
          .getLayer("batchNormalization")
          .setWeights([
            tf.tensor1d(weights.bnGamma),
            tf.tensor1d(weights.bnBeta),
            tf.tensor1d(weights.bnMovingMean),
            tf.tensor1d(weights.bnMovingVariance),
          ]);
        model
          .getLayer("dense")
          .setWeights([
            tf.tensor2d(weights.dense1Weights, [224, 24]),
            tf.tensor1d(weights.dense1Bias),
          ]);
        model
          .getLayer("dense_1")
          .setWeights([
            tf.tensor2d(weights.dense2Weights, [24, 2]),
            tf.tensor1d(weights.dense2Bias),
          ]);
        const featuresNormalized = tf
          .tensor3d([featureSequence], [1, timesteps, 61])
          .sub(tf.tensor1d(weights.featureMeans))
          .div(tf.tensor1d(weights.featureStds));
        const prediction = model.predict(featuresNormalized) as tf.Tensor2D;
        const probs = prediction.dataSync() as Float32Array;
        if (probs.length === 2 && probs.every((v) => Number.isFinite(v)))
          [sellProbBase, buyProbBase] = probs;
      });

      const volatility = adaIndicators.atr / adaIndicators.atrBaseline;
      const rsiSafe = adaIndicators.rsi ?? 50;
      const profitPotential =
        (adaIndicators.currentPrice -
          adaPrices.slice(-5).reduce((a, b) => a + b, 0) / 5) /
        (adaPrices.slice(-5).reduce((a, b) => a + b, 0) / 5);
      const momentumSafe = adaIndicators.momentum ?? 0;

      let buyProb = buyProbBase,
        sellProb = sellProbBase,
        holdProb = 0;
      if (rsiSafe < 20) buyProb += 0.3; // Tightened from 25
      if (rsiSafe > 80) sellProb += 0.3; // Tightened from 75
      const confidenceThreshold =
        volatility > 1.8
          ? 0.4
          : 0.35 + (adaIndicators.adxProxy > 0.02 ? 0.05 : 0); // Adjusted for backtesting
      if (buyProb < confidenceThreshold && sellProb < confidenceThreshold) {
        holdProb = 1 - (buyProb + sellProb);
        buyProb *= 0.85;
        sellProb *= 0.85;
      }

      const trendResult = this.evaluateStrategy(
        "TrendFollowing",
        adaIndicators,
        btcIndicators,
        sma50
      );
      const meanRevResult = this.evaluateStrategy(
        "MeanReversion",
        adaIndicators,
        btcIndicators,
        sma50
      );
      const breakoutResult = this.evaluateStrategy(
        "Breakout",
        adaIndicators,
        btcIndicators,
        sma50
      );

      const trendWeight = adaIndicators.adxProxy > 0.02 ? 0.45 : 0.25;
      const meanRevWeight = rsiSafe < 20 || rsiSafe > 80 ? 0.4 : 0.2;
      const breakoutWeight =
        volatility > 1.8 || adaIndicators.isVolumeSpike ? 0.4 : 0.2;

      buyProb +=
        trendWeight * trendResult.buyProb +
        meanRevWeight * meanRevResult.buyProb +
        breakoutWeight * breakoutResult.buyProb +
        (profitPotential > 0.15 ? 0.25 : profitPotential > 0.07 ? 0.15 : 0) +
        (momentumSafe > 0.15 ? 0.3 : -0.1);
      sellProb +=
        trendWeight * trendResult.sellProb +
        meanRevWeight * meanRevResult.sellProb +
        breakoutWeight * breakoutResult.sellProb +
        (profitPotential < -0.15 ? 0.25 : profitPotential < -0.07 ? 0.15 : 0) +
        (momentumSafe < -0.25 ? 0.35 : momentumSafe < -0.15 ? 0.2 : 0);

      const total = buyProb + sellProb + holdProb;
      buyProb = buyProb / total || 0.33;
      sellProb = sellProb / total || 0.33;
      holdProb = holdProb / total || 0.33;

      const recommendation =
        rsiSafe < 20 && buyProb > 0.45
          ? Recommendation.Buy
          : rsiSafe > 80 && sellProb > 0.45
          ? Recommendation.Sell
          : buyProb > confidenceThreshold && momentumSafe > 0
          ? Recommendation.Buy
          : sellProb > confidenceThreshold
          ? Recommendation.Sell
          : Recommendation.Hold;

      console.log(
        `Adjusted Probabilities - Buy: ${buyProb.toFixed(
          3
        )}, Sell: ${sellProb.toFixed(3)}, Hold: ${holdProb.toFixed(
          3
        )}, Confidence Threshold: ${confidenceThreshold.toFixed(3)}`
      );
      console.log(`Recommendation: ${recommendation}`);

      return {
        currentPrice: adaIndicators.currentPrice,
        rsi: rsiSafe.toFixed(2),
        sma7: adaIndicators.sma7.toFixed(2),
        sma21: adaIndicators.sma21.toFixed(2),
        macdLine: adaIndicators.macdLine.toFixed(2),
        signalLine: adaIndicators.signalLine.toFixed(2),
        sma20: adaIndicators.sma20.toFixed(2),
        upperBand: adaIndicators.upperBand.toFixed(2),
        lowerBand: adaIndicators.lowerBand.toFixed(2),
        obv: adaIndicators.obv.toFixed(0),
        atr: adaIndicators.atr.toFixed(2),
        zScore: adaIndicators.zScore.toFixed(2),
        vwap: adaIndicators.vwap.toFixed(2),
        stochRsi: adaIndicators.stochRsi.toFixed(2),
        stochRsiSignal: adaIndicators.stochRsiSignal.toFixed(2),
        fib61_8: adaIndicators.fib61_8.toFixed(2),
        volumeOscillator: adaIndicators.volumeOscillator.toFixed(2),
        metConditions: [],
        probabilities: { buy: buyProb, hold: holdProb, sell: sellProb },
        recommendation,
        timestamp: admin.firestore.FieldValue.serverTimestamp(),
        profitPotential,
        isDoubleTop: adaIndicators.isDoubleTop,
        isTripleTop: adaIndicators.isTripleTop,
        isHeadAndShoulders: adaIndicators.isHeadAndShoulders,
        isTripleBottom: adaIndicators.isTripleBottom,
        momentum: adaIndicators.momentum,
      };
    } catch (error: any) {
      console.error("Error in TradePredictor:", error.message, error.stack);
      throw error;
    }
  }
}
