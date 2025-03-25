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
    storageBucket: process.env.STORAGE_BUCKET,
  });
}

export default class TradeModelPredictor {
  private bucket = admin.storage().bucket();
  private readonly timesteps = 30;

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
    conditions: string[];
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
    const conditions: string[] = [];

    switch (name) {
      case "TrendFollowing":
        if (
          sma7 > sma21 &&
          prevSma7 <= prevSma21 &&
          sma7 > sma50 &&
          btcSma7 > btcSma21 &&
          btcPrevSma7 <= btcPrevSma21
        ) {
          buyProb += 0.5;
          conditions.push("SMA crossover up + BTC confirmation");
        }
        if (
          sma7 < sma21 &&
          prevSma7 >= prevSma21 &&
          sma7 < sma50 &&
          btcSma7 < btcSma21 &&
          btcPrevSma7 >= btcPrevSma21
        ) {
          sellProb += 0.5;
          conditions.push("SMA crossover down + BTC confirmation");
        }
        if (macdHistogram > 0 && prevMacdHistogram <= 0) {
          buyProb += 0.35;
          conditions.push("MACD bullish crossover");
        }
        if (macdHistogram < 0 && prevMacdHistogram >= 0) {
          sellProb += 0.35;
          conditions.push("MACD bearish crossover");
        }
        if (adxProxy <= 0.02) {
          holdProb += 0.2;
          conditions.push("Low ADX (trend weak)");
        }
        if (momentumSafe < -0.2) {
          sellProb += 0.3;
          conditions.push("Negative momentum");
        }
        if (momentumSafe > 0.2) {
          buyProb += 0.3;
          conditions.push("Positive momentum");
        }
        break;
      case "MeanReversion":
        if (rsiSafe < 20 && currentPrice < sma20 - 2.5 * atr) {
          buyProb += 0.5;
          conditions.push("Oversold RSI + below Bollinger");
        }
        if (rsiSafe > 80 && currentPrice > sma20 + 2.5 * atr) {
          sellProb += 0.5;
          conditions.push("Overbought RSI + above Bollinger");
        }
        if (rsiSafe >= 25 && rsiSafe <= 75) {
          holdProb += 0.2;
          conditions.push("Neutral RSI range");
        }
        break;
      case "Breakout":
        if (currentPrice > recentHigh && isVolumeSpike && volatility > 1.8) {
          buyProb += 0.5;
          conditions.push("Breakout above resistance + volume spike");
        }
        if (currentPrice < recentLow && isVolumeSpike && volatility > 1.8) {
          sellProb += 0.5;
          conditions.push("Breakdown below support + volume spike");
        }
        if (currentPrice <= recentHigh && currentPrice >= recentLow) {
          holdProb += 0.2;
          conditions.push("Price within range");
        }
        if (volatility <= 1.8) {
          holdProb += 0.15;
          conditions.push("Low volatility");
        }
        break;
    }

    const total = buyProb + sellProb + holdProb;
    buyProb /= total;
    sellProb /= total;
    holdProb /= total;

    const recommendation =
      buyProb > 0.5
        ? Recommendation.Buy
        : sellProb > 0.5
        ? Recommendation.Sell
        : Recommendation.Hold;
    return { buyProb, sellProb, holdProb, recommendation, conditions };
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

      const file = this.bucket.file("tradePredictorWeights.json");
      const [weightsData] = await file.download();
      const { weights } = JSON.parse(weightsData.toString("utf8"));

      const featureSequence: number[][] = [];
      for (
        let i = Math.max(0, adaPrices.length - this.timesteps);
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
        featureSequence.push([...adaFeatures, ...btcFeatures]); // 61 features
      }

      while (featureSequence.length < this.timesteps)
        featureSequence.unshift(featureSequence[0]);

      let buyProbBase = 0.5,
        sellProbBase = 0.5;
      await tf.tidy(() => {
        const factory = new TradeModelFactory(this.timesteps, 61);
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
        model.getLayer("lstm1").setWeights([
          tf.tensor2d(weights.lstm1Weights, [24, 512]), // 128 units * 4
          tf.tensor2d(weights.lstm1RecurrentWeights, [128, 512]),
          tf.tensor1d(weights.lstm1Bias),
        ]);
        model.getLayer("lstm2").setWeights([
          tf.tensor2d(weights.lstm2Weights, [128, 256]), // 64 units * 4
          tf.tensor2d(weights.lstm2RecurrentWeights, [64, 256]),
          tf.tensor1d(weights.lstm2Bias),
        ]);
        model.getLayer("lstm3").setWeights([
          tf.tensor2d(weights.lstm3Weights, [64, 64]), // 16 units * 4
          tf.tensor2d(weights.lstm3RecurrentWeights, [16, 64]),
          tf.tensor1d(weights.lstm3Bias),
        ]);
        model
          .getLayer("time_distributed")
          .setWeights([
            tf.tensor2d(weights.timeDistributedWeights, [16, 16]),
            tf.tensor1d(weights.timeDistributedBias),
          ]);
        model
          .getLayer("batchNormalization")
          .setWeights([
            tf.tensor1d(weights.bnGamma),
            tf.tensor1d(weights.bnBeta),
            tf.tensor1d(weights.bnMovingMean),
            tf.tensor1d(weights.bnMovingVariance),
          ]);
        model.getLayer("dense").setWeights([
          tf.tensor2d(weights.dense1Weights, [384, 24]), // Updated from [480, 24]
          tf.tensor1d(weights.dense1Bias),
        ]);
        model
          .getLayer("dense_1")
          .setWeights([
            tf.tensor2d(weights.dense2Weights, [24, 2]),
            tf.tensor1d(weights.dense2Bias),
          ]);
        const featuresNormalized = tf
          .tensor3d([featureSequence], [1, this.timesteps, 61])
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
      const metConditions: string[] = [];

      if (rsiSafe < 20) {
        buyProb += 0.35;
        metConditions.push("RSI < 20 (oversold)");
      }
      if (rsiSafe > 80) {
        sellProb += 0.35;
        metConditions.push("RSI > 80 (overbought)");
      }

      if (rsiSafe < 20) buyProb += 0.35;
      if (rsiSafe > 80) sellProb += 0.35;
      const confidenceThreshold =
        volatility > 1.8
          ? 0.5
          : 0.45 + (adaIndicators.adxProxy > 0.02 ? 0.05 : 0);
      if (buyProb < 0.4 || sellProb < 0.4) {
        holdProb = 1 - (buyProb + sellProb);
        buyProb *= 0.8;
        sellProb *= 0.8;
        metConditions.push("Low confidence in buy/sell");
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

      metConditions.push(...trendResult.conditions);
      metConditions.push(...meanRevResult.conditions);
      metConditions.push(...breakoutResult.conditions);

      const trendWeight = adaIndicators.adxProxy > 0.02 ? 0.5 : 0.3;
      const meanRevWeight = rsiSafe < 20 || rsiSafe > 80 ? 0.45 : 0.25;
      const breakoutWeight =
        volatility > 1.8 || adaIndicators.isVolumeSpike ? 0.45 : 0.25;

      buyProb +=
        trendWeight * trendResult.buyProb +
        meanRevWeight * meanRevResult.buyProb +
        breakoutWeight * breakoutResult.buyProb +
        (profitPotential > 0.2 ? 0.3 : profitPotential > 0.1 ? 0.15 : 0) +
        (momentumSafe > 0.2 ? 0.35 : -0.1);
      sellProb +=
        trendWeight * trendResult.sellProb +
        meanRevWeight * meanRevResult.sellProb +
        breakoutWeight * breakoutResult.sellProb +
        (profitPotential < -0.2 ? 0.3 : profitPotential < -0.1 ? 0.15 : 0) +
        (momentumSafe < -0.3 ? 0.4 : momentumSafe < -0.2 ? 0.25 : 0);

      if (profitPotential > 0.2) {
        buyProb += 0.3;
        metConditions.push("High profit potential (>20%)");
      } else if (profitPotential > 0.1) {
        buyProb += 0.15;
        metConditions.push("Moderate profit potential (>10%)");
      }
      if (profitPotential < -0.2) {
        sellProb += 0.3;
        metConditions.push("Significant loss potential (<-20%)");
      } else if (profitPotential < -0.1) {
        sellProb += 0.15;
        metConditions.push("Moderate loss potential (<-10%)");
      }
      if (momentumSafe > 0.2) {
        buyProb += 0.35;
        metConditions.push("Strong positive momentum");
      } else if (momentumSafe < -0.3) {
        sellProb += 0.4;
        metConditions.push("Strong negative momentum");
      } else if (momentumSafe < -0.2) {
        sellProb += 0.25;
        metConditions.push("Moderate negative momentum");
      }

      const total = buyProb + sellProb + holdProb;
      buyProb = buyProb / total || 0.33;
      sellProb = sellProb / total || 0.33;
      holdProb = holdProb / total || 0.33;

      const recommendation =
        rsiSafe < 20 && buyProb > 0.5
          ? Recommendation.Buy
          : rsiSafe > 80 && sellProb > 0.5
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
      console.log(`Met Conditions: ${metConditions.join(", ")}`);

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
        metConditions,
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
