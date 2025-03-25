import * as admin from "firebase-admin";
import * as tf from "@tensorflow/tfjs-node";
import dotenv from "dotenv";
import {
  TradeDecision,
  Indicators,
  TradingStrategy,
  Recommendation,
} from "../types";
import { getCurrentPrices } from "../api/getCurrentPrices";
import { getHistoricalPricesAndVolumes } from "../api/getHistoricalPricesAndVolumes";
import FeatureCalculator from "./FeatureCalculator";
import TradeModelFactory from "./TradeModelFactory";
import { FirebaseService } from "../api/FirebaseService";
import { ModelWeightManager } from "./TradeModelWeightManager";
import { FeatureSequenceGenerator } from "./FeatureSequenceGenerator";

dotenv.config();

FirebaseService.getInstance();

export default class TradeModelPredictor {
  private readonly timesteps = 30;
  private weightManager = new ModelWeightManager();
  private sequenceGenerator = new FeatureSequenceGenerator(this.timesteps);

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
    name: TradingStrategy,
    adaIndicators: Indicators,
    btcIndicators: Indicators,
    sma50: number
  ): {
    buyProb: number;
    sellProb: number;
    holdProb: number;
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

    let buyProb = 0.5, // Increased baseline
      sellProb = 0.5, // Increased baseline
      holdProb = 0.3;
    const conditions: string[] = [];

    switch (name) {
      case TradingStrategy.TrendFollowing:
        if (
          sma7 > sma21 &&
          prevSma7 <= prevSma21 &&
          sma7 > sma50 &&
          isVolumeSpike &&
          momentumSafe > 0.1 &&
          btcSma7 > btcSma21 &&
          btcPrevSma7 <= btcPrevSma21
        ) {
          buyProb += 0.4;
          conditions.push(
            "SMA crossover up with volume and momentum + BTC confirmation"
          );
        }
        if (
          sma7 < sma21 &&
          prevSma7 >= prevSma21 &&
          sma7 < sma50 &&
          isVolumeSpike &&
          momentumSafe < -0.1 &&
          btcSma7 < btcSma21 &&
          btcPrevSma7 >= btcPrevSma21
        ) {
          sellProb += 0.4;
          conditions.push(
            "SMA crossover down with volume and momentum + BTC confirmation"
          );
        }
        if (macdHistogram > 0 && prevMacdHistogram <= 0 && isVolumeSpike) {
          buyProb += 0.3;
          conditions.push("MACD bullish crossover with volume");
        }
        if (macdHistogram < 0 && prevMacdHistogram >= 0 && isVolumeSpike) {
          sellProb += 0.3;
          conditions.push("MACD bearish crossover with volume");
        }
        if (adxProxy <= 0.025) {
          holdProb += 0.25;
          conditions.push("Low ADX (trend weak)");
        }
        break;
      case TradingStrategy.MeanReversion:
        if (
          rsiSafe < 20 &&
          currentPrice < sma20 - 2.5 * atr &&
          momentumSafe > 0.5
        ) {
          buyProb += 0.45;
          conditions.push(
            "Oversold RSI + below Bollinger with strong positive momentum"
          );
        }
        if (
          rsiSafe > 80 &&
          currentPrice > sma20 + 2.5 * atr &&
          momentumSafe < 0
        ) {
          sellProb += 0.45;
          conditions.push(
            "Overbought RSI + above Bollinger with negative momentum"
          );
        } else if (rsiSafe > 70 && momentumSafe < 0) {
          sellProb += 0.35;
          conditions.push("High RSI with negative momentum");
        }
        if (rsiSafe >= 25 && rsiSafe <= 75) {
          holdProb += 0.2;
          conditions.push("Neutral RSI range");
        }
        break;
      case TradingStrategy.Breakout:
        if (currentPrice > recentHigh && isVolumeSpike && volatility > 1.8) {
          buyProb += 0.45;
          conditions.push("Breakout above resistance + volume spike");
        }
        if (currentPrice < recentLow && isVolumeSpike && volatility > 1.8) {
          sellProb += 0.45;
          conditions.push("Breakdown below support + volume spike");
        }
        if (currentPrice <= recentHigh && currentPrice >= recentLow) {
          holdProb += 0.25;
          conditions.push("Price within range");
        }
        break;
    }

    const total = buyProb + sellProb + holdProb || 1;
    buyProb /= total;
    sellProb /= total;
    holdProb /= total;

    return { buyProb, sellProb, holdProb, conditions };
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

      await this.weightManager.loadWeights();
      const factory = new TradeModelFactory(this.timesteps, 61);
      const model = factory.createModel();
      this.weightManager.setWeights(model);

      const featureSequence = this.sequenceGenerator.generateSequence(
        adaPrices,
        adaVolumes,
        btcPrices,
        btcVolumes,
        Math.max(0, adaPrices.length - this.timesteps),
        adaPrices.length - 1
      );

      let buyProbBase = 0.5,
        sellProbBase = 0.5;
      const means = this.weightManager.getFeatureMeans();
      const stds = this.weightManager.getFeatureStds();
      if (means.length !== 61 || stds.length !== 61) {
        throw new Error("Feature means or stds length mismatch");
      }
      const features = tf.tensor3d([featureSequence], [1, this.timesteps, 61]);
      const featuresNormalized = features
        .sub(tf.tensor1d(means))
        .div(tf.tensor1d(stds));
      const logits = model.predict(featuresNormalized) as tf.Tensor2D;
      console.log(`Raw Logits: ${logits.dataSync()}`);
      const scaledLogits = logits.mul(tf.scalar(2)); // Double logits for wider range
      const buyLogit = scaledLogits.slice([0, 1], [1, 1]).sigmoid();
      const sellLogit = scaledLogits.slice([0, 0], [1, 1]).sigmoid();
      buyProbBase = Math.min(buyLogit.dataSync()[0], 0.9);
      sellProbBase = Math.min(sellLogit.dataSync()[0], 0.9);
      const totalBase = buyProbBase + sellProbBase;
      if (totalBase > 0) {
        buyProbBase /= totalBase;
        sellProbBase /= totalBase;
      }
      console.log(
        `Pre-Adjusted Probs - Buy: ${buyProbBase.toFixed(
          3
        )}, Sell: ${sellProbBase.toFixed(3)}`
      );
      features.dispose();
      featuresNormalized.dispose();
      logits.dispose();
      scaledLogits.dispose();
      buyLogit.dispose();
      sellLogit.dispose();

      const rsiSafe = adaIndicators.rsi ?? 50;
      const momentumSafe = adaIndicators.momentum ?? 0;
      const isUptrend = adaIndicators.sma7 > sma50;
      const volatility = adaIndicators.atr / adaIndicators.atrBaseline;

      let buyProb = buyProbBase,
        sellProb = sellProbBase,
        holdProb = 0;
      const metConditions: string[] = [];

      if (!isUptrend && (rsiSafe >= 20 || momentumSafe <= 0.5)) {
        buyProb = 0;
        holdProb = 0.6;
        metConditions.push(
          "Downtrend (SMA7 < SMA50), RSI not oversold or weak momentum"
        );
      }
      if (rsiSafe > 70 || (adaIndicators.sma7 < sma50 && momentumSafe < 0)) {
        sellProb = 0.9;
        buyProb = 0;
        holdProb = 0.1;
        metConditions.push(
          "Sell override: RSI > 70 or Downtrend with negative momentum"
        );
      }
      if (rsiSafe < 20 && momentumSafe > 0.5) {
        buyProb = 0.9;
        sellProb = 0;
        holdProb = 0.1;
        metConditions.push(
          "Buy override: RSI < 20 with strong positive momentum"
        );
      }

      const trendResult = this.evaluateStrategy(
        TradingStrategy.TrendFollowing,
        adaIndicators,
        btcIndicators,
        sma50
      );
      const meanRevResult = this.evaluateStrategy(
        TradingStrategy.MeanReversion,
        adaIndicators,
        btcIndicators,
        sma50
      );
      const breakoutResult = this.evaluateStrategy(
        TradingStrategy.Breakout,
        adaIndicators,
        btcIndicators,
        sma50
      );

      metConditions.push(...trendResult.conditions);
      metConditions.push(...meanRevResult.conditions);
      metConditions.push(...breakoutResult.conditions);

      const trendWeight = adaIndicators.adxProxy > 0.025 ? 0.6 : 0.4;
      const meanRevWeight =
        volatility > 1.5 || rsiSafe < 20 || rsiSafe > 70 ? 0.7 : 0.3;
      const breakoutWeight =
        volatility > 1.8 || adaIndicators.isVolumeSpike ? 0.7 : 0.3;

      console.log(
        `Strategy Weights - Trend: ${trendWeight.toFixed(
          2
        )}, MeanRev: ${meanRevWeight.toFixed(
          2
        )}, Breakout: ${breakoutWeight.toFixed(2)}`
      );
      console.log(
        `Strategy Contributions - Trend: Buy=${trendResult.buyProb.toFixed(
          3
        )}, Sell=${trendResult.sellProb.toFixed(
          3
        )}, MeanRev: Buy=${meanRevResult.buyProb.toFixed(
          3
        )}, Sell=${meanRevResult.sellProb.toFixed(
          3
        )}, Breakout: Buy=${breakoutResult.buyProb.toFixed(
          3
        )}, Sell=${breakoutResult.sellProb.toFixed(3)}`
      );

      buyProb +=
        trendWeight * trendResult.buyProb +
        meanRevWeight * meanRevResult.buyProb +
        breakoutWeight * breakoutResult.buyProb;
      sellProb +=
        trendWeight * trendResult.sellProb +
        meanRevWeight * meanRevResult.sellProb +
        breakoutWeight * breakoutResult.sellProb;

      buyProb = Math.min(Math.max(buyProb, 0), 0.9);
      sellProb = Math.min(Math.max(sellProb, 0), 0.9);
      holdProb = 1 - buyProb - sellProb;

      const confidenceThreshold = 0.55; // Lowered to match model output
      const recommendation =
        buyProb > confidenceThreshold && momentumSafe > 0 && isUptrend
          ? Recommendation.Buy
          : sellProb > confidenceThreshold
          ? Recommendation.Sell
          : Recommendation.Hold;

      console.log(
        `Final Probabilities - Buy: ${buyProb.toFixed(
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
        profitPotential: 0,
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
