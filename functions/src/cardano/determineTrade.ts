import dotenv from "dotenv";
import * as admin from "firebase-admin";
import axios from "axios";
import {
  AVERAGE_BUY_PRICE,
  BITCOIN_30_DAY_HISTORICAL_URL,
  CARDANO_30_DAY_HISTORICAL_URL,
  CARDANO_BITCOIN_PRICE_URL,
} from "../constants";
import { predictTrade } from "./predictTrade";
import {
  CoinGeckoMarketChartResponse,
  CoinGeckoPriceResponse,
  Recommendation,
  TradeDecision,
} from "../types";
import { calculateIndicators } from "../calculations/calculateIndicators";

dotenv.config();

export const determineTrade = async (): Promise<TradeDecision> => {
  try {
    const priceResponse = await axios.get<CoinGeckoPriceResponse>(
      CARDANO_BITCOIN_PRICE_URL
    );

    const currentAdaPrice = priceResponse.data.cardano.usd;
    const currentBtcPrice = priceResponse.data.bitcoin.usd;

    const historicalResponse = await axios.get<CoinGeckoMarketChartResponse>(
      CARDANO_30_DAY_HISTORICAL_URL
    );

    const btcResponse = await axios.get<CoinGeckoMarketChartResponse>(
      BITCOIN_30_DAY_HISTORICAL_URL
    );

    const adaPrices = historicalResponse.data.prices.map(([_, price]) => price);
    const adaVolumes = historicalResponse.data.total_volumes.map(
      ([_, vol]) => vol
    );
    const btcPrices = btcResponse.data.prices.map(([_, price]) => price);
    const btcVolumes = btcResponse.data.total_volumes.map(([_, vol]) => vol);

    const adaIndicators = calculateIndicators({
      prices: adaPrices,
      volumes: adaVolumes,
      currentPrice: currentAdaPrice,
    });
    const btcIndicators = calculateIndicators({
      prices: btcPrices,
      volumes: btcVolumes,
      currentPrice: currentBtcPrice,
    });

    const { metConditions, probabilities, recommendation } = await predictTrade(
      {
        rsi: adaIndicators.rsi,
        prevRsi: adaIndicators.prevRsi,
        sma7: adaIndicators.sma7,
        sma21: adaIndicators.sma21,
        prevSma7: adaIndicators.prevSma7,
        prevSma21: adaIndicators.prevSma21,
        macdLine: adaIndicators.macdLine,
        signalLine: adaIndicators.signalLine,
        currentPrice: currentAdaPrice,
        upperBand: adaIndicators.upperBand,
        obvValues: adaIndicators.obvValues,
        atr: adaIndicators.atr,
        atrBaseline: adaIndicators.atrBaseline,
        zScore: adaIndicators.zScore,
        vwap: adaIndicators.vwap,
        stochRsi: adaIndicators.stochRsi,
        prevStochRsi: adaIndicators.prevStochRsi,
        fib61_8: adaIndicators.fib61_8,
        prices: adaPrices,
        volumeOscillator: adaIndicators.volumeOscillator,
        prevVolumeOscillator: adaIndicators.prevVolumeOscillator,
        isDoubleTop: adaIndicators.isDoubleTop,
        isHeadAndShoulders: adaIndicators.isHeadAndShoulders,
        prevMacdLine: adaIndicators.prevMacdLine,
        isTripleTop: adaIndicators.isTripleTop,
        isVolumeSpike: adaIndicators.isVolumeSpike,
        momentum: adaIndicators.momentum,
        priceChangePct: adaIndicators.priceChangePct,
        btcRsi: btcIndicators.rsi,
        btcPrevRsi: btcIndicators.prevRsi,
        btcSma7: btcIndicators.sma7,
        btcSma21: btcIndicators.sma21,
        btcPrevSma7: btcIndicators.prevSma7,
        btcPrevSma21: btcIndicators.prevSma21,
        btcMacdLine: btcIndicators.macdLine,
        btcSignalLine: btcIndicators.signalLine,
        btcCurrentPrice: currentBtcPrice,
        btcUpperBand: btcIndicators.upperBand,
        btcObvValues: btcIndicators.obvValues,
        btcAtr: btcIndicators.atr,
        btcAtrBaseline: btcIndicators.atrBaseline,
        btcZScore: btcIndicators.zScore,
        btcVwap: btcIndicators.vwap,
        btcStochRsi: btcIndicators.stochRsi,
        btcPrevStochRsi: btcIndicators.prevStochRsi,
        btcFib61_8: btcIndicators.fib61_8,
        btcPrices,
        btcVolumeOscillator: btcIndicators.volumeOscillator,
        btcPrevVolumeOscillator: btcIndicators.prevVolumeOscillator,
        btcIsDoubleTop: btcIndicators.isDoubleTop,
        btcIsHeadAndShoulders: btcIndicators.isHeadAndShoulders,
        btcPrevMacdLine: btcIndicators.prevMacdLine,
        btcIsTripleTop: btcIndicators.isTripleTop,
        btcIsVolumeSpike: btcIndicators.isVolumeSpike,
        btcMomentum: btcIndicators.momentum,
        btcPriceChangePct: btcIndicators.priceChangePct,
      }
    );

    const recommendationBasedOnBuyPrice =
      currentAdaPrice < AVERAGE_BUY_PRICE &&
      recommendation === Recommendation.Sell
        ? Recommendation.HoldBasedOnBuyPrice
        : recommendation;

    return {
      currentPrice: currentAdaPrice,
      rsi: adaIndicators.rsi?.toFixed(2),
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
      probabilities,
      recommendation: recommendationBasedOnBuyPrice,
      timestamp: admin.firestore.FieldValue.serverTimestamp(),
    };
  } catch (error: any) {
    console.error("Error in determineTrade:", JSON.stringify(error));
    throw error;
  }
};
