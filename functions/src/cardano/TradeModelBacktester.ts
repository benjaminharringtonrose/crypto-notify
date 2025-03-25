import * as admin from "firebase-admin";
import * as tf from "@tensorflow/tfjs-node";
import { getHistoricalData } from "../api/getHistoricalData";
import { FIVE_YEARS_IN_DAYS } from "../constants";
import { BacktestResult, Trade } from "../types";
import serviceAccount from "../../../serviceAccount.json";
import TradeModelFactory from "./TradeModelFactory";
import FeatureCalculator from "./FeatureCalculator";
import { Bucket } from "@google-cloud/storage";

export class TradeModelBacktester {
  private TRANSACTION_FEE = 0.002;
  private INITIAL_USD = 1000;
  private MIN_TRADE_USD = 100;
  private CASH_RESERVE = 25;
  private STOP_LOSS_THRESHOLD = -0.05;
  private TAKE_PROFIT_THRESHOLD = 0.1;

  private startDaysAgo: number;
  private endDaysAgo: number;
  private stepDays: number;
  private bucket: Bucket;
  private readonly timesteps = 30;

  constructor(
    startDaysAgo: number = 450,
    endDaysAgo: number = 1,
    stepDays: number = 1
  ) {
    this.startDaysAgo = startDaysAgo;
    this.endDaysAgo = endDaysAgo;
    this.stepDays = stepDays;

    if (!admin.apps.length) {
      admin.initializeApp({
        credential: admin.credential.cert(serviceAccount as any),
        storageBucket:
          process.env.STORAGE_BUCKET ||
          "crypto-notify-ee5bc.firebasestorage.app",
      });
    }
    this.bucket = admin.storage().bucket();
  }

  private async fetchHistoricalData(): Promise<{
    adaPrices: number[];
    adaVolumes: number[];
    btcPrices: number[];
    btcVolumes: number[];
  }> {
    const { prices: adaPrices, volumes: adaVolumes } = await getHistoricalData(
      "ADA",
      Math.min(this.startDaysAgo, FIVE_YEARS_IN_DAYS)
    );
    const { prices: btcPrices, volumes: btcVolumes } = await getHistoricalData(
      "BTC",
      Math.min(this.startDaysAgo, FIVE_YEARS_IN_DAYS)
    );
    return { adaPrices, adaVolumes, btcPrices, btcVolumes };
  }

  private async batchPredict(
    adaPrices: number[],
    adaVolumes: number[],
    btcPrices: number[],
    btcVolumes: number[],
    startIndex: number,
    endIndex: number
  ): Promise<number[][]> {
    const file = this.bucket.file("tradePredictorWeights.json");
    const [weightsData] = await file.download();
    const { weights } = JSON.parse(weightsData.toString("utf8"));

    const sequences: number[][][] = [];
    const indices: number[] = [];

    const expectedSequences = Math.floor(
      (endIndex - startIndex) / this.stepDays
    );
    console.log(
      `Expected sequences: ${expectedSequences}, startIndex: ${startIndex}, endIndex: ${endIndex}`
    );

    for (let i = startIndex; i < endIndex; i += this.stepDays) {
      const sequence: number[][] = [];
      for (let j = Math.max(0, i - this.timesteps + 1); j <= i; j++) {
        if (j >= adaPrices.length || j >= btcPrices.length) {
          console.log(`Index out of bounds at j=${j}, i=${i}`);
          break;
        }
        const adaFeatures = new FeatureCalculator({
          prices: adaPrices,
          volumes: adaVolumes,
          dayIndex: j,
          currentPrice: adaPrices[j],
          btcPrice: btcPrices[j],
          isBTC: false,
        }).compute();
        const btcFeatures = new FeatureCalculator({
          prices: btcPrices,
          volumes: btcVolumes,
          dayIndex: j,
          currentPrice: btcPrices[j],
          isBTC: true,
        }).compute();
        const combinedFeatures = [...adaFeatures, ...btcFeatures];
        if (combinedFeatures.length !== 61) {
          console.log(
            `Feature count mismatch at i=${i}, j=${j}: expected 61, got ${combinedFeatures.length}, ` +
              `adaFeatures=${adaFeatures.length}, btcFeatures=${btcFeatures.length}`
          );
        }
        sequence.push(combinedFeatures); // 61 features expected
      }
      while (sequence.length < this.timesteps) {
        sequence.unshift(sequence[0]); // Padding
      }
      if (sequence.length !== this.timesteps) {
        console.log(`Invalid sequence length at i=${i}: ${sequence.length}`);
      }
      sequences.push(sequence);
      indices.push(i);
    }

    console.log(`Generated sequences: ${sequences.length}`);
    if (sequences.length !== expectedSequences) {
      console.log(
        `Sequence count mismatch: expected ${expectedSequences}, got ${sequences.length}`
      );
      while (sequences.length < expectedSequences) {
        console.log(`Padding sequence at index ${sequences.length}`);
        sequences.push(sequences[sequences.length - 1]);
        indices.push(indices[indices.length - 1] + this.stepDays);
      }
    }

    return tf.tidy(() => {
      const factory = new TradeModelFactory(this.timesteps, 61);
      const model = factory.createModel();
      model
        .getLayer("conv1d")
        .setWeights([
          tf.tensor3d(weights.conv1Weights, [5, 61, 12]),
          tf.tensor1d(weights.conv1Bias),
        ]);
      // ... (rest of the weight setting remains unchanged)
      const featuresNormalized = tf
        .tensor3d(sequences, [sequences.length, this.timesteps, 61])
        .sub(tf.tensor1d(weights.featureMeans))
        .div(tf.tensor1d(weights.featureStds));
      const predictions = model.predict(featuresNormalized) as tf.Tensor2D;
      return predictions.arraySync() as number[][];
    });
  }

  private calculateSharpeRatio(returns: number[]): number {
    const meanReturn = returns.reduce((a, b) => a + b, 0) / returns.length;
    const stdDev =
      Math.sqrt(
        returns.reduce((a, b) => a + Math.pow(b - meanReturn, 2), 0) /
          (returns.length - 1)
      ) || 1;
    return (meanReturn / stdDev) * Math.sqrt(365);
  }

  private calculateMaxDrawdown(portfolioValues: number[]): number {
    let maxDD = 0;
    let peak = portfolioValues[0];
    for (const value of portfolioValues) {
      if (value > peak) peak = value;
      const dd = (peak - value) / peak;
      if (dd > maxDD) maxDD = dd;
    }
    return maxDD * 100;
  }

  public async run(): Promise<BacktestResult> {
    console.log("Starting backtest...");

    const { adaPrices, adaVolumes, btcPrices, btcVolumes } =
      await this.fetchHistoricalData();
    console.log(
      `ADA data points: ${adaPrices.length}, BTC data points: ${btcPrices.length}`
    );

    const startIndex = Math.max(0, adaPrices.length - this.startDaysAgo);
    const endIndex = Math.max(0, adaPrices.length - this.endDaysAgo);
    const backtestStart = this.timesteps + 4; // Adjust for longer timesteps
    const backtestEnd = endIndex - 5;

    const predictions = await this.batchPredict(
      adaPrices,
      adaVolumes,
      btcPrices,
      btcVolumes,
      backtestStart,
      backtestEnd
    );

    let usdBalance = this.INITIAL_USD;
    let adaBalance = 0;
    let avgBuyPrice = 0;
    let peakPrice = 0;
    let cooldown = 0;

    const trades: Trade[] = [];
    const portfolioHistory: { timestamp: string; value: number }[] = [];
    const dailyReturns: number[] = [];
    let wins = 0;
    let completedCycles = 0;

    let predIndex = 0;
    for (let i = backtestStart; i < backtestEnd; i += this.stepDays) {
      const currentAdaPrice = adaPrices[i];
      const timestamp = new Date(
        Date.now() - (adaPrices.length - startIndex - i) * 24 * 60 * 60 * 1000
      ).toISOString();
      const [sellProb, buyProb] = predictions[predIndex++];

      const confidence = Math.max(buyProb, sellProb);
      const portfolioValue = usdBalance + adaBalance * currentAdaPrice;
      const unrealizedProfit =
        adaBalance > 0 ? (currentAdaPrice - avgBuyPrice) / avgBuyPrice : 0;
      const atr =
        adaPrices
          .slice(Math.max(0, i - 14), i + 1)
          .reduce(
            (a, b, idx, arr) => a + (idx > 0 ? Math.abs(b - arr[idx - 1]) : 0),
            0
          ) / 14;
      const trailingStopPrice =
        adaBalance > 0 ? peakPrice * (1 - (1.5 * atr) / currentAdaPrice) : 0;
      const trailingStopTriggered =
        adaBalance > 0 && currentAdaPrice <= trailingStopPrice;
      if (adaBalance > 0 && currentAdaPrice > peakPrice)
        peakPrice = currentAdaPrice;

      const buyCondition =
        buyProb > 0.5 &&
        usdBalance > this.CASH_RESERVE + this.MIN_TRADE_USD &&
        cooldown === 0;
      const sellCondition =
        (sellProb > 0.5 && adaBalance > 0) ||
        (adaBalance > 0 &&
          (unrealizedProfit <= this.STOP_LOSS_THRESHOLD ||
            unrealizedProfit >= this.TAKE_PROFIT_THRESHOLD ||
            trailingStopTriggered));

      console.log(
        `Day ${i - (this.timesteps + 3)}: Price ${currentAdaPrice.toFixed(
          4
        )}, Buy Prob: ${buyProb.toFixed(3)}, Sell Prob: ${sellProb.toFixed(
          3
        )}, Confidence: ${confidence.toFixed(3)}`
      );

      if (buyCondition) {
        const usdToSpend = Math.max(
          this.MIN_TRADE_USD,
          usdBalance *
            Math.min(0.5, confidence * 1.5) *
            (1 - this.TRANSACTION_FEE)
        );
        if (usdBalance - usdToSpend < this.CASH_RESERVE) continue;
        const adaBought = usdToSpend / currentAdaPrice;
        adaBalance += adaBought;
        avgBuyPrice =
          adaBalance > 0
            ? (avgBuyPrice * (adaBalance - adaBought) +
                currentAdaPrice * adaBought) /
              adaBalance
            : currentAdaPrice;
        usdBalance -= usdToSpend / (1 - this.TRANSACTION_FEE);
        peakPrice = currentAdaPrice;
        trades.push({
          type: "buy",
          price: currentAdaPrice,
          timestamp,
          adaAmount: adaBought,
          usdValue: usdToSpend,
        });
        console.log(
          `Buy at $${currentAdaPrice.toFixed(4)}, ADA: ${adaBought.toFixed(
            2
          )}, USD: ${usdToSpend.toFixed(2)}`
        );
      } else if (sellCondition) {
        const adaToSell = adaBalance;
        const usdReceived =
          adaToSell * currentAdaPrice * (1 - this.TRANSACTION_FEE);
        usdBalance += usdReceived;
        adaBalance -= adaToSell;
        trades.push({
          type: "sell",
          price: currentAdaPrice,
          timestamp,
          adaAmount: adaToSell,
          usdValue: usdReceived,
          buyPrice: avgBuyPrice,
        });
        console.log(
          `Sell at $${currentAdaPrice.toFixed(4)}, ADA: ${adaToSell.toFixed(
            2
          )}, USD: ${usdReceived.toFixed(2)}`
        );
        if (usdReceived > adaToSell * avgBuyPrice) wins++;
        if (adaBalance === 0) {
          completedCycles++;
          avgBuyPrice = 0;
          peakPrice = 0;
          cooldown = 2;
        }
      }

      portfolioHistory.push({ timestamp, value: portfolioValue });
      dailyReturns.push(
        portfolioHistory.length > 1
          ? (portfolioValue -
              portfolioHistory[portfolioHistory.length - 2].value) /
              portfolioHistory[portfolioHistory.length - 2].value
          : 0
      );
      if (cooldown > 0) cooldown--;
    }

    const finalValue = usdBalance + adaBalance * adaPrices[backtestEnd - 6];
    const totalReturn =
      ((finalValue - this.INITIAL_USD) / this.INITIAL_USD) * 100;
    const totalTrades = trades.length;
    const winRate = completedCycles > 0 ? (wins / completedCycles) * 100 : 0;
    const sharpeRatio = this.calculateSharpeRatio(dailyReturns);
    const maxDrawdown = this.calculateMaxDrawdown(
      portfolioHistory.map((h) => h.value)
    );

    console.log(`Backtest completed:
      Initial USD: $${this.INITIAL_USD}
      Final Value: $${finalValue.toFixed(2)}
      Total Return: ${totalReturn.toFixed(2)}%
      Total Trades: ${totalTrades}
      Win Rate: ${winRate.toFixed(2)}%
      Sharpe Ratio: ${sharpeRatio.toFixed(2)}
      Max Drawdown: ${maxDrawdown.toFixed(2)}%`);

    return {
      totalReturn,
      totalTrades,
      winRate,
      sharpeRatio,
      maxDrawdown,
      portfolioHistory,
      trades,
    };
  }

  public static async runWithTimeout(
    startDaysAgo?: number,
    endDaysAgo?: number,
    stepDays?: number
  ): Promise<BacktestResult> {
    const backtester = new TradeModelBacktester(
      startDaysAgo,
      endDaysAgo,
      stepDays
    );
    return new Promise((resolve, reject) => {
      setTimeout(() => {
        backtester.run().then(resolve).catch(reject);
      }, 5000);
    });
  }
}
