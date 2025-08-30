/**
 * Feature Engineering Module for Bitcoin Price Prediction Bot
 * Creates technical indicators and features for machine learning models.
 */

import { BitcoinDataPoint } from "./DataCollector";

export interface FeatureDataPoint extends BitcoinDataPoint {
  // Price-based indicators
  Price_Change?: number;
  Price_Change_2d?: number;
  Price_Change_5d?: number;
  Price_Change_10d?: number;

  // Volatility indicators
  Volatility_5d?: number;
  Volatility_10d?: number;
  Volatility_20d?: number;

  // Moving averages
  SMA_5?: number;
  SMA_10?: number;
  SMA_20?: number;
  SMA_50?: number;
  EMA_5?: number;
  EMA_10?: number;
  EMA_20?: number;

  // Price relative to moving averages
  Price_vs_SMA_5?: number;
  Price_vs_SMA_10?: number;
  Price_vs_SMA_20?: number;
  Price_vs_SMA_50?: number;

  // RSI
  RSI?: number;
  RSI_5?: number;
  RSI_10?: number;

  // MACD
  MACD?: number;
  MACD_Signal?: number;
  MACD_Histogram?: number;

  // Bollinger Bands
  BB_Upper?: number;
  BB_Lower?: number;
  BB_Middle?: number;
  BB_Width?: number;
  BB_Position?: number;

  // Stochastic Oscillator
  Stoch_K?: number;
  Stoch_D?: number;

  // Williams %R
  Williams_R?: number;

  // Commodity Channel Index
  CCI?: number;

  // Average True Range
  ATR?: number;

  // Volume indicators
  Volume_SMA_5?: number;
  Volume_SMA_10?: number;
  Volume_Ratio?: number;
  OBV?: number;
  MFI?: number;

  // Price patterns
  High_Low_Ratio?: number;
  Open_Close_Ratio?: number;

  // Support and resistance
  Support_20d?: number;
  Resistance_20d?: number;
  Price_vs_Support?: number;
  Price_vs_Resistance?: number;

  // Time features
  Day_of_Week?: number;
  Day_of_Month?: number;
  Month?: number;
  Quarter?: number;
  Day_of_Week_Sin?: number;
  Day_of_Week_Cos?: number;
  Month_Sin?: number;
  Month_Cos?: number;

  // Target variables
  Future_Price?: number;
  Future_Price_Change?: number;
  Target?: number;
  Target_3Class?: number;
}

export interface FeatureEngineeringResult {
  X: number[][];
  y: number[];
  featureColumns: string[];
}

export class FeatureEngineer {
  createTechnicalIndicators(data: BitcoinDataPoint[]): FeatureDataPoint[] {
    console.log("Creating technical indicators...");

    const df: FeatureDataPoint[] = data.map((point) => ({ ...point }));

    // Price-based indicators
    for (let i = 1; i < df.length; i++) {
      df[i].Price_Change = (df[i].Close - df[i - 1].Close) / df[i - 1].Close;
    }

    for (let i = 2; i < df.length; i++) {
      df[i].Price_Change_2d = (df[i].Close - df[i - 2].Close) / df[i - 2].Close;
    }

    for (let i = 5; i < df.length; i++) {
      df[i].Price_Change_5d = (df[i].Close - df[i - 5].Close) / df[i - 5].Close;
    }

    for (let i = 10; i < df.length; i++) {
      df[i].Price_Change_10d =
        (df[i].Close - df[i - 10].Close) / df[i - 10].Close;
    }

    // Volatility indicators
    for (let i = 5; i < df.length; i++) {
      const changes = df.slice(i - 4, i + 1).map((d) => d.Price_Change || 0);
      df[i].Volatility_5d = this.calculateStandardDeviation(changes);
    }

    for (let i = 10; i < df.length; i++) {
      const changes = df.slice(i - 9, i + 1).map((d) => d.Price_Change || 0);
      df[i].Volatility_10d = this.calculateStandardDeviation(changes);
    }

    for (let i = 20; i < df.length; i++) {
      const changes = df.slice(i - 19, i + 1).map((d) => d.Price_Change || 0);
      df[i].Volatility_20d = this.calculateStandardDeviation(changes);
    }

    // Moving averages
    for (let i = 4; i < df.length; i++) {
      const prices = df.slice(i - 4, i + 1).map((d) => d.Close);
      df[i].SMA_5 = prices.reduce((sum, price) => sum + price, 0) / 5;
    }

    for (let i = 9; i < df.length; i++) {
      const prices = df.slice(i - 9, i + 1).map((d) => d.Close);
      df[i].SMA_10 = prices.reduce((sum, price) => sum + price, 0) / 10;
    }

    for (let i = 19; i < df.length; i++) {
      const prices = df.slice(i - 19, i + 1).map((d) => d.Close);
      df[i].SMA_20 = prices.reduce((sum, price) => sum + price, 0) / 20;
    }

    for (let i = 49; i < df.length; i++) {
      const prices = df.slice(i - 49, i + 1).map((d) => d.Close);
      df[i].SMA_50 = prices.reduce((sum, price) => sum + price, 0) / 50;
    }

    // EMA calculations
    for (let i = 4; i < df.length; i++) {
      df[i].EMA_5 = this.calculateEMA(df, i, 5, "Close");
    }

    for (let i = 9; i < df.length; i++) {
      df[i].EMA_10 = this.calculateEMA(df, i, 10, "Close");
    }

    for (let i = 19; i < df.length; i++) {
      df[i].EMA_20 = this.calculateEMA(df, i, 20, "Close");
    }

    // Price relative to moving averages
    for (let i = 4; i < df.length; i++) {
      if (df[i].SMA_5 !== undefined) {
        df[i].Price_vs_SMA_5 = (df[i].Close - df[i].SMA_5!) / df[i].SMA_5!;
      }
    }

    for (let i = 9; i < df.length; i++) {
      if (df[i].SMA_10 !== undefined) {
        df[i].Price_vs_SMA_10 = (df[i].Close - df[i].SMA_10!) / df[i].SMA_10!;
      }
    }

    for (let i = 19; i < df.length; i++) {
      if (df[i].SMA_20 !== undefined) {
        df[i].Price_vs_SMA_20 = (df[i].Close - df[i].SMA_20!) / df[i].SMA_20!;
      }
    }

    for (let i = 49; i < df.length; i++) {
      if (df[i].SMA_50 !== undefined) {
        df[i].Price_vs_SMA_50 = (df[i].Close - df[i].SMA_50!) / df[i].SMA_50!;
      }
    }

    // RSI
    for (let i = 13; i < df.length; i++) {
      df[i].RSI = this.calculateRSI(df, i, 14);
    }

    for (let i = 4; i < df.length; i++) {
      df[i].RSI_5 = this.calculateRSI(df, i, 5);
    }

    for (let i = 9; i < df.length; i++) {
      df[i].RSI_10 = this.calculateRSI(df, i, 10);
    }

    // MACD
    for (let i = 25; i < df.length; i++) {
      const macd = this.calculateMACD(df, i);
      df[i].MACD = macd.macd;
      df[i].MACD_Signal = macd.signal;
      df[i].MACD_Histogram = macd.histogram;
    }

    // Bollinger Bands
    for (let i = 19; i < df.length; i++) {
      const bb = this.calculateBollingerBands(df, i);
      df[i].BB_Upper = bb.upper;
      df[i].BB_Lower = bb.lower;
      df[i].BB_Middle = bb.middle;
      df[i].BB_Width = (bb.upper - bb.lower) / bb.middle;
      df[i].BB_Position = (df[i].Close - bb.lower) / (bb.upper - bb.lower);
    }

    // Stochastic Oscillator
    for (let i = 13; i < df.length; i++) {
      const stoch = this.calculateStochastic(df, i);
      df[i].Stoch_K = stoch.k;
      df[i].Stoch_D = stoch.d;
    }

    // Williams %R
    for (let i = 13; i < df.length; i++) {
      df[i].Williams_R = this.calculateWilliamsR(df, i);
    }

    // CCI
    for (let i = 19; i < df.length; i++) {
      df[i].CCI = this.calculateCCI(df, i);
    }

    // ATR
    for (let i = 13; i < df.length; i++) {
      df[i].ATR = this.calculateATR(df, i);
    }

    // Volume indicators
    for (let i = 4; i < df.length; i++) {
      const volumes = df.slice(i - 4, i + 1).map((d) => d.Volume);
      df[i].Volume_SMA_5 = volumes.reduce((sum, vol) => sum + vol, 0) / 5;
    }

    for (let i = 9; i < df.length; i++) {
      const volumes = df.slice(i - 9, i + 1).map((d) => d.Volume);
      df[i].Volume_SMA_10 = volumes.reduce((sum, vol) => sum + vol, 0) / 10;
    }

    for (let i = 4; i < df.length; i++) {
      if (df[i].Volume_SMA_5 !== undefined) {
        df[i].Volume_Ratio = df[i].Volume / df[i].Volume_SMA_5!;
      }
    }

    // OBV
    for (let i = 1; i < df.length; i++) {
      if (i === 1) {
        df[i].OBV = df[i].Volume;
      } else {
        const prevOBV = df[i - 1].OBV || 0;
        if (df[i].Close > df[i - 1].Close) {
          df[i].OBV = prevOBV + df[i].Volume;
        } else if (df[i].Close < df[i - 1].Close) {
          df[i].OBV = prevOBV - df[i].Volume;
        } else {
          df[i].OBV = prevOBV;
        }
      }
    }

    // MFI
    for (let i = 13; i < df.length; i++) {
      df[i].MFI = this.calculateMFI(df, i);
    }

    // Price patterns
    for (let i = 0; i < df.length; i++) {
      df[i].High_Low_Ratio = df[i].High / df[i].Low;
      df[i].Open_Close_Ratio = df[i].Open / df[i].Close;
    }

    // Support and resistance
    for (let i = 19; i < df.length; i++) {
      const lows = df.slice(i - 19, i + 1).map((d) => d.Low);
      const highs = df.slice(i - 19, i + 1).map((d) => d.High);
      df[i].Support_20d = Math.min(...lows);
      df[i].Resistance_20d = Math.max(...highs);

      if (df[i].Support_20d !== undefined) {
        df[i].Price_vs_Support =
          (df[i].Close - df[i].Support_20d!) / df[i].Support_20d!;
      }
      if (df[i].Resistance_20d !== undefined) {
        df[i].Price_vs_Resistance =
          (df[i].Close - df[i].Resistance_20d!) / df[i].Resistance_20d!;
      }
    }

    // Time features
    for (let i = 0; i < df.length; i++) {
      const date = df[i].Date;
      df[i].Day_of_Week = date.getDay();
      df[i].Day_of_Month = date.getDate();
      df[i].Month = date.getMonth() + 1;
      df[i].Quarter = Math.ceil((date.getMonth() + 1) / 3);

      // Cyclical encoding
      if (df[i].Day_of_Week !== undefined) {
        df[i].Day_of_Week_Sin = Math.sin(
          (2 * Math.PI * df[i].Day_of_Week!) / 7
        );
        df[i].Day_of_Week_Cos = Math.cos(
          (2 * Math.PI * df[i].Day_of_Week!) / 7
        );
      }
      if (df[i].Month !== undefined) {
        df[i].Month_Sin = Math.sin((2 * Math.PI * df[i].Month!) / 12);
        df[i].Month_Cos = Math.cos((2 * Math.PI * df[i].Month!) / 12);
      }
    }

    console.log(`Created technical indicators for ${df.length} data points`);
    return df;
  }

  createTargetVariable(
    data: FeatureDataPoint[],
    predictionDays: number = 3
  ): FeatureDataPoint[] {
    console.log(
      `Creating target variable for ${predictionDays}-day prediction...`
    );

    const df: FeatureDataPoint[] = data.map((point) => ({ ...point }));

    // Calculate future price change
    for (let i = 0; i < df.length - predictionDays; i++) {
      df[i].Future_Price = df[i + predictionDays].Close;
      if (df[i].Future_Price !== undefined) {
        df[i].Future_Price_Change =
          (df[i].Future_Price! - df[i].Close) / df[i].Close;

        // Create binary target (1 for price increase, 0 for decrease)
        df[i].Target = df[i].Future_Price_Change! > 0 ? 1 : 0;

        // Create multi-class target for more granular prediction
        if (df[i].Future_Price_Change! < -0.05) {
          df[i].Target_3Class = 0; // Strong decrease
        } else if (df[i].Future_Price_Change! > 0.05) {
          df[i].Target_3Class = 2; // Strong increase
        } else {
          df[i].Target_3Class = 1; // Stable
        }
      }
    }

    // Remove rows where we don't have future data
    const filteredData = df.filter((point) => point.Future_Price !== undefined);

    const targetDistribution = this.calculateTargetDistribution(filteredData);
    console.log(`Target variable created. Distribution:`, targetDistribution);

    return filteredData;
  }

  selectFeatures(
    data: FeatureDataPoint[],
    targetCol: string = "Target",
    excludeCols: string[] = []
  ): FeatureEngineeringResult {
    const defaultExcludeCols = [
      "Date",
      "Future_Price",
      "Future_Price_Change",
      "Target_3Class",
    ];
    const allExcludeCols = [...defaultExcludeCols, ...excludeCols, targetCol];

    // Get feature columns
    const featureCols = Object.keys(data[0]).filter(
      (col) => !allExcludeCols.includes(col)
    );

    // Create feature matrix and target vector
    const X: number[][] = [];
    const y: number[] = [];

    for (const point of data) {
      const features: number[] = [];
      let hasNaN = false;

      for (const col of featureCols) {
        const value = (point as any)[col];
        if (value === undefined || value === null || isNaN(value)) {
          hasNaN = true;
          break;
        }
        features.push(value);
      }

      if (!hasNaN) {
        X.push(features);
        y.push((point as any)[targetCol]);
      }
    }

    console.log(`Selected ${featureCols.length} features`);
    console.log(`Final dataset shape: ${X.length} x ${X[0]?.length || 0}`);

    return { X, y, featureColumns: featureCols };
  }

  scaleFeatures(
    X: number[][],
    method: "standard" | "minmax" = "standard"
  ): { scaledX: number[][]; scaler: any } {
    console.log(`Scaling features using ${method} scaling...`);

    if (method === "standard") {
      return this.standardScaler(X);
    } else {
      return this.minMaxScaler(X);
    }
  }

  createLagFeatures(
    data: FeatureDataPoint[],
    columns: string[],
    lags: number[] = [1, 2, 3, 5, 10]
  ): FeatureDataPoint[] {
    console.log(`Creating lag features for ${columns.length} columns...`);

    const df: FeatureDataPoint[] = data.map((point) => ({ ...point }));

    for (const col of columns) {
      for (const lag of lags) {
        const lagColName = `${col}_lag_${lag}`;
        for (let i = lag; i < df.length; i++) {
          (df[i] as any)[lagColName] = (df[i - lag] as any)[col];
        }
      }
    }

    console.log(`Created ${columns.length * lags.length} lag features`);
    return df;
  }

  createRollingFeatures(
    data: FeatureDataPoint[],
    columns: string[],
    windows: number[] = [5, 10, 20]
  ): FeatureDataPoint[] {
    console.log(`Creating rolling features for ${columns.length} columns...`);

    const df: FeatureDataPoint[] = data.map((point) => ({ ...point }));

    for (const col of columns) {
      for (const window of windows) {
        const meanColName = `${col}_rolling_mean_${window}`;
        const stdColName = `${col}_rolling_std_${window}`;
        const minColName = `${col}_rolling_min_${window}`;
        const maxColName = `${col}_rolling_max_${window}`;

        for (let i = window - 1; i < df.length; i++) {
          const values = df
            .slice(i - window + 1, i + 1)
            .map((d) => (d as any)[col])
            .filter((v) => v !== undefined && !isNaN(v));

          if (values.length > 0) {
            (df[i] as any)[meanColName] =
              values.reduce((sum, val) => sum + val, 0) / values.length;
            (df[i] as any)[stdColName] =
              this.calculateStandardDeviation(values);
            (df[i] as any)[minColName] = Math.min(...values);
            (df[i] as any)[maxColName] = Math.max(...values);
          }
        }
      }
    }

    console.log(
      `Created ${columns.length * windows.length * 4} rolling features`
    );
    return df;
  }

  getFeatureImportanceRanking(
    featureImportanceDict: Record<string, number>
  ): Array<{ feature: string; importance: number }> {
    return Object.entries(featureImportanceDict)
      .map(([feature, importance]) => ({ feature, importance }))
      .sort((a, b) => b.importance - a.importance);
  }

  // Helper methods for technical indicators
  private calculateStandardDeviation(values: number[]): number {
    const mean = values.reduce((sum, val) => sum + val, 0) / values.length;
    const squaredDiffs = values.map((val) => Math.pow(val - mean, 2));
    const variance =
      squaredDiffs.reduce((sum, diff) => sum + diff, 0) / values.length;
    return Math.sqrt(variance);
  }

  private calculateEMA(
    data: FeatureDataPoint[],
    index: number,
    period: number,
    field: keyof BitcoinDataPoint
  ): number {
    const alpha = 2 / (period + 1);
    let ema = (data[index - period + 1] as any)[field];

    for (let i = index - period + 2; i <= index; i++) {
      ema = alpha * (data[i] as any)[field] + (1 - alpha) * ema;
    }

    return ema;
  }

  private calculateRSI(
    data: FeatureDataPoint[],
    index: number,
    period: number
  ): number {
    if (index < period) return 50; // Default value for insufficient data

    const gains: number[] = [];
    const losses: number[] = [];

    for (let i = index - period + 1; i <= index; i++) {
      if (i > 0 && data[i] && data[i - 1]) {
        const change = data[i].Close - data[i - 1].Close;
        gains.push(change > 0 ? change : 0);
        losses.push(change < 0 ? -change : 0);
      }
    }

    if (gains.length === 0 || losses.length === 0) return 50;

    const avgGain = gains.reduce((sum, gain) => sum + gain, 0) / period;
    const avgLoss = losses.reduce((sum, loss) => sum + loss, 0) / period;

    if (avgLoss === 0) return 100;

    const rs = avgGain / avgLoss;
    return 100 - 100 / (1 + rs);
  }

  private calculateMACD(
    data: FeatureDataPoint[],
    index: number
  ): { macd: number; signal: number; histogram: number } {
    const ema12 = this.calculateEMA(data, index, 12, "Close");
    const ema26 = this.calculateEMA(data, index, 26, "Close");
    const macd = ema12 - ema26;

    // Calculate signal line as EMA of MACD (9-period)
    let signal = macd;
    if (index >= 34) {
      // Need at least 26 + 9 - 1 = 34 data points
      const macdValues: number[] = [];
      for (let i = index - 8; i <= index; i++) {
        const ema12_i = this.calculateEMA(data, i, 12, "Close");
        const ema26_i = this.calculateEMA(data, i, 26, "Close");
        macdValues.push(ema12_i - ema26_i);
      }
      signal = this.calculateEMAFromArray(macdValues, 9);
    }

    return {
      macd,
      signal,
      histogram: macd - signal,
    };
  }

  private calculateEMAFromArray(values: number[], period: number): number {
    if (values.length === 0) return 0;
    if (values.length === 1) return values[0];

    const alpha = 2 / (period + 1);
    let ema = values[0];

    for (let i = 1; i < values.length; i++) {
      ema = alpha * values[i] + (1 - alpha) * ema;
    }

    return ema;
  }

  private calculateBollingerBands(
    data: FeatureDataPoint[],
    index: number
  ): { upper: number; lower: number; middle: number } {
    const prices = data.slice(index - 19, index + 1).map((d) => d.Close);
    const middle = prices.reduce((sum, price) => sum + price, 0) / 20;

    // Calculate rolling standard deviation
    const squaredDiffs = prices.map((price) => Math.pow(price - middle, 2));
    const variance = squaredDiffs.reduce((sum, diff) => sum + diff, 0) / 20;
    const std = Math.sqrt(variance);

    return {
      upper: middle + 2 * std,
      lower: middle - 2 * std,
      middle,
    };
  }

  private calculateStochastic(
    data: FeatureDataPoint[],
    index: number
  ): { k: number; d: number } {
    if (index < 13) return { k: 50, d: 50 }; // Default values for insufficient data

    const high14 = Math.max(
      ...data.slice(index - 13, index + 1).map((d) => d.High)
    );
    const low14 = Math.min(
      ...data.slice(index - 13, index + 1).map((d) => d.Low)
    );
    const close = data[index].Close;

    if (high14 === low14) return { k: 50, d: 50 }; // Avoid division by zero

    const k = ((close - low14) / (high14 - low14)) * 100;

    // Calculate D as 3-period SMA of K
    let d = k;
    if (index >= 15) {
      const kValues: number[] = [];
      for (let i = index - 2; i <= index; i++) {
        const high14_i = Math.max(
          ...data.slice(i - 13, i + 1).map((d) => d.High)
        );
        const low14_i = Math.min(
          ...data.slice(i - 13, i + 1).map((d) => d.Low)
        );
        const close_i = data[i].Close;
        if (high14_i !== low14_i) {
          kValues.push(((close_i - low14_i) / (high14_i - low14_i)) * 100);
        } else {
          kValues.push(50);
        }
      }
      d = kValues.reduce((sum, val) => sum + val, 0) / kValues.length;
    }

    return { k, d };
  }

  private calculateWilliamsR(data: FeatureDataPoint[], index: number): number {
    if (index < 13) return -50; // Default value for insufficient data

    const high14 = Math.max(
      ...data.slice(index - 13, index + 1).map((d) => d.High)
    );
    const low14 = Math.min(
      ...data.slice(index - 13, index + 1).map((d) => d.Low)
    );
    const close = data[index].Close;

    if (high14 === low14) return -50; // Avoid division by zero

    return ((high14 - close) / (high14 - low14)) * -100;
  }

  private calculateCCI(data: FeatureDataPoint[], index: number): number {
    if (index < 19) return 0; // Default value for insufficient data

    const typicalPrices = data
      .slice(index - 19, index + 1)
      .map((d) => (d.High + d.Low + d.Close) / 3);
    const sma = typicalPrices.reduce((sum, price) => sum + price, 0) / 20;

    // Calculate mean deviation
    const meanDeviation =
      typicalPrices.reduce((sum, price) => sum + Math.abs(price - sma), 0) / 20;
    const currentTypicalPrice =
      (data[index].High + data[index].Low + data[index].Close) / 3;

    if (meanDeviation === 0) return 0; // Avoid division by zero

    return (currentTypicalPrice - sma) / (0.015 * meanDeviation);
  }

  private calculateATR(data: FeatureDataPoint[], index: number): number {
    if (index < 13) return 0; // Default value for insufficient data

    const trueRanges: number[] = [];

    for (let i = index - 13; i <= index; i++) {
      if (i > 0 && data[i] && data[i - 1]) {
        const high = data[i].High;
        const low = data[i].Low;
        const prevClose = data[i - 1].Close;

        const tr1 = high - low;
        const tr2 = Math.abs(high - prevClose);
        const tr3 = Math.abs(low - prevClose);

        trueRanges.push(Math.max(tr1, tr2, tr3));
      }
    }

    if (trueRanges.length === 0) return 0;

    return trueRanges.reduce((sum, tr) => sum + tr, 0) / trueRanges.length;
  }

  private calculateMFI(data: FeatureDataPoint[], index: number): number {
    if (index < 13) return 50; // Default value for insufficient data

    const moneyFlows: number[] = [];

    for (let i = index - 13; i <= index; i++) {
      if (i > 0 && data[i] && data[i - 1]) {
        const typicalPrice = (data[i].High + data[i].Low + data[i].Close) / 3;
        const prevTypicalPrice =
          (data[i - 1].High + data[i - 1].Low + data[i - 1].Close) / 3;
        const rawMoneyFlow = typicalPrice * data[i].Volume;

        if (typicalPrice > prevTypicalPrice) {
          moneyFlows.push(rawMoneyFlow);
        } else {
          moneyFlows.push(-rawMoneyFlow);
        }
      }
    }

    if (moneyFlows.length === 0) return 50;

    const positiveFlow = moneyFlows
      .filter((mf) => mf > 0)
      .reduce((sum, mf) => sum + mf, 0);
    const negativeFlow = Math.abs(
      moneyFlows.filter((mf) => mf < 0).reduce((sum, mf) => sum + mf, 0)
    );

    if (negativeFlow === 0) return 100;

    const moneyRatio = positiveFlow / negativeFlow;
    return 100 - 100 / (1 + moneyRatio);
  }

  private calculateTargetDistribution(
    data: FeatureDataPoint[]
  ): Record<string, number> {
    const distribution: Record<string, number> = {};

    for (const point of data) {
      const target = point.Target?.toString() || "unknown";
      distribution[target] = (distribution[target] || 0) + 1;
    }

    return distribution;
  }

  private standardScaler(X: number[][]): { scaledX: number[][]; scaler: any } {
    const numFeatures = X[0].length;
    const means: number[] = new Array(numFeatures).fill(0);
    const stds: number[] = new Array(numFeatures).fill(0);

    // Calculate means
    for (let j = 0; j < numFeatures; j++) {
      for (let i = 0; i < X.length; i++) {
        means[j] += X[i][j];
      }
      means[j] /= X.length;
    }

    // Calculate standard deviations
    for (let j = 0; j < numFeatures; j++) {
      for (let i = 0; i < X.length; i++) {
        stds[j] += Math.pow(X[i][j] - means[j], 2);
      }
      stds[j] = Math.sqrt(stds[j] / X.length);
    }

    // Scale the data
    const scaledX = X.map((row) =>
      row.map((val, j) => (val - means[j]) / (stds[j] || 1))
    );

    return { scaledX, scaler: { means, stds } };
  }

  private minMaxScaler(X: number[][]): { scaledX: number[][]; scaler: any } {
    const numFeatures = X[0].length;
    const mins: number[] = new Array(numFeatures).fill(Infinity);
    const maxs: number[] = new Array(numFeatures).fill(-Infinity);

    // Find min and max for each feature
    for (let j = 0; j < numFeatures; j++) {
      for (let i = 0; i < X.length; i++) {
        mins[j] = Math.min(mins[j], X[i][j]);
        maxs[j] = Math.max(maxs[j], X[i][j]);
      }
    }

    // Scale the data
    const scaledX = X.map((row) =>
      row.map((val, j) => (val - mins[j]) / (maxs[j] - mins[j] || 1))
    );

    return { scaledX, scaler: { mins, maxs } };
  }
}

// Import for test function
import { BitcoinDataCollector } from "./DataCollector";

// Test function
export async function testFeatureEngineering(): Promise<void> {
  // Load sample data
  const collector = new BitcoinDataCollector();
  const data = await collector.getLatestData(100);

  // Create features
  const engineer = new FeatureEngineer();
  const dataWithFeatures = engineer.createTechnicalIndicators(data);
  const dataWithTarget = engineer.createTargetVariable(dataWithFeatures);

  // Select features
  const { X, y, featureColumns } = engineer.selectFeatures(dataWithTarget);

  console.log(`Feature engineering completed successfully!`);
  console.log(`Feature matrix shape: ${X.length} x ${X[0]?.length || 0}`);
  console.log(
    `Target distribution:`,
    y.reduce((acc, val) => {
      acc[val] = (acc[val] || 0) + 1;
      return acc;
    }, {} as Record<number, number>)
  );
  console.log(`Number of features: ${featureColumns.length}`);
}
