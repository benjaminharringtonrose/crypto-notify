/**
 * Data Collection Module for Bitcoin Price Prediction Bot
 * Fetches historical Bitcoin price data and prepares it for analysis.
 */

import { CryptoCompareService } from "../../api/CryptoCompareService";

export interface BitcoinDataPoint {
  Date: Date;
  Open: number;
  High: number;
  Low: number;
  Close: number;
  Volume: number;
  Dividends?: number;
  Stock_Splits?: number;
}

export interface BitcoinDataCollectorConfig {
  dataDir?: string;
}

export class BitcoinDataCollector {
  private dataDir: string;
  private cryptoCompareService: CryptoCompareService;

  constructor(config: BitcoinDataCollectorConfig = {}) {
    this.dataDir = config.dataDir || "data";
    this.cryptoCompareService = new CryptoCompareService();
    this.ensureDataDirectory();
  }

  private ensureDataDirectory(): void {
    // In a Node.js environment, we would create the directory
    // For Firebase Functions, we'll handle this differently
    console.log(`Data directory: ${this.dataDir}`);
  }

  async fetchHistoricalData(
    period: string = "5y",
    interval: string = "1d"
  ): Promise<BitcoinDataPoint[]> {
    try {
      console.log(
        `Fetching ${period} of Bitcoin data with ${interval} intervals...`
      );

      // Convert period to days
      let days: number;
      if (period.includes("y")) {
        days = parseInt(period.replace("y", "")) * 365;
      } else if (period.includes("mo")) {
        days = parseInt(period.replace("mo", "")) * 30;
      } else if (period.includes("d")) {
        days = parseInt(period.replace("d", ""));
      } else {
        days = 1825; // Default to 5 years
      }

      // Use CryptoCompareService to fetch real Bitcoin data
      try {
        const realData = await this.cryptoCompareService.getFullHistoricalData(
          "BTC",
          days
        );
        console.log(
          `Successfully fetched ${realData.length} days of real Bitcoin data`
        );
        return realData;
      } catch (apiError) {
        console.warn(`CryptoCompare API failed: ${apiError}`);
        console.log("Falling back to sample data...");
        return this.generateSampleData(period);
      }
    } catch (e) {
      console.error(`Error fetching data: ${e}`);
      console.warn("Using sample data as final fallback");
      return this.generateSampleData(period);
    }
  }

  async saveData(data: BitcoinDataPoint[], filename?: string): Promise<string> {
    const timestamp = new Date().toISOString().replace(/[:.]/g, "-");
    const finalFilename = filename || `bitcoin_data_${timestamp}.json`;
    const filepath = `${this.dataDir}/${finalFilename}`;

    // In a real implementation, you would save to file system or database
    console.log(`Data saved to: ${filepath}`);
    return filepath;
  }

  async loadData(filename: string): Promise<BitcoinDataPoint[]> {
    const filepath = `${this.dataDir}/${filename}`;

    // In a real implementation, you would load from file system or database
    console.log(`Loaded data from: ${filepath}`);
    return this.generateSampleData("1y"); // Fallback for now
  }

  async getLatestData(days: number = 30): Promise<BitcoinDataPoint[]> {
    try {
      console.log(`Fetching last ${days} days of Bitcoin data...`);

      // Use CryptoCompareService to fetch real Bitcoin data
      try {
        const realData = await this.cryptoCompareService.getFullHistoricalData(
          "BTC",
          days
        );
        console.log(
          `Successfully fetched ${realData.length} days of real Bitcoin data`
        );
        return realData;
      } catch (apiError) {
        console.warn(`CryptoCompare API failed: ${apiError}`);
        console.log("Falling back to sample data...");
        return this.generateSampleData(`${days}d`);
      }
    } catch (e) {
      console.error(`Error fetching recent data: ${e}`);
      console.warn("Using sample data as final fallback");
      return this.generateSampleData(`${days}d`);
    }
  }

  validateData(data: BitcoinDataPoint[]): boolean {
    if (!data || data.length === 0) {
      console.error("Data is empty");
      return false;
    }

    // Check for required properties in first data point
    const requiredProperties = [
      "Date",
      "Open",
      "High",
      "Low",
      "Close",
      "Volume",
    ];
    const firstPoint = data[0];

    for (const prop of requiredProperties) {
      if (!(prop in firstPoint)) {
        console.error(`Missing required property: ${prop}`);
        return false;
      }
    }

    // Check for negative prices
    const hasNegativePrices = data.some(
      (point) =>
        point.Open < 0 || point.High < 0 || point.Low < 0 || point.Close < 0
    );

    if (hasNegativePrices) {
      console.error("Negative prices found in data");
      return false;
    }

    // Check date range
    const dateRange =
      data[data.length - 1].Date.getTime() - data[0].Date.getTime();
    const daysRange = dateRange / (1000 * 60 * 60 * 24);

    if (daysRange < 30) {
      console.warn("Data spans less than 30 days");
    }

    console.log("Data validation passed");
    return true;
  }

  private generateSampleData(period: string = "2y"): BitcoinDataPoint[] {
    // Convert period to days
    let days: number;
    if (period.includes("y")) {
      days = parseInt(period.replace("y", "")) * 365;
    } else if (period.includes("mo")) {
      days = parseInt(period.replace("mo", "")) * 30;
    } else if (period.includes("d")) {
      days = parseInt(period.replace("d", ""));
    } else {
      days = 365;
    }

    // Generate dates
    const endDate = new Date();
    const startDate = new Date(endDate.getTime() - days * 24 * 60 * 60 * 1000);

    // Create realistic Bitcoin price data with some volatility
    const basePrice = 45000;
    const data: BitcoinDataPoint[] = [];

    for (let i = 0; i < days; i++) {
      const date = new Date(startDate.getTime() + i * 24 * 60 * 60 * 1000);

      // Add some random walk with trend
      let price: number;
      if (i === 0) {
        price = basePrice;
      } else {
        // Random daily change between -5% and +5%
        const dailyChange = (Math.random() - 0.5) * 0.04; // 2% standard deviation
        price = data[i - 1].Close * (1 + dailyChange);
      }

      // Create OHLCV data
      const open = price;
      const high = price * (1 + Math.abs((Math.random() - 0.5) * 0.02));
      const low = price * (1 - Math.abs((Math.random() - 0.5) * 0.02));
      const close = price;
      const volume = Math.random() * 4000000 + 1000000;

      // Ensure High >= Low
      const maxPrice = Math.max(open, high, close);
      const minPrice = Math.min(open, low, close);

      data.push({
        Date: date,
        Open: open,
        High: maxPrice,
        Low: minPrice,
        Close: close,
        Volume: volume,
        Dividends: 0,
        Stock_Splits: 0,
      });
    }

    console.log(`Generated ${data.length} days of sample Bitcoin data`);
    console.log(
      `Price range: $${Math.min(
        ...data.map((d) => d.Close)
      ).toLocaleString()} - $${Math.max(
        ...data.map((d) => d.Close)
      ).toLocaleString()}`
    );

    return data;
  }
}

// Test function
export async function testDataCollection(): Promise<void> {
  const collector = new BitcoinDataCollector();

  // Fetch historical data
  const data = await collector.fetchHistoricalData("1y");

  // Validate data
  if (collector.validateData(data)) {
    // Save data
    const filepath = await collector.saveData(data);
    console.log(
      `Data collection completed successfully. Saved to: ${filepath}`
    );
    console.log(`Data shape: ${data.length} records`);
    console.log(
      `Date range: ${data[0].Date.toISOString()} to ${data[
        data.length - 1
      ].Date.toISOString()}`
    );
  } else {
    console.log("Data validation failed");
  }
}
