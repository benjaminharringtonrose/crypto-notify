# CryptoCompareService Class Documentation

## Overview

The `CryptoCompareService` class is a comprehensive service that provides access to cryptocurrency market data through the CryptoCompare API. It offers methods for retrieving historical price and volume data, real-time market information, and comprehensive cryptocurrency analytics for the Bitcoin trading system.

## Architecture

The CryptoCompareService follows a RESTful API integration pattern:

```
API Requests → Data Processing → Error Handling → Cached Results
     ↓              ↓                ↓              ↓
CryptoCompare    JSON Parsing    Retry Logic    Memory Cache
REST Endpoints   Data Validation  Fallback      Performance
```

### Key Responsibilities

- **Historical Data Retrieval**: Fetch historical price and volume data
- **Real-time Market Data**: Get current market prices and statistics
- **API Integration**: Manage CryptoCompare API requests and responses
- **Error Handling**: Robust error management and retry logic
- **Data Caching**: Optimize performance with intelligent caching
- **Rate Limiting**: Respect API rate limits and quotas

## Class Structure

```typescript
export class CryptoCompareService {
  private apiKey: string;
  private baseUrl: string;
  private cache: Map<string, any>;
  private rateLimiter: RateLimiter;

  constructor(apiKey?: string);

  public async getHistoricalData(
    cryptoSymbol: string,
    days: number
  ): Promise<HistoricalData>;

  public async getCurrentPrice(cryptoSymbol: string): Promise<number>;
  public async getMarketCap(cryptoSymbol: string): Promise<number>;
  public async get24hVolume(cryptoSymbol: string): Promise<number>;
  public async getPriceChange24h(cryptoSymbol: string): Promise<number>;
}
```

### Constructor Parameters

- `apiKey?: string` - Optional CryptoCompare API key for enhanced rate limits

## Core Interfaces

### HistoricalData

```typescript
interface HistoricalData {
  prices: number[]; // Historical price data
  volumes: number[]; // Historical volume data
  timestamps: number[]; // Unix timestamps
  metadata: {
    symbol: string;
    days: number;
    lastUpdated: number;
  };
}
```

### MarketData

```typescript
interface MarketData {
  price: number;
  marketCap: number;
  volume24h: number;
  priceChange24h: number;
  priceChangePercent24h: number;
  high24h: number;
  low24h: number;
  lastUpdated: number;
}
```

## Methods

### Public Methods

#### `getHistoricalData(cryptoSymbol, days): Promise<HistoricalData>`

Retrieves historical price and volume data for a specified cryptocurrency.

**Parameters:**

- `cryptoSymbol: string` - Cryptocurrency symbol (e.g., "BTC", "ETH", "ADA")
- `days: number` - Number of days of historical data to retrieve

**Returns:**

- `Promise<HistoricalData>` - Historical price and volume data

**Process:**

1. **API Request**: Constructs CryptoCompare API request
2. **Data Retrieval**: Fetches historical data from API
3. **Data Processing**: Parses and validates response data
4. **Cache Storage**: Stores results in memory cache
5. **Error Handling**: Manages API failures and retries

#### `getCurrentPrice(cryptoSymbol): Promise<number>`

Retrieves the current price for a specified cryptocurrency.

**Parameters:**

- `cryptoSymbol: string` - Cryptocurrency symbol

**Returns:**

- `Promise<number>` - Current price in USD

#### `getMarketCap(cryptoSymbol): Promise<number>`

Retrieves the current market capitalization for a specified cryptocurrency.

**Parameters:**

- `cryptoSymbol: string` - Cryptocurrency symbol

**Returns:**

- `Promise<number>` - Market capitalization in USD

#### `get24hVolume(cryptoSymbol): Promise<number>`

Retrieves the 24-hour trading volume for a specified cryptocurrency.

**Parameters:**

- `cryptoSymbol: string` - Cryptocurrency symbol

**Returns:**

- `Promise<number>` - 24-hour volume in USD

#### `getPriceChange24h(cryptoSymbol): Promise<number>`

Retrieves the 24-hour price change for a specified cryptocurrency.

**Parameters:**

- `cryptoSymbol: string` - Cryptocurrency symbol

**Returns:**

- `Promise<number>` - 24-hour price change in USD

## Data Flow

### 1. API Request Construction

```typescript
// Construct API URL
const url = `${this.baseUrl}/data/v2/histoday?fsym=${cryptoSymbol}&tsym=USD&limit=${days}`;

// Add API key if available
if (this.apiKey) {
  url += `&api_key=${this.apiKey}`;
}

// Set request headers
const headers = {
  "Content-Type": "application/json",
  "User-Agent": "CryptoNotify/1.0",
};
```

### 2. Data Retrieval and Processing

```typescript
// Make API request
const response = await fetch(url, { headers });

if (!response.ok) {
  throw new Error(
    `API request failed: ${response.status} ${response.statusText}`
  );
}

// Parse response data
const data = await response.json();

// Extract and validate data
const historicalData: HistoricalData = {
  prices: data.Data.Data.map((item: any) => item.close),
  volumes: data.Data.Data.map((item: any) => item.volumeto),
  timestamps: data.Data.Data.map((item: any) => item.time),
  metadata: {
    symbol: cryptoSymbol,
    days: days,
    lastUpdated: Date.now(),
  },
};
```

### 3. Caching and Performance

```typescript
// Check cache first
const cacheKey = `${cryptoSymbol}_${days}`;
const cachedData = this.cache.get(cacheKey);

if (
  cachedData &&
  Date.now() - cachedData.metadata.lastUpdated < CACHE_DURATION
) {
  return cachedData;
}

// Fetch fresh data
const freshData = await this.fetchFromAPI(cryptoSymbol, days);

// Store in cache
this.cache.set(cacheKey, freshData);

return freshData;
```

## Usage Examples

### Basic Historical Data Retrieval

```typescript
import { CryptoCompareService } from "./CryptoCompareService";

const cryptoCompare = new CryptoCompareService();

// Retrieve 1 year of daily data for BTC
const btcData = await cryptoCompare.getHistoricalData("BTC", 365);

console.log(`Retrieved ${btcData.prices.length} days of BTC data`);
console.log(`Latest price: $${btcData.prices[btcData.prices.length - 1]}`);
console.log(`Latest volume: ${btcData.volumes[btcData.volumes.length - 1]}`);
```

### Current Market Data

```typescript
// Get current BTC price
const btcPrice = await cryptoCompare.getCurrentPrice("BTC");
console.log(`Current BTC price: $${btcPrice}`);

// Get market cap
const btcMarketCap = await cryptoCompare.getMarketCap("BTC");
console.log(`BTC market cap: $${btcMarketCap.toLocaleString()}`);

// Get 24h volume
const btcVolume = await cryptoCompare.get24hVolume("BTC");
console.log(`BTC 24h volume: $${btcVolume.toLocaleString()}`);

// Get price change
const btcPriceChange = await cryptoCompare.getPriceChange24h("BTC");
console.log(`BTC 24h price change: $${btcPriceChange}`);
```

### Multiple Cryptocurrency Data

```typescript
// Fetch data for multiple cryptocurrencies
const symbols = ["BTC", "ETH", "SOL", "MATIC"];

const marketData = await Promise.all(
  symbols.map(async (symbol) => {
    const price = await cryptoCompare.getCurrentPrice(symbol);
    const marketCap = await cryptoCompare.getMarketCap(symbol);
    const volume = await cryptoCompare.get24hVolume(symbol);

    return {
      symbol,
      price,
      marketCap,
      volume,
    };
  })
);

console.log("Market Data Summary:");
marketData.forEach((data) => {
  console.log(
    `${data.symbol}: $${
      data.price
    } | MC: $${data.marketCap.toLocaleString()} | Vol: $${data.volume.toLocaleString()}`
  );
});
```

### Integration with DataProcessor

```typescript
import { DataProcessor } from "./bitcoin/DataProcessor";

// Use CryptoCompareService with DataProcessor
const dataProcessor = new DataProcessor();
const cryptoCompare = new CryptoCompareService();

// Fetch historical data
const btcData = await cryptoCompare.getHistoricalData("BTC", 450); // 1.25 years

// Process data for training
const trainingData = await dataProcessor.prepareTrainingData();

console.log(`Prepared ${trainingData.sequences.length} training sequences`);
```

### Integration with Backtesting

```typescript
import { TradeModelBacktester } from "./bitcoin/TradeModelBacktester";

// Use CryptoCompareService for backtesting
const backtester = new TradeModelBacktester(model, strategy, 10000);
const cryptoCompare = new CryptoCompareService();

// Fetch historical data for backtesting
const recentData = await cryptoCompare.getHistoricalData("BTC", 180); // 6 months
const historicalData = await cryptoCompare.getHistoricalData("BTC", 365); // 1 year

// Run backtest
const trades = await backtester.backtest(
  historicalData,
  100, // startIndex
  300 // endIndex
);

console.log(`Backtest completed with ${trades.length} trades`);
```

## Configuration

### API Configuration

```typescript
const API_CONFIG = {
  BASE_URL: "https://min-api.cryptocompare.com",
  RATE_LIMIT: 1000, // Requests per minute
  CACHE_DURATION: 5 * 60 * 1000, // 5 minutes in milliseconds
  TIMEOUT: 10000, // 10 seconds
  RETRY_ATTEMPTS: 3, // Number of retry attempts
  RETRY_DELAY: 1000, // Delay between retries in ms
};
```

### Error Handling Configuration

```typescript
const ERROR_CONFIG = {
  MAX_RETRIES: 3,
  RETRY_DELAY: 1000,
  TIMEOUT: 10000,
  RATE_LIMIT_WINDOW: 60000, // 1 minute
  RATE_LIMIT_MAX: 1000, // Max requests per window
};
```

## Error Handling

### API Error Management

```typescript
// Handle different types of API errors
try {
  const data = await this.getHistoricalData("BTC", 365);
  return data;
} catch (error) {
  if (error.message.includes("429")) {
    // Rate limit exceeded
    console.log("Rate limit exceeded, waiting before retry...");
    await this.delay(this.retryDelay);
    return this.getHistoricalData("BTC", 365);
  } else if (error.message.includes("404")) {
    // Symbol not found
    throw new Error(`Cryptocurrency symbol not found: BTC`);
  } else if (error.message.includes("500")) {
    // Server error
    console.log("Server error, retrying...");
    return this.retryRequest(() => this.getHistoricalData("BTC", 365));
  } else {
    // Other errors
    throw new Error(`API request failed: ${error.message}`);
  }
}
```

### Rate Limiting

```typescript
// Implement rate limiting
private async checkRateLimit(): Promise<void> {
  const now = Date.now();
  const windowStart = now - this.rateLimitWindow;

  // Remove old requests from window
  this.requestTimestamps = this.requestTimestamps.filter(
    timestamp => timestamp > windowStart
  );

  // Check if we're at the limit
  if (this.requestTimestamps.length >= this.rateLimitMax) {
    const oldestRequest = this.requestTimestamps[0];
    const waitTime = this.rateLimitWindow - (now - oldestRequest);
    await this.delay(waitTime);
  }

  // Add current request
  this.requestTimestamps.push(now);
}
```

### Fallback Mechanisms

```typescript
// Implement fallback data sources
private async getFallbackData(cryptoSymbol: string, days: number): Promise<HistoricalData> {
  console.log(`Attempting fallback data source for ${cryptoSymbol}`);

  // Try alternative API endpoints
  const fallbackUrls = [
    `https://api.coingecko.com/api/v3/coins/${cryptoSymbol}/market_chart?vs_currency=usd&days=${days}`,
    `https://api.binance.com/api/v3/klines?symbol=${cryptoSymbol}USDT&interval=1d&limit=${days}`
  ];

  for (const url of fallbackUrls) {
    try {
      const response = await fetch(url);
      if (response.ok) {
        return this.parseFallbackResponse(await response.json(), cryptoSymbol, days);
      }
    } catch (error) {
      console.log(`Fallback source failed: ${url}`);
    }
  }

  throw new Error(`All data sources failed for ${cryptoSymbol}`);
}
```

## Performance Considerations

### Caching Strategy

```typescript
// Implement intelligent caching
private getCacheKey(cryptoSymbol: string, days: number): string {
  return `${cryptoSymbol}_${days}_${Math.floor(Date.now() / this.cacheDuration)}`;
}

private isCacheValid(cachedData: any): boolean {
  const age = Date.now() - cachedData.metadata.lastUpdated;
  return age < this.cacheDuration;
}
```

### Batch Processing

```typescript
// Process multiple requests efficiently
public async getMultipleHistoricalData(
  symbols: string[],
  days: number
): Promise<Record<string, HistoricalData>> {
  const results: Record<string, HistoricalData> = {};

  // Process in batches to respect rate limits
  const batchSize = 5;
  for (let i = 0; i < symbols.length; i += batchSize) {
    const batch = symbols.slice(i, i + batchSize);

    const batchPromises = batch.map(async (symbol) => {
      const data = await this.getHistoricalData(symbol, days);
      return { symbol, data };
    });

    const batchResults = await Promise.all(batchPromises);

    batchResults.forEach(({ symbol, data }) => {
      results[symbol] = data;
    });

    // Rate limiting delay between batches
    if (i + batchSize < symbols.length) {
      await this.delay(1000);
    }
  }

  return results;
}
```

## Dependencies

### External Dependencies

- **Fetch API**: HTTP requests (Node.js 18+ or polyfill)
- **CryptoCompare API**: External cryptocurrency data source

### Internal Dependencies

- **Rate Limiter**: Custom rate limiting implementation
- **Cache Manager**: Memory-based caching system
- **Error Handler**: Comprehensive error management

## Testing

### Unit Testing Strategy

- **API Mocking**: Mock CryptoCompare API responses
- **Error Scenarios**: Test various error conditions
- **Rate Limiting**: Verify rate limit enforcement
- **Caching**: Test cache hit/miss scenarios

### Integration Testing

- **Real API Calls**: Test with actual CryptoCompare API
- **Performance Testing**: Measure response times and throughput
- **Error Recovery**: Test fallback mechanisms

## Future Enhancements

### Potential Improvements

- **WebSocket Support**: Real-time data streaming
- **Multiple Data Sources**: Integration with other APIs
- **Advanced Caching**: Redis-based distributed caching
- **Data Validation**: Enhanced data quality checks

### Advanced Features

- **Predictive Caching**: Pre-fetch frequently requested data
- **Data Compression**: Compress cached data for memory efficiency
- **Analytics**: Track API usage and performance metrics
- **Webhook Support**: Real-time data updates via webhooks
