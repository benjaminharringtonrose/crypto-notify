# CoinbaseService Class Documentation

## Overview

The `CoinbaseService` class is a service layer component that provides a clean, type-safe interface for interacting with the Coinbase Advanced Trade API. It abstracts the complexity of API authentication, request formatting, and response parsing, offering simplified methods for retrieving cryptocurrency market data including current prices and historical price/volume information.

## Architecture

The CoinbaseService implements a client-wrapper pattern around the official Coinbase Advanced Trade API:

```
CoinbaseService → CBAdvancedTradeClient → Coinbase Advanced Trade API → Market Data
      ↓                    ↓                        ↓                    ↓
   Type Safety         Authentication          Real-time Data      Structured Response
   Error Handling      Request Formatting      Market Analysis     Data Processing
```

### Key Design Principles

- **Abstraction**: Simplifies complex API interactions into intuitive method calls
- **Type Safety**: Leverages TypeScript interfaces for robust data handling
- **Error Handling**: Comprehensive error catching and logging
- **Data Processing**: Automatic parsing and formatting of API responses
- **Rate Limiting**: Respects API rate limits through proper request management

## Class Structure

```typescript
export class CoinbaseService {
  private client: CBAdvancedTradeClient;

  constructor(config: CoinbaseClientConfig);

  public async getCurrentPrice(product_id: CoinbaseProductIds): Promise<number>;
  public async getPricesAndVolumes(
    request: GetAdvTradeProductCandlesRequest
  ): Promise<{ prices: number[]; volumes: number[] }>;
}
```

### Core Dependencies

- **coinbase-api**: Official Coinbase Advanced Trade API client
- **@tensorflow/tfjs-node**: TensorFlow.js for data processing (if needed)
- **TIME_CONVERSIONS**: Application constants for time calculations

## Configuration

### Constructor Parameters

```typescript
interface CoinbaseClientConfig {
  apiKey?: string; // Coinbase Advanced Trade API key
  apiSecret?: string; // Coinbase Advanced Trade API secret
}
```

**Authentication Setup**:

```typescript
const coinbaseService = new CoinbaseService({
  apiKey: process.env.COINBASE_API_KEY,
  apiSecret: process.env.COINBASE_API_SECRET,
});
```

**Environment Variables**:

- `COINBASE_API_KEY`: Your Coinbase Advanced Trade API key
- `COINBASE_API_SECRET`: Your Coinbase Advanced Trade API secret

## Core Methods

### Public Methods

#### `getCurrentPrice(product_id: CoinbaseProductIds): Promise<number>`

Retrieves the current market price for a specified trading pair.

**Parameters**:

- `product_id: CoinbaseProductIds` - Trading pair identifier (e.g., "ADA-USD", "BTC-USD")

**Returns**:

- `Promise<number>` - Current market price as a number

**Process Flow**:

1. **Time Calculation**: Determines the time range for recent market data

   ```typescript
   const now = Math.floor(
     Date.now() / TIME_CONVERSIONS.ONE_SECOND_IN_MILLISECONDS
   );
   const oneHourAgo = now - TIME_CONVERSIONS.ONE_HOUR_IN_SECONDS;
   ```

2. **API Request**: Fetches hourly candlestick data from Coinbase

   ```typescript
   const response = await this.client.getProductCandles({
     product_id,
     granularity: Granularity.OneHour,
     start: oneHourAgo.toString(),
     end: now.toString(),
   });
   ```

3. **Data Processing**: Extracts and processes closing prices

   ```typescript
   const prices = response.candles
     .map((candle) => parseFloat(candle.close))
     .reverse();
   const currentPrice = prices[prices.length - 1];
   ```

4. **Response**: Returns the most recent closing price

**Usage Example**:

```typescript
const currentPrice = await coinbaseService.getCurrentPrice(
  CoinbaseProductIds.ADA
);
console.log(`Current ADA price: $${currentPrice}`);
```

#### `getPricesAndVolumes(request: GetAdvTradeProductCandlesRequest): Promise<{ prices: number[]; volumes: number[] }>`

Retrieves historical price and volume data for a specified trading pair and time range.

**Parameters**:

- `request: GetAdvTradeProductCandlesRequest` - Complete request object with:
  - `product_id`: Trading pair identifier
  - `granularity`: Time interval (e.g., Granularity.OneDay)
  - `start`: Start timestamp (Unix seconds)
  - `end`: End timestamp (Unix seconds)

**Returns**:

- `Promise<{ prices: number[]; volumes: number[] }>` - Arrays of historical prices and volumes

**Process Flow**:

1. **API Request**: Sends request to Coinbase Advanced Trade API

   ```typescript
   const response = await this.client.getProductCandles(request);
   ```

2. **Data Extraction**: Processes candlestick data

   ```typescript
   const candles = response.candles;
   const prices = candles.map((candle) => parseFloat(candle.close)).reverse();
   const volumes = candles.map((candle) => parseFloat(candle.volume)).reverse();
   ```

3. **Data Formatting**: Returns structured arrays with chronological ordering

**Usage Example**:

```typescript
const now = Math.floor(Date.now() / 1000);
const start = now - TIME_CONVERSIONS.TIMESTEP_IN_SECONDS;

const marketData = await coinbaseService.getPricesAndVolumes({
  product_id: CoinbaseProductIds.ADA,
  granularity: Granularity.OneDay,
  start: start.toString(),
  end: now.toString(),
});

console.log(`Retrieved ${marketData.prices.length} price points`);
console.log(
  `Latest price: $${marketData.prices[marketData.prices.length - 1]}`
);
```

## Data Types and Interfaces

### CoinbaseProductIds

```typescript
enum CoinbaseProductIds {
  ADA = "ADA-USD", // Cardano to US Dollar
  BTC = "BTC-USD", // Bitcoin to US Dollar
  // ... other trading pairs
}
```

### Granularity

```typescript
enum Granularity {
  OneMinute = "ONE_MINUTE",
  FiveMinutes = "FIVE_MINUTE",
  FifteenMinutes = "FIFTEEN_MINUTE",
  ThirtyMinutes = "THIRTY_MINUTE",
  OneHour = "ONE_HOUR",
  TwoHours = "TWO_HOUR",
  SixHours = "SIX_HOUR",
  OneDay = "ONE_DAY",
}
```

### AdvTradeCandle

```typescript
interface AdvTradeCandle {
  start: string; // ISO 8601 timestamp
  low: string; // Lowest price during the interval
  high: string; // Highest price during the interval
  open: string; // Opening price
  close: string; // Closing price
  volume: string; // Trading volume
}
```

## Error Handling

### Comprehensive Error Management

The service implements robust error handling for various failure scenarios:

```typescript
try {
  const currentPrice = await coinbaseService.getCurrentPrice(
    CoinbaseProductIds.BTC
  );
} catch (error: any) {
  console.error("Failed to get current price:", error.message || error);
  // Handle specific error types
  if (error.code === "RATE_LIMIT_EXCEEDED") {
    // Implement retry logic with exponential backoff
  } else if (error.code === "INVALID_PRODUCT_ID") {
    // Handle invalid product ID
  }
}
```

### Common Error Scenarios

1. **Rate Limiting**: API rate limits exceeded
2. **Authentication**: Invalid API credentials
3. **Network Issues**: Connection timeouts or failures
4. **Invalid Parameters**: Incorrect product IDs or time ranges
5. **Data Parsing**: Malformed API responses

## Performance Considerations

### Caching Strategy

For high-frequency applications, consider implementing caching:

```typescript
// Example caching implementation
private priceCache = new Map<string, { price: number; timestamp: number }>();
private CACHE_DURATION = 60000; // 1 minute

public async getCurrentPriceWithCache(product_id: CoinbaseProductIds): Promise<number> {
  const cacheKey = product_id;
  const cached = this.priceCache.get(cacheKey);

  if (cached && Date.now() - cached.timestamp < this.CACHE_DURATION) {
    return cached.price;
  }

  const price = await this.getCurrentPrice(product_id);
  this.priceCache.set(cacheKey, { price, timestamp: Date.now() });
  return price;
}
```

### Rate Limiting

The service respects Coinbase API rate limits:

- **REST API**: 3 requests per second per API key
- **WebSocket**: 30 messages per second per connection
- **Automatic Backoff**: Implement exponential backoff for rate limit errors

## Integration Examples

### Trading System Integration

```typescript
// Integration with trading system
class TradingSystem {
  private coinbaseService: CoinbaseService;

  constructor() {
    this.coinbaseService = new CoinbaseService({
      apiKey: process.env.COINBASE_API_KEY,
      apiSecret: process.env.COINBASE_API_SECRET,
    });
  }

  async executeTrade(symbol: CoinbaseProductIds, action: "buy" | "sell") {
    const currentPrice = await this.coinbaseService.getCurrentPrice(symbol);
    // Execute trade logic based on current price
  }

  async analyzeMarket(symbol: CoinbaseProductIds) {
    const now = Math.floor(Date.now() / 1000);
    const start = now - 30 * 24 * 60 * 60; // 30 days ago

    const marketData = await this.coinbaseService.getPricesAndVolumes({
      product_id: symbol,
      granularity: Granularity.OneDay,
      start: start.toString(),
      end: now.toString(),
    });

    // Perform technical analysis on market data
    return this.calculateIndicators(marketData.prices, marketData.volumes);
  }
}
```

### Real-time Price Monitoring

```typescript
// Real-time price monitoring
class PriceMonitor {
  private coinbaseService: CoinbaseService;
  private monitoringInterval: NodeJS.Timeout | null = null;

  constructor() {
    this.coinbaseService = new CoinbaseService({
      apiKey: process.env.COINBASE_API_KEY,
      apiSecret: process.env.COINBASE_API_SECRET,
    });
  }

  startMonitoring(symbol: CoinbaseProductIds, intervalMs: number = 30000) {
    this.monitoringInterval = setInterval(async () => {
      try {
        const price = await this.coinbaseService.getCurrentPrice(symbol);
        this.onPriceUpdate(symbol, price);
      } catch (error) {
        console.error(`Failed to get price for ${symbol}:`, error);
      }
    }, intervalMs);
  }

  stopMonitoring() {
    if (this.monitoringInterval) {
      clearInterval(this.monitoringInterval);
      this.monitoringInterval = null;
    }
  }

  private onPriceUpdate(symbol: CoinbaseProductIds, price: number) {
    console.log(`${symbol}: $${price}`);
    // Implement price alert logic
  }
}
```

## Testing

### Unit Testing

```typescript
// Example unit test
describe("CoinbaseService", () => {
  let coinbaseService: CoinbaseService;

  beforeEach(() => {
    coinbaseService = new CoinbaseService({
      apiKey: "test-key",
      apiSecret: "test-secret",
    });
  });

  it("should get current price for BTC", async () => {
    const price = await coinbaseService.getCurrentPrice(CoinbaseProductIds.BTC);
    expect(typeof price).toBe("number");
    expect(price).toBeGreaterThan(0);
  });

  it("should handle API errors gracefully", async () => {
    // Mock API error
    jest
      .spyOn(coinbaseService["client"], "getProductCandles")
      .mockRejectedValue(new Error("API Error"));

    await expect(
      coinbaseService.getCurrentPrice(CoinbaseProductIds.BTC)
    ).rejects.toThrow("API Error");
  });
});
```

## Best Practices

### Security

1. **Environment Variables**: Never hardcode API credentials
2. **Credential Rotation**: Regularly rotate API keys
3. **Access Control**: Use minimal required permissions
4. **Error Logging**: Avoid logging sensitive data

### Performance

1. **Caching**: Implement appropriate caching for frequently accessed data
2. **Rate Limiting**: Respect API rate limits
3. **Connection Pooling**: Reuse HTTP connections when possible
4. **Error Handling**: Implement retry logic with exponential backoff

### Monitoring

1. **API Usage**: Monitor API call frequency and success rates
2. **Response Times**: Track API response times
3. **Error Rates**: Monitor error rates and types
4. **Data Quality**: Validate received data quality

## Troubleshooting

### Common Issues

1. **Authentication Errors**: Verify API key and secret are correct
2. **Rate Limit Errors**: Implement exponential backoff
3. **Network Timeouts**: Check network connectivity
4. **Data Parsing Errors**: Verify API response format

### Debug Mode

Enable debug logging for troubleshooting:

```typescript
// Enable debug mode
const coinbaseService = new CoinbaseService({
  apiKey: process.env.COINBASE_API_KEY,
  apiSecret: process.env.COINBASE_API_SECRET,
});

// Debug logging
console.log("API Configuration:", {
  hasApiKey: !!process.env.COINBASE_API_KEY,
  hasApiSecret: !!process.env.COINBASE_API_SECRET,
});
```

## Conclusion

The CoinbaseService provides a robust, type-safe interface for interacting with the Coinbase Advanced Trade API. Its comprehensive error handling, performance optimizations, and ease of integration make it an essential component for cryptocurrency trading applications.

The service abstracts away the complexity of API interactions while providing the flexibility needed for various trading scenarios. Whether used for real-time price monitoring, historical data analysis, or automated trading systems, the CoinbaseService delivers reliable and efficient access to cryptocurrency market data.
