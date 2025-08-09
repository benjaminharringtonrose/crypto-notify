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
  start: string; // Start time of the candle
  low: string; // Lowest price during the period
  high: string; // Highest price during the period
  open: string; // Opening price
  close: string; // Closing price
  volume: string; // Trading volume
}
```

## Error Handling

### Comprehensive Error Catching

```typescript
try {
  const response = await this.client.getProductCandles(request);
  // Process response
} catch (error: any) {
  console.error("Failed to fetch market data:", error.message || error);
  throw error;
}
```

**Error Scenarios Handled**:

- **Network Failures**: Connection timeouts and network errors
- **API Errors**: Invalid requests, authentication failures
- **Data Parsing Errors**: Malformed responses or missing data
- **Rate Limiting**: API quota exceeded errors

**Error Propagation**:

- Logs detailed error information for debugging
- Re-throws errors to allow calling code to handle them appropriately
- Maintains error context for proper error handling upstream

## Performance Considerations

### Request Optimization

- **Efficient Data Fetching**: Uses appropriate granularity for data needs
- **Minimal API Calls**: Consolidates data requests when possible
- **Response Processing**: Efficient array operations for data transformation

### Memory Management

- **Streaming Processing**: Processes data in chunks to avoid memory issues
- **Garbage Collection**: Proper cleanup of temporary variables
- **Efficient Parsing**: Direct parsing without unnecessary object creation

## Security Considerations

### API Key Management

- **Environment Variables**: API credentials stored securely in environment
- **No Hardcoding**: Credentials never committed to source code
- **Access Control**: Minimal required permissions for API keys

### Request Validation

- **Input Sanitization**: Validates all input parameters
- **Type Safety**: TypeScript interfaces prevent invalid data types
- **Error Boundaries**: Graceful handling of malformed requests

## Integration Examples

### Trading Strategy Integration

```typescript
import { CoinbaseService } from "./api/CoinbaseService";
import { TradingStrategy } from "./cardano/TradingStrategy";

async function executeTradingStrategy() {
  const coinbaseService = new CoinbaseService({
    apiKey: process.env.COINBASE_API_KEY,
    apiSecret: process.env.COINBASE_API_SECRET,
  });

  const strategy = new TradingStrategy();

  // Get current market data
  const currentPrice = await coinbaseService.getCurrentPrice(
    CoinbaseProductIds.ADA
  );

  // Get historical data for analysis
  const now = Math.floor(Date.now() / 1000);
  const start = now - TIME_CONVERSIONS.TIMESTEP_IN_SECONDS;

  const adaData = await coinbaseService.getPricesAndVolumes({
    product_id: CoinbaseProductIds.ADA,
    granularity: Granularity.OneDay,
    start: start.toString(),
    end: now.toString(),
  });

  // Execute strategy with market data
  const tradeDecision = await strategy.decideTrade({
    currentPrice,
    historicalPrices: adaData.prices,
    historicalVolumes: adaData.volumes,
    // ... other parameters
  });

  return tradeDecision;
}
```

### Price Monitoring Service

```typescript
import { CoinbaseService } from "./api/CoinbaseService";
import { CoinbaseProductIds } from "../types";

class PriceMonitor {
  private coinbaseService: CoinbaseService;
  private monitoringInterval: NodeJS.Timeout;

  constructor() {
    this.coinbaseService = new CoinbaseService({
      apiKey: process.env.COINBASE_API_KEY,
      apiSecret: process.env.COINBASE_API_SECRET,
    });
  }

  startMonitoring(intervalMs: number = 60000) {
    this.monitoringInterval = setInterval(async () => {
      try {
        const adaPrice = await this.coinbaseService.getCurrentPrice(
          CoinbaseProductIds.ADA
        );
        const btcPrice = await this.coinbaseService.getCurrentPrice(
          CoinbaseProductIds.BTC
        );

        console.log(`ADA: $${adaPrice} | BTC: $${btcPrice}`);

        // Check for price alerts
        this.checkPriceAlerts(adaPrice, btcPrice);
      } catch (error) {
        console.error("Price monitoring failed:", error);
      }
    }, intervalMs);
  }

  stopMonitoring() {
    if (this.monitoringInterval) {
      clearInterval(this.monitoringInterval);
    }
  }

  private checkPriceAlerts(adaPrice: number, btcPrice: number) {
    // Implement price alert logic
    if (adaPrice > 1.0) {
      console.log("ADA price alert: Above $1.00");
    }
  }
}
```

### Backtesting Data Collection

```typescript
import { CoinbaseService } from "./api/CoinbaseService";
import { TIME_CONVERSIONS } from "../constants";

async function collectBacktestData(days: number = 365) {
  const coinbaseService = new CoinbaseService({
    apiKey: process.env.COINBASE_API_KEY,
    apiSecret: process.env.COINBASE_API_SECRET,
  });

  const now = Math.floor(Date.now() / 1000);
  const start = now - days * TIME_CONVERSIONS.ONE_DAY_IN_SECONDS;

  const [adaData, btcData] = await Promise.all([
    coinbaseService.getPricesAndVolumes({
      product_id: CoinbaseProductIds.ADA,
      granularity: Granularity.OneDay,
      start: start.toString(),
      end: now.toString(),
    }),
    coinbaseService.getPricesAndVolumes({
      product_id: CoinbaseProductIds.BTC,
      granularity: Granularity.OneDay,
      start: start.toString(),
      end: now.toString(),
    }),
  ]);

  return {
    ada: adaData,
    btc: btcData,
    metadata: {
      startDate: new Date(start * 1000),
      endDate: new Date(now * 1000),
      totalDays: days,
      dataPoints: adaData.prices.length,
    },
  };
}
```

## Testing

### Unit Testing Strategy

- **Mock API Responses**: Test with simulated Coinbase API responses
- **Error Scenarios**: Test various error conditions and edge cases
- **Data Processing**: Verify correct parsing and formatting of responses
- **Configuration**: Test different API key configurations

### Integration Testing

- **Live API Testing**: Test with actual Coinbase API (sandbox environment)
- **Rate Limiting**: Verify proper handling of API rate limits
- **Data Consistency**: Ensure data integrity across different time periods
- **Performance**: Measure response times and data processing efficiency

### Test Data Requirements

- **Valid Responses**: Real API response structures
- **Error Responses**: Various error scenarios from the API
- **Edge Cases**: Empty data, malformed responses, network timeouts
- **Performance Data**: Large datasets for stress testing

## Monitoring and Logging

### Request Logging

```typescript
console.log(`Fetching current price for ${product_id}`);
console.log(`Retrieved ${response.candles.length} candles`);
console.log(`Current price: $${currentPrice}`);
```

### Error Logging

```typescript
console.error("Failed to get current price:", error.message || error);
console.error("Failed to fetch market data:", error.message || error);
```

### Performance Metrics

- **Response Times**: Track API response latency
- **Success Rates**: Monitor API call success/failure rates
- **Data Volume**: Track amount of data processed
- **Error Frequency**: Monitor error patterns and frequency

## Future Enhancements

### Potential Improvements

- **Caching Layer**: Implement response caching for frequently requested data
- **Batch Requests**: Support for multiple product requests in single API call
- **WebSocket Integration**: Real-time price updates via WebSocket connections
- **Rate Limit Management**: Automatic backoff and retry logic

### Advanced Features

- **Data Compression**: Efficient storage and transmission of historical data
- **Predictive Caching**: Intelligent caching based on usage patterns
- **Multi-Exchange Support**: Integration with additional cryptocurrency exchanges
- **Advanced Analytics**: Built-in technical analysis and market insights

### Integration Enhancements

- **Event-Driven Architecture**: Publish price updates to message queues
- **Microservice Integration**: Service mesh integration for distributed systems
- **Cloud-Native Features**: Kubernetes and container orchestration support
- **Observability**: Enhanced metrics, tracing, and monitoring capabilities
