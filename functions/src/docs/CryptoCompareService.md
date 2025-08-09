# CryptoCompareService Class Documentation

## Overview

The `CryptoCompareService` class is a specialized service component that provides access to historical cryptocurrency market data through the CryptoCompare API. It implements intelligent data chunking, rate limiting, and robust error handling to efficiently retrieve large datasets of historical prices and volumes for cryptocurrency trading analysis and machine learning model training.

## Architecture

The CryptoCompareService implements a sophisticated data retrieval system with chunked requests and rate limiting:

```
CryptoCompareService → CryptoCompare API → Historical Data → Processed Arrays
        ↓                      ↓                ↓                ↓
   Chunked Requests      Rate Limiting      Data Validation    Price/Volume
   Time Management      Error Handling      Response Parsing    Arrays
```

### Key Design Principles

- **Efficient Data Retrieval**: Chunks large time ranges into manageable API requests
- **Rate Limit Compliance**: Respects API rate limits with intelligent throttling
- **Robust Error Handling**: Comprehensive error catching and validation
- **Data Integrity**: Ensures complete and accurate historical data retrieval
- **Memory Efficiency**: Processes data in chunks to avoid memory issues

## Class Structure

```typescript
export class CryptoCompareService {
  private baseUrl: string = "https://min-api.cryptocompare.com/data/v2";

  public constructor();

  public async getHistoricalData(
    cryptoSymbol: string,
    totalDays: number,
    chunkDays: number = 90
  ): Promise<{ prices: number[]; volumes: number[] }>;
}
```

### Core Dependencies

- **axios**: HTTP client for API requests
- **TIME_CONVERSIONS**: Application constants for time calculations
- **CryptoCompare API**: External service for historical market data

## Configuration

### Constructor

The service uses a simple constructor with no parameters, initializing with the CryptoCompare API base URL.

```typescript
public constructor() {
  // No configuration required - uses default CryptoCompare API endpoint
}
```

**Base URL**: `https://min-api.cryptocompare.com/data/v2`

**API Key**: Retrieved from environment variable `CRYPTOCOMPARE_API_KEY`

## Core Methods

### Public Methods

#### `getHistoricalData(cryptoSymbol: string, totalDays: number, chunkDays: number = 90): Promise<{ prices: number[]; volumes: number[] }>`

Retrieves historical price and volume data for a specified cryptocurrency over a given time period, automatically chunking large requests to comply with API limits.

**Parameters**:

- `cryptoSymbol: string` - Cryptocurrency symbol (e.g., "ADA", "BTC", "ETH")
- `totalDays: number` - Total number of days of historical data to retrieve
- `chunkDays: number = 90` - Number of days per API request chunk (default: 90)

**Returns**:

- `Promise<{ prices: number[]; volumes: number[] }>` - Arrays of historical prices and volumes in chronological order

**Process Flow**:

1. **Initialization**: Sets up data arrays and calculates time ranges

   ```typescript
   const prices: number[] = [];
   const volumes: number[] = [];
   const endDate = new Date();
   const totalMilliseconds =
     totalDays * TIME_CONVERSIONS.ONE_DAY_IN_MILLISECONDS;
   const chunkMilliseconds =
     chunkDays * TIME_CONVERSIONS.ONE_DAY_IN_MILLISECONDS;
   ```

2. **Chunk Calculation**: Determines number of API requests needed

   ```typescript
   const numChunks = Math.ceil(totalDays / chunkDays);
   ```

3. **Iterative Data Retrieval**: Processes each chunk sequentially

   ```typescript
   for (let i = 0; i < numChunks; i++) {
     const chunkEnd = new Date(endDate.getTime() - i * chunkMilliseconds);
     const chunkStart = new Date(chunkEnd.getTime() - chunkMilliseconds);
     // ... chunk processing
   }
   ```

4. **Rate Limiting**: Implements delays between requests

   ```typescript
   if (i > 0) await delay(50); // 50ms delay ensures < 20 calls/second
   ```

5. **API Request**: Sends request to CryptoCompare API

   ```typescript
   const response = await axios.get(`${this.baseUrl}/histoday`, {
     params: {
       fsym: cryptoSymbol.toUpperCase(),
       tsym: "USD",
       limit: limit,
       toTs: toTimestamp,
       api_key: process.env.CRYPTOCOMPARE_API_KEY,
     },
   });
   ```

6. **Data Validation**: Ensures response structure integrity

   ```typescript
   if (
     !response.data ||
     !response.data.Data ||
     !Array.isArray(response.data.Data.Data)
   ) {
     throw new Error(
       `Invalid API response structure: ${JSON.stringify(response.data)}`
     );
   }
   ```

7. **Data Extraction**: Processes candlestick data

   ```typescript
   const chunkPrices = data.map((entry: any) => {
     if (entry.close === undefined) {
       throw new Error(
         `Missing 'close' price in entry: ${JSON.stringify(entry)}`
       );
     }
     return entry.close;
   });
   ```

8. **Data Assembly**: Combines chunks into final arrays
   ```typescript
   prices.unshift(...chunkPrices);
   volumes.unshift(...chunkVolumes);
   ```

**Usage Example**:

```typescript
import { CryptoCompareService } from "./api/CryptoCompareService";

const cryptoCompare = new CryptoCompareService();

// Retrieve 1 year of daily data for ADA
const adaData = await cryptoCompare.getHistoricalData("ADA", 365);

console.log(`Retrieved ${adaData.prices.length} days of ADA data`);
console.log(`Latest price: $${adaData.prices[adaData.prices.length - 1]}`);
console.log(`Latest volume: ${adaData.volumes[adaData.volumes.length - 1]}`);
```

## Data Processing

### Chunking Strategy

The service automatically breaks large time ranges into smaller chunks to comply with API limits and ensure reliable data retrieval.

**Chunk Size Calculation**:

```typescript
const chunkDays = 90; // Default chunk size
const numChunks = Math.ceil(totalDays / chunkDays);
```

**Time Range Management**:

```typescript
const chunkEnd = new Date(endDate.getTime() - i * chunkMilliseconds);
const chunkStart = new Date(chunkEnd.getTime() - chunkMilliseconds);
const actualChunkStart =
  chunkStart < new Date(endDate.getTime() - totalMilliseconds)
    ? new Date(endDate.getTime() - totalMilliseconds)
    : chunkStart;
```

### Rate Limiting

Implements intelligent throttling to respect API rate limits and ensure reliable data retrieval.

**Delay Implementation**:

```typescript
const delay = (ms: number) => new Promise((resolve) => setTimeout(resolve, ms));

// Add delay before each request (50ms ensures < 20 calls/second)
if (i > 0) await delay(50);
```

**Rate Limit Compliance**:

- **Default Delay**: 50ms between requests
- **Maximum Rate**: < 20 calls per second
- **Configurable**: Can be adjusted based on API tier limits

### Data Validation

Comprehensive validation ensures data integrity and handles various error scenarios gracefully.

**Response Structure Validation**:

```typescript
if (
  !response.data ||
  !response.data.Data ||
  !Array.isArray(response.data.Data.Data)
) {
  throw new Error(
    `Invalid API response structure: ${JSON.stringify(response.data)}`
  );
}
```

**Data Field Validation**:

```typescript
const chunkPrices = data.map((entry: any) => {
  if (entry.close === undefined) {
    throw new Error(`Missing 'close' price in entry: ${JSON.stringify(entry)}`);
  }
  return entry.close;
});
```

**Empty Data Handling**:

```typescript
if (data.length === 0) {
  console.warn(`Empty data array received for ${cryptoSymbol} chunk ${i + 1}`);
  continue;
}
```

## Error Handling

### Comprehensive Error Scenarios

The service handles various error conditions with appropriate logging and error propagation.

**API Response Errors**:

```typescript
if (
  !response.data ||
  !response.data.Data ||
  !Array.isArray(response.data.Data.Data)
) {
  throw new Error(
    `Invalid API response structure: ${JSON.stringify(response.data)}`
  );
}
```

**Missing Data Fields**:

```typescript
if (entry.close === undefined) {
  throw new Error(`Missing 'close' price in entry: ${JSON.stringify(entry)}`);
}
```

**Empty Data Arrays**:

```typescript
if (prices.length === 0 || volumes.length === 0) {
  throw new Error(`No valid data retrieved for ${cryptoSymbol}`);
}
```

**Network and General Errors**:

```typescript
} catch (error) {
  console.error(`Error fetching historical data for ${cryptoSymbol}:`, error);
  throw error;
}
```

### Error Propagation

- **Detailed Logging**: Comprehensive error information for debugging
- **Error Context**: Includes cryptocurrency symbol and chunk information
- **Graceful Degradation**: Continues processing when possible
- **Upstream Handling**: Propagates errors to calling code for appropriate handling

## Performance Considerations

### Request Optimization

- **Efficient Chunking**: Optimal chunk sizes for API performance
- **Parallel Processing**: Sequential processing to avoid rate limit issues
- **Memory Management**: Processes data in chunks to avoid memory overflow

### Data Processing Efficiency

- **Streaming Assembly**: Builds result arrays incrementally
- **Minimal Object Creation**: Direct array operations without unnecessary objects
- **Efficient Parsing**: Direct field access and type conversion

## Security Considerations

### API Key Management

- **Environment Variables**: API key stored securely in environment
- **No Hardcoding**: Credentials never committed to source code
- **Minimal Permissions**: Read-only access to historical data

### Request Validation

- **Input Sanitization**: Validates cryptocurrency symbols and time ranges
- **Type Safety**: TypeScript interfaces prevent invalid parameter types
- **Error Boundaries**: Graceful handling of malformed requests

## Integration Examples

### Machine Learning Data Collection

```typescript
import { CryptoCompareService } from "./api/CryptoCompareService";
import { DataProcessor } from "./cardano/DataProcessor";

async function prepareTrainingData() {
  const cryptoCompare = new CryptoCompareService();

  // Collect comprehensive historical data
  const [adaData, btcData] = await Promise.all([
    cryptoCompare.getHistoricalData("ADA", 450), // 1.25 years
    cryptoCompare.getHistoricalData("BTC", 450),
  ]);

  // Process data for machine learning
  const dataProcessor = new DataProcessor(
    { timesteps: 30, featureCount: 20 },
    450
  );

  const processedData = await dataProcessor.processData();
  return processedData;
}
```

### Backtesting Data Retrieval

```typescript
import { CryptoCompareService } from "./api/CryptoCompareService";
import { TradeModelBacktester } from "./cardano/TradeModelBacktester";

async function runComprehensiveBacktest() {
  const cryptoCompare = new CryptoCompareService();
  const backtester = new TradeModelBacktester(10000);

  // Collect data for different time periods
  const recentData = await cryptoCompare.getHistoricalData("ADA", 180); // 6 months
  const historicalData = await cryptoCompare.getHistoricalData("ADA", 365); // 1 year

  // Run backtests on different datasets
  const recentResult = await backtester.backtest(
    recentData,
    btcData,
    30,
    recentData.prices.length - 1
  );
  const historicalResult = await backtester.backtest(
    historicalData,
    btcData,
    30,
    historicalData.prices.length - 1
  );

  return { recentResult, historicalResult };
}
```

### Market Analysis Service

```typescript
import { CryptoCompareService } from "./api/CryptoCompareService";

class MarketAnalyzer {
  private cryptoCompare: CryptoCompareService;

  constructor() {
    this.cryptoCompare = new CryptoCompareService();
  }

  async analyzeTrends(symbol: string, days: number = 90) {
    const data = await this.cryptoCompare.getHistoricalData(symbol, days);

    // Calculate trend indicators
    const trendSlope = this.calculateTrendSlope(data.prices);
    const volumeTrend = this.calculateVolumeTrend(data.volumes);
    const volatility = this.calculateVolatility(data.prices);

    return {
      symbol,
      period: days,
      trendSlope,
      volumeTrend,
      volatility,
      dataPoints: data.prices.length,
    };
  }

  private calculateTrendSlope(prices: number[]): number {
    // Linear regression slope calculation
    const n = prices.length;
    const xMean = (n - 1) / 2;
    const yMean = prices.reduce((sum, price) => sum + price, 0) / n;

    let numerator = 0;
    let denominator = 0;

    for (let i = 0; i < n; i++) {
      const x = i - xMean;
      const y = prices[i] - yMean;
      numerator += x * y;
      denominator += x * x;
    }

    return denominator !== 0 ? numerator / denominator : 0;
  }

  private calculateVolumeTrend(volumes: number[]): number {
    // Volume trend calculation
    const recentVolumes = volumes.slice(-30);
    const olderVolumes = volumes.slice(0, 30);

    const recentAvg =
      recentVolumes.reduce((sum, vol) => sum + vol, 0) / recentVolumes.length;
    const olderAvg =
      olderVolumes.reduce((sum, vol) => sum + vol, 0) / olderVolumes.length;

    return olderAvg !== 0 ? ((recentAvg - olderAvg) / olderAvg) * 100 : 0;
  }

  private calculateVolatility(prices: number[]): number {
    // Price volatility calculation
    const returns = [];
    for (let i = 1; i < prices.length; i++) {
      returns.push((prices[i] - prices[i - 1]) / prices[i - 1]);
    }

    const meanReturn =
      returns.reduce((sum, ret) => sum + ret, 0) / returns.length;
    const variance =
      returns.reduce((sum, ret) => sum + Math.pow(ret - meanReturn, 2), 0) /
      returns.length;

    return Math.sqrt(variance);
  }
}
```

### Data Quality Monitoring

```typescript
import { CryptoCompareService } from "./api/CryptoCompareService";

class DataQualityMonitor {
  private cryptoCompare: CryptoCompareService;

  constructor() {
    this.cryptoCompare = new CryptoCompareService();
  }

  async validateDataQuality(symbol: string, days: number = 30) {
    try {
      const data = await this.cryptoCompare.getHistoricalData(symbol, days);

      const qualityMetrics = {
        completeness: this.checkCompleteness(data),
        consistency: this.checkConsistency(data),
        accuracy: this.checkAccuracy(data),
        timeliness: this.checkTimeliness(data),
      };

      return {
        symbol,
        period: days,
        dataPoints: data.prices.length,
        qualityMetrics,
        overallScore: this.calculateOverallScore(qualityMetrics),
      };
    } catch (error) {
      console.error(`Data quality validation failed for ${symbol}:`, error);
      throw error;
    }
  }

  private checkCompleteness(data: {
    prices: number[];
    volumes: number[];
  }): number {
    const expectedPoints = data.prices.length;
    const actualPoints = data.prices.filter(
      (price) => price !== null && price !== undefined
    ).length;
    return (actualPoints / expectedPoints) * 100;
  }

  private checkConsistency(data: {
    prices: number[];
    volumes: number[];
  }): number {
    // Check for price anomalies (e.g., negative prices, extreme spikes)
    const anomalies = data.prices.filter(
      (price) => price <= 0 || price > 1000000
    ).length;
    return Math.max(0, 100 - (anomalies / data.prices.length) * 100);
  }

  private checkAccuracy(data: { prices: number[]; volumes: number[] }): number {
    // Check for reasonable price movements (e.g., no 1000% daily changes)
    let accuracyScore = 100;
    for (let i = 1; i < data.prices.length; i++) {
      const change = Math.abs(
        (data.prices[i] - data.prices[i - 1]) / data.prices[i - 1]
      );
      if (change > 1) {
        // More than 100% change
        accuracyScore -= 5;
      }
    }
    return Math.max(0, accuracyScore);
  }

  private checkTimeliness(data: {
    prices: number[];
    volumes: number[];
  }): number {
    // Check if data is recent (within last 24 hours for daily data)
    return 100; // Assuming data is always current when retrieved
  }

  private calculateOverallScore(metrics: any): number {
    const weights = {
      completeness: 0.3,
      consistency: 0.3,
      accuracy: 0.3,
      timeliness: 0.1,
    };
    return Object.keys(metrics).reduce((score, key) => {
      return score + metrics[key] * weights[key];
    }, 0);
  }
}
```

## Testing

### Unit Testing Strategy

- **Mock API Responses**: Test with simulated CryptoCompare API responses
- **Chunking Logic**: Verify correct chunk calculation and processing
- **Error Scenarios**: Test various error conditions and edge cases
- **Data Processing**: Verify correct parsing and array assembly

### Integration Testing

- **Live API Testing**: Test with actual CryptoCompare API
- **Rate Limiting**: Verify proper delay implementation
- **Large Datasets**: Test with extended time ranges
- **Data Consistency**: Ensure data integrity across different time periods

### Test Data Requirements

- **Valid Responses**: Real API response structures
- **Error Responses**: Various error scenarios from the API
- **Edge Cases**: Empty data, malformed responses, network timeouts
- **Performance Data**: Large datasets for stress testing

## Monitoring and Logging

### Request Logging

```typescript
console.log(
  `Fetching ${cryptoSymbol} data (Timestamp ${currentStart} to ${currentEnd})...`
);
console.log(`Fetched chunk for ${cryptoSymbol}: ${prices.length} candles`);
```

### Error Logging

```typescript
console.error(`Error fetching historical data for ${cryptoSymbol}:`, error);
console.warn(`Empty data array received for ${cryptoSymbol} chunk ${i + 1}`);
```

### Performance Metrics

- **Request Count**: Track number of API calls made
- **Processing Time**: Measure data retrieval and processing duration
- **Data Volume**: Track amount of data retrieved
- **Error Frequency**: Monitor error patterns and frequency

## Future Enhancements

### Potential Improvements

- **Caching Layer**: Implement response caching for frequently requested data
- **Parallel Processing**: Support for concurrent chunk processing (with rate limit awareness)
- **Data Compression**: Efficient storage and transmission of historical data
- **WebSocket Integration**: Real-time data updates for recent periods

### Advanced Features

- **Intelligent Chunking**: Dynamic chunk size optimization based on API performance
- **Predictive Caching**: Cache frequently accessed time periods
- **Multi-Currency Support**: Batch requests for multiple cryptocurrencies
- **Advanced Analytics**: Built-in technical analysis and market insights

### Integration Enhancements

- **Event-Driven Architecture**: Publish data updates to message queues
- **Microservice Integration**: Service mesh integration for distributed systems
- **Cloud-Native Features**: Kubernetes and container orchestration support
- **Observability**: Enhanced metrics, tracing, and monitoring capabilities
