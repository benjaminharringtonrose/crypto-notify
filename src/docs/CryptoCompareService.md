# CryptoCompareService Class Documentation

## Overview

The `CryptoCompareService` class is a simplified service that provides access to historical cryptocurrency market data through the CryptoCompare API. It offers a single method for retrieving historical price and volume data for Bitcoin and other cryptocurrencies used in the trading system.

## Architecture

The CryptoCompareService follows a simple RESTful API integration pattern:

```
API Requests → Chunked Data Processing → Rate Limiting → Historical Data
     ↓                    ↓                   ↓              ↓
CryptoCompare         JSON Parsing        Request Throttling  Price/Volume Arrays
REST Endpoints       Data Validation     50ms Delays        Chronological Order
```

### Key Responsibilities

- **Historical Data Retrieval**: Fetch historical price and volume data
- **Chunked Processing**: Handle large date ranges through chunk-based requests
- **API Integration**: Manage CryptoCompare API requests and responses
- **Error Handling**: Robust error management and data validation
- **Rate Limiting**: Respect API rate limits with request throttling

## Class Structure

```typescript
export class CryptoCompareService {
  private baseUrl: string;

  constructor();

  public async getHistoricalData(
    cryptoSymbol: string,
    totalDays: number,
    chunkDays?: number
  ): Promise<{ prices: number[]; volumes: number[] }>;
}
```

### Constructor Parameters

- No parameters required - uses default CryptoCompare endpoint

## Core Interfaces

### Return Type

```typescript
interface HistoricalDataResponse {
  prices: number[]; // Historical closing prices in chronological order
  volumes: number[]; // Historical volume data in chronological order
}
```

## Methods

### Public Methods

#### `getHistoricalData(cryptoSymbol, totalDays, chunkDays): Promise<{prices: number[], volumes: number[]}>`

Retrieves historical price and volume data for a specified cryptocurrency using chunked requests to handle large date ranges.

**Parameters:**

- `cryptoSymbol: string` - Cryptocurrency symbol (e.g., "BTC", "ETH", "ADA")
- `totalDays: number` - Total number of days of historical data to retrieve
- `chunkDays: number` - Optional chunk size for API requests (default: 90 days)

**Returns:**

- `Promise<{prices: number[], volumes: number[]}>` - Historical price and volume arrays

**Process:**

1. **Chunk Calculation**: Divides total days into manageable chunks (90 days by default)
2. **API Requests**: Makes sequential API requests with 50ms throttling
3. **Data Processing**: Parses and validates response data from each chunk
4. **Data Merging**: Combines chunks in chronological order using `unshift`
5. **Error Handling**: Validates API responses and data structure

## Data Flow

### 1. Chunked Data Processing

```typescript
// Calculate chunk parameters
const endDate = new Date();
const totalMilliseconds = totalDays * TIME_CONVERSIONS.ONE_DAY_IN_MILLISECONDS;
const chunkMilliseconds = chunkDays * TIME_CONVERSIONS.ONE_DAY_IN_MILLISECONDS;
const numChunks = Math.ceil(totalDays / chunkDays);

// Process each chunk
for (let i = 0; i < numChunks; i++) {
  const chunkEnd = new Date(endDate.getTime() - i * chunkMilliseconds);
  const chunkStart = new Date(chunkEnd.getTime() - chunkMilliseconds);

  // Add 50ms delay for rate limiting
  if (i > 0) await delay(50);

  // Make API request
  const response = await axios.get(`${this.baseUrl}/histoday`, {
    params: {
      fsym: cryptoSymbol.toUpperCase(),
      tsym: "USD",
      limit: limit,
      toTs: toTimestamp,
      api_key: process.env.CRYPTOCOMPARE_API_KEY,
    },
  });
}
```

### 2. Data Validation and Processing

```typescript
// Validate API response structure
if (
  !response.data ||
  !response.data.Data ||
  !Array.isArray(response.data.Data.Data)
) {
  throw new Error(
    `Invalid API response structure: ${JSON.stringify(response.data)}`
  );
}

// Extract prices and volumes
const chunkPrices = data.map((entry: any) => {
  if (entry.close === undefined) {
    throw new Error(`Missing 'close' price in entry: ${JSON.stringify(entry)}`);
  }
  return entry.close;
});

const chunkVolumes = data.map((entry: any) => {
  if (entry.volumeto === undefined) {
    throw new Error(`Missing 'volumeto' in entry: ${JSON.stringify(entry)}`);
  }
  return entry.volumeto;
});
```

### 3. Data Merging

```typescript
// Merge chunks in chronological order
prices.unshift(...chunkPrices);
volumes.unshift(...chunkVolumes);

// Final validation
if (prices.length === 0 || volumes.length === 0) {
  throw new Error(`No valid data retrieved for ${cryptoSymbol}`);
}

return { prices, volumes };
```

## Usage Examples

### Basic Historical Data Retrieval

```typescript
import { CryptoCompareService } from "./CryptoCompareService";

const cryptoCompare = new CryptoCompareService();

// Retrieve 1 year of daily data for BTC
const { prices, volumes } = await cryptoCompare.getHistoricalData("BTC", 365);

console.log(`Retrieved ${prices.length} days of BTC data`);
console.log(`Latest price: $${prices[prices.length - 1]}`);
console.log(`Latest volume: ${volumes[volumes.length - 1]}`);
```

### Custom Chunk Size

```typescript
// Retrieve 2 years of data with 60-day chunks
const { prices, volumes } = await cryptoCompare.getHistoricalData(
  "BTC",
  730, // 2 years
  60 // 60-day chunks
);

console.log(`Retrieved ${prices.length} days with custom chunking`);
```

### Integration with DataProcessor

```typescript
import { DataProcessor } from "./bitcoin/DataProcessor";

// Use CryptoCompareService with DataProcessor
const cryptoCompare = new CryptoCompareService();

// Fetch historical data for the DataProcessor
const btcData = await cryptoCompare.getHistoricalData("BTC", 1200); // ~3.3 years

console.log(`Fetched ${btcData.prices.length} days for training`);
```

## Configuration

### API Configuration

```typescript
const API_CONFIG = {
  BASE_URL: "https://min-api.cryptocompare.com/data/v2",
  DEFAULT_CHUNK_DAYS: 90, // Days per API request chunk
  RATE_LIMIT_DELAY: 50, // Milliseconds between requests
  DEFAULT_SYMBOL: "USD", // Base currency for price data
};
```

### Environment Variables

```bash
CRYPTOCOMPARE_API_KEY=your_cryptocompare_api_key
```

## Error Handling

### API Error Management

```typescript
try {
  const response = await axios.get(`${this.baseUrl}/histoday`, {
    params: {
      fsym: cryptoSymbol.toUpperCase(),
      tsym: "USD",
      limit: limit,
      toTs: toTimestamp,
      api_key: process.env.CRYPTOCOMPARE_API_KEY,
    },
  });
} catch (error) {
  console.error(`Error fetching historical data for ${cryptoSymbol}:`, error);
  throw error;
}
```

### Data Validation

```typescript
// Validate API response structure
if (
  !response.data ||
  !response.data.Data ||
  !Array.isArray(response.data.Data.Data)
) {
  throw new Error(
    `Invalid API response structure: ${JSON.stringify(response.data)}`
  );
}

// Validate required fields
const chunkPrices = data.map((entry: any) => {
  if (entry.close === undefined) {
    throw new Error(`Missing 'close' price in entry: ${JSON.stringify(entry)}`);
  }
  return entry.close;
});
```

### Rate Limiting

- **Fixed Delay**: 50ms delay between requests
- **Sequential Processing**: Chunks processed sequentially to avoid overwhelming API
- **Respect Limits**: Stays well below CryptoCompare's rate limits

## Performance Considerations

### Chunking Strategy

- **Default 90-day chunks**: Balances API efficiency with memory usage
- **Configurable chunk size**: Can be adjusted based on requirements
- **Sequential processing**: Prevents API rate limit violations

### Memory Management

- **Efficient array operations**: Uses unshift for chronological ordering
- **Minimal memory allocation**: Processes chunks incrementally
- **No caching**: Simple implementation without memory overhead

## Dependencies

### External Dependencies

- **axios**: HTTP client for API requests
- **TIME_CONVERSIONS**: Time utility constants from application

### Internal Dependencies

- **Environment variables**: CRYPTOCOMPARE_API_KEY for API access

## Testing

### Unit Testing Strategy

- **API Mocking**: Mock CryptoCompare API responses
- **Error Scenarios**: Test various error conditions and data validation
- **Chunking Logic**: Verify correct chunk calculation and processing

### Integration Testing

- **Real API Calls**: Test with actual CryptoCompare API
- **Rate Limiting**: Verify throttling behavior
- **Data Integrity**: Validate chronological ordering and completeness

## Future Enhancements

### Potential Improvements

- **Caching Layer**: Add optional response caching for frequently requested data
- **Parallel Processing**: Process multiple chunks concurrently with rate limiting
- **Retry Logic**: Add automatic retry for failed requests
- **Multiple Symbols**: Support batch requests for multiple cryptocurrencies
