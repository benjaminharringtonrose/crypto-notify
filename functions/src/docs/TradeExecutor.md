# TradeExecutor Class Documentation

## Overview

The `TradeExecutor` class is a core component of the Cardano trading system that handles the execution of trading decisions and market interactions through the Coinbase Advanced Trade API. It provides a secure, reliable interface for executing trades, fetching market data, and managing account balances in real-time cryptocurrency trading operations.

## Architecture

The TradeExecutor follows a client-server architecture with secure API integration:

```
Trading Strategy → TradeExecutor → Coinbase Advanced Trade API → Market Execution
      ↓                ↓                        ↓                    ↓
Recommendations    Order Management         Real-time Data      Trade Confirmation
Risk Assessment    Price Validation         Market Analysis     Balance Updates
```

### Key Responsibilities

- **Trade Execution**: Execute buy/sell orders with proper order management
- **Market Data Retrieval**: Fetch real-time price and volume information
- **Account Management**: Monitor and retrieve account balances
- **API Integration**: Secure communication with Coinbase Advanced Trade
- **Error Handling**: Robust error handling and logging for trading operations
- **Order Validation**: Ensure proper order parameters and market conditions

## Class Structure

```typescript
export class TradeExecutor {
  private client: CBAdvancedTradeClient;

  constructor(config: CoinbaseClientConfig);

  public async executeTrade(trade: Trade): Promise<any>;
  public async getCurrentPrice(product_id: CoinbaseProductIds): Promise<number>;
  public async getMarketData(
    request: GetAdvTradeProductCandlesRequest
  ): Promise<{ prices: number[]; volumes: number[] }>;
  public async getAccountBalances(): Promise<{
    usd?: AdvTradeAccount;
    ada?: AdvTradeAccount;
  }>;
}
```

### Constructor Parameters

```typescript
interface CoinbaseClientConfig {
  apiKey?: string; // Coinbase Advanced Trade API key
  apiSecret?: string; // Coinbase Advanced Trade API secret
}
```

## Core Methods

### Public Methods

#### `executeTrade(trade: Trade): Promise<any>`

Executes a trading order based on the provided trade recommendation.

**Parameters:**

- `trade: Trade` - Trade object containing type, amount, and price information

**Returns:**

- `Promise<any>` - API response from Coinbase or undefined for hold recommendations

**Process:**

1. **Recommendation Validation**: Checks if trade type is not "Hold"
2. **Order Construction**: Builds order parameters with proper configuration
3. **Market Order Execution**: Submits immediate-or-cancel (IOC) market order
4. **Response Handling**: Returns API response or throws error on failure

**Order Parameters:**

```typescript
const orderParams = {
  product_id: CoinbaseProductIds.ADA, // ADA-USD trading pair
  side: trade.type, // Buy or Sell
  order_configuration: {
    market_market_ioc: {
      // Immediate-or-Cancel market order
      base_size: trade.adaAmount.toString(), // Amount of ADA to trade
    },
  },
  client_order_id: crypto.randomUUID(), // Unique order identifier
};
```

**Trade Types Supported:**

- `Recommendation.Buy` - Purchase ADA with USD
- `Recommendation.Sell` - Sell ADA for USD
- `Recommendation.Hold` - No action taken

#### `getCurrentPrice(product_id: CoinbaseProductIds): Promise<number>`

Retrieves the current market price for a specified trading pair.

**Parameters:**

- `product_id: CoinbaseProductIds` - Trading pair identifier (e.g., ADA-USD)

**Returns:**

- `Promise<number>` - Current market price as a number

**Process:**

1. **Time Calculation**: Determines time range for recent market data
2. **Market Data Fetch**: Retrieves hourly candlestick data
3. **Price Extraction**: Extracts most recent closing price
4. **Logging**: Logs current price with timestamp
5. **Error Handling**: Throws error on API failure

**Time Range:**

- **Duration**: Last hour of market data
- **Granularity**: One-hour candlesticks
- **Data Source**: Coinbase Advanced Trade API

#### `getMarketData(request: GetAdvTradeProductCandlesRequest): Promise<{ prices: number[]; volumes: number[] }>`

Fetches historical market data including prices and volumes.

**Parameters:**

- `request: GetAdvTradeProductCandlesRequest` - Market data request parameters

**Returns:**

- `Promise<{ prices: number[]; volumes: number[] }>` - Arrays of historical prices and volumes

**Process:**

1. **API Request**: Submits request to Coinbase Advanced Trade API
2. **Data Processing**: Extracts close prices and volumes from candles
3. **Array Reversal**: Reverses arrays to maintain chronological order
4. **Error Handling**: Throws error on API failure

**Data Structure:**

```typescript
interface MarketDataResponse {
  prices: number[]; // Historical closing prices
  volumes: number[]; // Historical trading volumes
}
```

#### `getAccountBalances(): Promise<{ usd?: AdvTradeAccount; ada?: AdvTradeAccount }>`

Retrieves current account balances for USD and ADA holdings.

**Returns:**

- `Promise<{ usd?: AdvTradeAccount; ada?: AdvTradeAccount }>` - Account balance information

**Process:**

1. **Account Fetch**: Retrieves all accounts from Coinbase
2. **Currency Filtering**: Separates USD and ADA accounts
3. **Balance Assembly**: Returns structured balance object
4. **Error Handling**: Throws error on API failure

**Account Structure:**

```typescript
interface AccountBalances {
  usd?: AdvTradeAccount; // USD account information
  ada?: AdvTradeAccount; // ADA account information
}
```

## Trading Operations

### Order Execution Flow

```typescript
// 1. Receive trade recommendation
const trade: Trade = {
  type: Recommendation.Buy,
  adaAmount: 100,
  price: 0.5,
};

// 2. Execute trade
try {
  const result = await tradeExecutor.executeTrade(trade);
  console.log("Trade executed successfully:", result);
} catch (error) {
  console.error("Trade execution failed:", error);
}
```

### Market Data Retrieval

```typescript
// Get current ADA price
const currentPrice = await tradeExecutor.getCurrentPrice(
  CoinbaseProductIds.ADA
);
console.log(`Current ADA price: $${currentPrice}`);

// Get historical market data
const marketData = await tradeExecutor.getMarketData({
  product_id: CoinbaseProductIds.ADA,
  granularity: Granularity.OneHour,
  start: "1640995200", // Unix timestamp
  end: "1640998800", // Unix timestamp
});

console.log(`Prices: ${marketData.prices.length} data points`);
console.log(`Volumes: ${marketData.volumes.length} data points`);
```

### Account Balance Monitoring

```typescript
// Get current account balances
const balances = await tradeExecutor.getAccountBalances();

if (balances.usd) {
  console.log(`USD Balance: $${balances.usd.available_balance.value}`);
}

if (balances.ada) {
  console.log(`ADA Balance: ${balances.ada.available_balance.value} ADA`);
}
```

## Security and Authentication

### API Key Management

- **Secure Storage**: API keys should be stored in environment variables
- **Access Control**: Limited API permissions for trading operations
- **Key Rotation**: Regular API key updates for security

### Order Security

- **Unique IDs**: Each order gets a cryptographically secure UUID
- **Validation**: Order parameters validated before submission
- **Error Handling**: Secure error messages without exposing sensitive data

## Error Handling

### API Error Management

- **Network Failures**: Handles connection timeouts and network issues
- **API Limits**: Respects Coinbase API rate limits
- **Authentication Errors**: Handles invalid or expired API credentials
- **Market Errors**: Manages market-specific trading errors

### Error Recovery

- **Retry Logic**: Implements exponential backoff for transient failures
- **Fallback Mechanisms**: Graceful degradation on service unavailability
- **Logging**: Comprehensive error logging for debugging and monitoring

### Common Error Scenarios

```typescript
try {
  await tradeExecutor.executeTrade(trade);
} catch (error: any) {
  if (error.message.includes("insufficient_funds")) {
    console.error("Insufficient funds for trade");
  } else if (error.message.includes("market_closed")) {
    console.error("Market is currently closed");
  } else if (error.message.includes("rate_limit")) {
    console.error("API rate limit exceeded");
  } else {
    console.error("Unexpected error:", error.message);
  }
}
```

## Performance Considerations

### API Optimization

- **Request Batching**: Efficient market data retrieval
- **Caching**: Implement caching for frequently accessed data
- **Connection Pooling**: Reuse API connections when possible

### Memory Management

- **Data Processing**: Efficient array operations for market data
- **Resource Cleanup**: Proper disposal of API responses
- **Streaming**: Consider streaming for large datasets

## Dependencies

### External Dependencies

- **coinbase-api**: Official Coinbase Advanced Trade API client
- **crypto**: Node.js built-in cryptographic functions
- **types**: Custom type definitions for trading operations

### Internal Dependencies

- **TIME_CONVERSIONS**: Time conversion constants
- **CoinbaseProductIds**: Supported trading pair identifiers
- **Granularity**: Time interval specifications

## Configuration

### Environment Variables

```bash
# Required for production
COINBASE_API_KEY=your_api_key_here
COINBASE_API_SECRET=your_api_secret_here

# Optional configurations
COINBASE_SANDBOX=true          # Use sandbox environment
COINBASE_TIMEOUT=30000         # API timeout in milliseconds
```

### Trading Parameters

- **Order Type**: Market orders with immediate-or-cancel (IOC)
- **Product ID**: ADA-USD trading pair
- **Granularity**: One-hour candlesticks for market data
- **Time Range**: Configurable historical data periods

## Testing

### Unit Testing Strategy

- **Mock API Responses**: Test with simulated Coinbase API responses
- **Error Scenarios**: Test various error conditions and edge cases
- **Order Validation**: Verify order parameter construction
- **Balance Calculations**: Test account balance retrieval logic

### Integration Testing

- **Sandbox Environment**: Use Coinbase sandbox for testing
- **Real API Calls**: Test with actual API endpoints
- **Order Simulation**: Test order execution without real trades
- **Performance Testing**: Measure API response times and throughput

### Test Data Requirements

- **Valid Trades**: Test with realistic trade parameters
- **Market Conditions**: Test various market scenarios
- **Error Conditions**: Test API failures and network issues
- **Edge Cases**: Test boundary conditions and limits

## Monitoring and Logging

### Performance Metrics

- **Trade Execution Time**: Measure order submission latency
- **API Response Time**: Monitor API endpoint performance
- **Success Rates**: Track successful vs. failed trades
- **Error Frequency**: Monitor error patterns and rates

### Logging Strategy

- **Trade Execution**: Log all trade attempts and results
- **Market Data**: Log price updates and data retrieval
- **Error Details**: Comprehensive error logging with context
- **Performance Data**: Log timing and performance metrics

## Future Enhancements

### Potential Improvements

- **Order Types**: Support for limit orders and stop-loss orders
- **Batch Trading**: Execute multiple trades simultaneously
- **Real-time Streaming**: WebSocket integration for live market data
- **Advanced Order Management**: Order modification and cancellation

### Advanced Features

- **Risk Management**: Position sizing and risk controls
- **Portfolio Rebalancing**: Automated portfolio adjustment
- **Multi-exchange Support**: Integration with additional exchanges
- **Algorithmic Trading**: Advanced trading strategy implementation

### Integration Enhancements

- **Database Logging**: Persistent trade and balance history
- **Notification System**: Real-time trade execution alerts
- **Analytics Dashboard**: Trading performance visualization
- **Backtesting Integration**: Historical strategy performance analysis
