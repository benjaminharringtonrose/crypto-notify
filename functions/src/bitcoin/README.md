# Bitcoin Trading System

This directory contains the Bitcoin trading system with three main components:

## Directory Structure

```
bitcoin/
├── shared/           # Shared components used by both ML and RL
│   ├── FeatureCalculator.ts
│   ├── FeatureRegistry.ts
│   ├── FeatureDetector.ts
│   ├── FeatureSequenceGenerator.ts
│   ├── Metrics.ts
│   └── callbacks/    # Training callbacks
├── ml/              # Machine Learning components
│   ├── TradeModelFactory.ts
│   ├── TradeModelTrainer.ts
│   ├── TradeModelPredictor.ts
│   ├── TradeModelBacktester.ts
│   ├── TradeModelWeightManager.ts
│   ├── DataProcessor.ts
│   ├── TradingStrategy.ts
│   └── TradeExecutor.ts
├── rl/              # Reinforcement Learning components
│   ├── RLTradingEnvironment.ts
│   ├── RLTradingAgent.ts
│   ├── EnhancedRLAgent.ts
│   ├── RLTradingStrategy.ts
│   ├── EnhancedRLConfig.ts
│   └── HighReturnRLConfig.ts
└── README.md
```

## Shared Components (`shared/`)

### FeatureCalculator

Computes technical indicators and market patterns.

**Indicators:**

- RSI, MACD, Bollinger Bands
- VWAP, ATR, Fibonacci levels
- Pattern detection (Double Top, Head and Shoulders)
- Volume analysis and momentum indicators

### FeatureRegistry

Manages feature registration and metadata.

### FeatureDetector

Dynamically detects available features and their counts.

### FeatureSequenceGenerator

Creates time-series sequences for LSTM/RNN models.

### Metrics

Tracks and calculates training and performance metrics.

### Callbacks

Training callbacks for monitoring and optimization:

- EarlyStoppingCallback
- PredictionLoggerCallback
- ExponentialDecayLRCallback
- GradientClippingCallback
- TrainingLoggerCallback
- CurriculumLearningCallback

## Machine Learning Components (`ml/`)

### TradeModelFactory

Builds and configures the neural network architecture.

**Architecture:**

- CNN layers for feature extraction
- LSTM layers for temporal modeling
- Dense layers for final predictions

### TradeModelTrainer

Handles the training process with validation and early stopping.

### TradeModelPredictor

Makes predictions using trained models.

### TradeModelBacktester

Simulates historical trading performance.

### TradeModelWeightManager

Manages model weight persistence and loading.

### DataProcessor

Processes and prepares training data.

### TradingStrategy

Implements rule-based trading logic.

### TradeExecutor

Executes trades through the Coinbase API.

## Reinforcement Learning Components (`rl/`)

### RLTradingEnvironment

Defines the trading environment for RL agents.

### RLTradingAgent

Basic RL agent implementation.

### EnhancedRLAgent

Advanced RL agent with multiple algorithms.

### RLTradingStrategy

RL-based trading strategy implementation.

### EnhancedRLConfig

Configuration for enhanced RL training.

### HighReturnRLConfig

Configuration optimized for high returns.

## Training the Models

### ML Training Process

1. **Data Preparation**: Load historical BTC price and volume data
2. **Feature Engineering**: Compute technical indicators and patterns
3. **Sequence Generation**: Create time-series sequences for training
4. **Model Training**: Train the neural network with validation
5. **Weight Persistence**: Save trained weights to Firebase Storage

### RL Training Process

1. **Environment Setup**: Configure trading environment
2. **Agent Training**: Train RL agent through episodes
3. **Strategy Optimization**: Optimize trading strategies
4. **Performance Evaluation**: Evaluate agent performance

## Usage Examples

### ML Model Training

```typescript
import { TradeModelTrainer } from "./bitcoin/ml/TradeModelTrainer";

const trainer = new TradeModelTrainer(seed);
await trainer.train();
```

### RL Agent Training

```typescript
import { RLTradingEnvironment } from "./bitcoin/rl/RLTradingEnvironment";
import { EnhancedRLAgent } from "./bitcoin/rl/EnhancedRLAgent";

const environment = new RLTradingEnvironment(prices, volumes, config);
const agent = new EnhancedRLAgent(environment, agentConfig);
await agent.trainForEpisodes(100);
```

### Feature Calculation

```typescript
import FeatureCalculator from "./bitcoin/shared/FeatureCalculator";

const calculator = new FeatureCalculator();
const features = calculator.calculateFeatures(prices, volumes);
```

## Performance Considerations

### ML Training

- **Training Time**: ~73 seconds for 35 epochs
- **Model Size**: ~35K parameters
- **Memory Usage**: ~150MB during training

### RL Training

- **Simple RL**: 2-3 minutes
- **Enhanced RL**: 15-30 minutes
- **Production RL**: 30-60 minutes

## Configuration

### Model Architecture

- **Timesteps**: 35 days of historical data
- **Features**: 42 technical indicators
- **Architecture**: Conv1D(48,3) → LSTM(64) → Dense(32) → Output(2)

### Training Parameters

- **Epochs**: 35 (early stopping)
- **Batch Size**: 16
- **Learning Rate**: 0.0005
- **Loss Function**: Focal Loss

### RL Parameters

- **Episodes**: 100-150
- **Memory Size**: 15000
- **Learning Rate**: 0.002
- **Epsilon**: 0.4 (decay to 0.05)
