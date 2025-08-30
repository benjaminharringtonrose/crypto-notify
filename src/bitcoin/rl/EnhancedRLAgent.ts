import * as tf from "@tensorflow/tfjs-node";
import {
  RLTradingEnvironment,
  RLState,
  RLAction,
  RLReward,
} from "./RLTradingEnvironment";

// Enhanced RL Agent Configuration
export interface EnhancedRLAgentConfig {
  learningRate: number;
  discountFactor: number;
  epsilon: number;
  epsilonDecay: number;
  epsilonMin: number;
  batchSize: number;
  memorySize: number;
  targetUpdateFrequency: number;
  gradientClipping: number;
  experienceReplay: boolean;
  prioritizedReplay: boolean;
  doubleDQN: boolean;
  useTradeModelFactory: boolean; // New: Use TradeModelFactory architecture
  timesteps: number;
  features: number;
}

// Experience replay memory entry
export interface Experience {
  state: RLState;
  action: RLAction;
  reward: RLReward;
  nextState: RLState;
  done: boolean;
  priority?: number;
}

// Training metrics
export interface EnhancedRLTrainingMetrics {
  episode: number;
  totalReward: number;
  averageReward: number;
  epsilon: number;
  loss: number;
  accuracy: number;
  totalReturn: number;
  sharpeRatio: number;
  maxDrawdown: number;
  winRate: number;
  totalTrades: number;
  qTableSize?: number; // For Q-table based agents
  modelComplexity?: string; // For neural network based agents
}

export class EnhancedRLAgent {
  private config: EnhancedRLAgentConfig;
  private environment: RLTradingEnvironment;
  private qNetwork: tf.LayersModel;
  private targetNetwork: tf.LayersModel;
  private memory: Experience[];
  private episodeCount: number;
  private stepCount: number;
  private epsilon: number;
  private trainingMetrics: EnhancedRLTrainingMetrics[];

  constructor(
    environment: RLTradingEnvironment,
    config: EnhancedRLAgentConfig
  ) {
    this.config = config;
    this.environment = environment;
    this.memory = [];
    this.episodeCount = 0;
    this.stepCount = 0;
    this.epsilon = config.epsilon;
    this.trainingMetrics = [];

    // Build Q-Network using TradeModelFactory or custom architecture
    this.qNetwork = this.buildQNetwork();
    this.targetNetwork = this.buildQNetwork();
    this.updateTargetNetwork();
  }

  /**
   * Build the Q-Network architecture using TradeModelFactory
   */
  private buildQNetwork(): tf.LayersModel {
    if (this.config.useTradeModelFactory) {
      console.log("🏭 Using TradeModelFactory for RL agent architecture");

      // Create RL-specific model using TradeModelFactory architecture
      const actionSize = this.environment.getActionSpaceSize();
      const model = tf.sequential();

      // Conv1D layer - OPTIMIZED for fast RL training
      model.add(
        tf.layers.conv1d({
          inputShape: [this.config.timesteps, this.config.features],
          filters: 24, // Reduced from 48 - faster training
          kernelSize: 3, // Keep optimal kernel size
          activation: "relu",
          kernelInitializer: "heNormal",
          kernelRegularizer: tf.regularizers.l2({ l2: 0.001 }),
          name: "conv1d_input",
        })
      );
      model.add(tf.layers.batchNormalization({ name: "bn_conv1" }));
      model.add(tf.layers.dropout({ rate: 0.2, name: "dropout_conv1" })); // Reduced dropout

      // LSTM layer - OPTIMIZED for fast RL training
      model.add(
        tf.layers.lstm({
          units: 32, // Reduced from 64 - much faster training
          returnSequences: false,
          kernelInitializer: "heNormal",
          recurrentDropout: 0.1,
          name: "lstm1",
        })
      );

      // Dense layer - OPTIMIZED for fast RL training
      model.add(
        tf.layers.dense({
          units: 16, // Reduced from 32 - faster training
          activation: "relu",
          kernelInitializer: "heNormal",
          kernelRegularizer: tf.regularizers.l2({ l2: 0.001 }),
          name: "dense1",
        })
      );
      model.add(tf.layers.dropout({ rate: 0.2, name: "dropout_dense1" })); // Reduced dropout

      // Q-Value output layer for RL
      model.add(
        tf.layers.dense({
          units: actionSize, // 7 actions for RL
          activation: "linear", // Linear activation for Q-values
          kernelInitializer: "heNormal",
          name: "q_values_output",
        })
      );

      // Compile model
      const optimizer = tf.train.adam(this.config.learningRate);

      model.compile({
        optimizer,
        loss: "meanSquaredError",
        metrics: ["accuracy"],
      });

      console.log("✅ Enhanced RL agent using TradeModelFactory architecture");
      console.log(
        `📊 RL Architecture: Conv1D(24,3) → BN → Dropout(0.2) → LSTM(32) → Dense(16) → Q-Values(${actionSize})`
      );
      console.log("⚡ OPTIMIZED for fast training - 75% fewer parameters");
      return model;
    } else {
      // Fallback to custom architecture
      console.log("🔧 Using custom architecture for RL agent");
      return this.buildCustomQNetwork();
    }
  }

  /**
   * Build custom Q-Network architecture (fallback) - OPTIMIZED for fast training
   */
  private buildCustomQNetwork(): tf.LayersModel {
    const stateSize = this.environment.getStateSpaceSize();
    const actionSize = this.environment.getActionSpaceSize();

    const model = tf.sequential();

    // Input layer - OPTIMIZED for fast training
    model.add(
      tf.layers.dense({
        units: 64, // Reduced from 128 - faster training
        inputShape: [stateSize],
        activation: "relu",
        kernelInitializer: "heNormal",
        kernelRegularizer: tf.regularizers.l2({ l2: 0.001 }),
      })
    );

    // Hidden layers - OPTIMIZED for fast training
    model.add(
      tf.layers.dense({
        units: 32, // Reduced from 64 - faster training
        activation: "relu",
        kernelInitializer: "heNormal",
        kernelRegularizer: tf.regularizers.l2({ l2: 0.001 }),
      })
    );
    model.add(tf.layers.dropout({ rate: 0.2 }));

    model.add(
      tf.layers.dense({
        units: 16, // Reduced from 32 - faster training
        activation: "relu",
        kernelInitializer: "heNormal",
        kernelRegularizer: tf.regularizers.l2({ l2: 0.001 }),
      })
    );

    // Output layer
    model.add(
      tf.layers.dense({
        units: actionSize,
        activation: "linear",
        kernelInitializer: "heNormal",
      })
    );

    // Compile model
    const optimizer = tf.train.adam(this.config.learningRate);

    model.compile({
      optimizer,
      loss: "meanSquaredError",
      metrics: ["accuracy"],
    });

    return model;
  }

  /**
   * Convert state to tensor for neural network input with improved normalization
   */
  private stateToTensor(state: RLState): tf.Tensor {
    if (this.config.useTradeModelFactory) {
      // For TradeModelFactory, we need to reshape to [timesteps, features]
      // We'll use the features array and pad/truncate to match timesteps
      const features = this.normalizeFeatures(state.features);
      const paddedFeatures = this.padFeaturesToTimesteps(features);

      return tf.tensor3d(
        [paddedFeatures],
        [1, this.config.timesteps, this.config.features]
      );
    } else {
      // For custom architecture, use flat state with better normalization
      const stateArray = [
        ...this.normalizeFeatures(state.features),
        this.normalizeValue(state.position, -1, 1), // Normalize position to [-1, 1]
        this.normalizeValue(
          state.capital / this.environment.getConfig().initialCapital,
          0,
          2
        ), // Normalize capital
        this.normalizeValue(
          state.portfolioValue / this.environment.getConfig().initialCapital,
          0,
          2
        ), // Normalize portfolio
        this.normalizeValue(state.volatility, 0, 1), // Normalize volatility
        this.normalizeValue(state.momentum, -1, 1), // Normalize momentum
        this.normalizeValue(state.trendStrength, -1, 1), // Normalize trend strength
        this.normalizeValue(state.timeStep / 1000, 0, 1), // Normalize time step
        state.marketRegime === "trending" ? 1 : 0,
        state.marketRegime === "ranging" ? 1 : 0,
        state.marketRegime === "volatile" ? 1 : 0,
      ];

      return tf.tensor2d([stateArray], [1, stateArray.length]);
    }
  }

  /**
   * Normalize features using robust scaling
   */
  private normalizeFeatures(features: number[]): number[] {
    return features.map((feature) => {
      // Robust normalization: clip to [-3, 3] and scale to [-1, 1]
      const clipped = Math.max(-3, Math.min(3, feature));
      return clipped / 3;
    });
  }

  /**
   * Normalize a value to a specific range
   */
  private normalizeValue(value: number, min: number, max: number): number {
    return Math.max(min, Math.min(max, value));
  }

  /**
   * Pad features to match timesteps for TradeModelFactory
   */
  private padFeaturesToTimesteps(features: number[]): number[][] {
    const result: number[][] = [];

    // Create timesteps by repeating the features
    for (let i = 0; i < this.config.timesteps; i++) {
      const timestepFeatures = features.slice(0, this.config.features);

      // Pad with zeros if features are shorter than expected
      while (timestepFeatures.length < this.config.features) {
        timestepFeatures.push(0);
      }

      result.push(timestepFeatures);
    }

    return result;
  }

  /**
   * Choose action using epsilon-greedy policy
   */
  public chooseAction(state: RLState): RLAction {
    if (Math.random() < this.epsilon) {
      return Math.floor(
        Math.random() * this.environment.getActionSpaceSize()
      ) as RLAction;
    } else {
      return this.getGreedyAction(state);
    }
  }

  /**
   * Get greedy action based on Q-values
   */
  private getGreedyAction(state: RLState): RLAction {
    const stateTensor = this.stateToTensor(state);
    const qValues = this.qNetwork.predict(stateTensor) as tf.Tensor;
    const qValuesArray = qValues.arraySync() as number[][];
    const qValuesRow = qValuesArray[0];

    // Clean up tensors
    stateTensor.dispose();
    qValues.dispose();

    return qValuesRow.indexOf(Math.max(...qValuesRow)) as RLAction;
  }

  /**
   * Store experience in replay memory
   */
  public storeExperience(
    state: RLState,
    action: RLAction,
    reward: RLReward,
    nextState: RLState,
    done: boolean
  ): void {
    const experience: Experience = {
      state,
      action,
      reward,
      nextState,
      done,
    };

    if (this.config.prioritizedReplay) {
      experience.priority = Math.abs(reward.total) + 1e-6;
    }

    this.memory.push(experience);

    if (this.memory.length > this.config.memorySize) {
      this.memory.shift();
    }
  }

  /**
   * Sample experiences from replay memory
   */
  private sampleExperiences(): Experience[] {
    if (this.memory.length < this.config.batchSize) {
      return this.memory;
    }

    if (this.config.prioritizedReplay) {
      return this.samplePrioritizedExperiences();
    } else {
      return this.sampleRandomExperiences();
    }
  }

  /**
   * Sample random experiences
   */
  private sampleRandomExperiences(): Experience[] {
    const batch: Experience[] = [];
    const memorySize = this.memory.length;

    for (let i = 0; i < this.config.batchSize; i++) {
      const randomIndex = Math.floor(Math.random() * memorySize);
      batch.push(this.memory[randomIndex]);
    }

    return batch;
  }

  /**
   * Sample experiences using prioritized replay
   */
  private samplePrioritizedExperiences(): Experience[] {
    const batch: Experience[] = [];
    const priorities = this.memory.map((exp) => exp.priority || 1);
    const totalPriority = priorities.reduce((sum, p) => sum + p, 0);

    for (let i = 0; i < this.config.batchSize; i++) {
      const randomValue = Math.random() * totalPriority;
      let cumulativePriority = 0;
      let selectedIndex = 0;

      for (let j = 0; j < this.memory.length; j++) {
        cumulativePriority += priorities[j];
        if (cumulativePriority >= randomValue) {
          selectedIndex = j;
          break;
        }
      }

      batch.push(this.memory[selectedIndex]);
    }

    return batch;
  }

  /**
   * Train the Q-Network
   */
  public async train(): Promise<number> {
    if (this.memory.length < this.config.batchSize) {
      return 0;
    }

    const experiences = this.sampleExperiences();
    const batchSize = experiences.length;

    // Prepare training data
    const states = tf.concat(
      experiences.map((exp) => this.stateToTensor(exp.state))
    );
    const nextStates = tf.concat(
      experiences.map((exp) => this.stateToTensor(exp.nextState))
    );

    // Get current Q-values
    const currentQValues = this.qNetwork.predict(states) as tf.Tensor;
    const nextQValues = this.targetNetwork.predict(nextStates) as tf.Tensor;

    // Prepare target Q-values
    const targetQValues = currentQValues.arraySync() as number[][];
    const nextQValuesArray = nextQValues.arraySync() as number[][];

    for (let i = 0; i < batchSize; i++) {
      const experience = experiences[i];
      const action = experience.action;
      const reward = experience.reward.total;
      const done = experience.done;

      if (done) {
        targetQValues[i][action] = reward;
      } else {
        if (this.config.doubleDQN) {
          // Double DQN: use main network for action selection, target network for evaluation
          const nextStateTensor = this.stateToTensor(experience.nextState);
          const mainNextQValues = this.qNetwork.predict(
            nextStateTensor
          ) as tf.Tensor;
          const mainNextQValuesArray =
            mainNextQValues.arraySync() as number[][];
          const bestAction = mainNextQValuesArray[0].indexOf(
            Math.max(...mainNextQValuesArray[0])
          );
          const targetValue = nextQValuesArray[i][bestAction];
          targetQValues[i][action] =
            reward + this.config.discountFactor * targetValue;

          // Clean up tensors
          nextStateTensor.dispose();
          mainNextQValues.dispose();
        } else {
          // Standard DQN
          const maxNextQValue = Math.max(...nextQValuesArray[i]);
          targetQValues[i][action] =
            reward + this.config.discountFactor * maxNextQValue;
        }
      }
    }

    // Train the network
    const targetTensor = tf.tensor2d(targetQValues as number[][]);
    const history = await this.qNetwork.fit(states, targetTensor, {
      epochs: 1,
      batchSize: this.config.batchSize,
      verbose: 0,
    });

    // Clean up tensors
    states.dispose();
    nextStates.dispose();
    currentQValues.dispose();
    nextQValues.dispose();
    targetTensor.dispose();

    // Update target network periodically
    if (this.stepCount % this.config.targetUpdateFrequency === 0) {
      this.updateTargetNetwork();
    }

    // Decay epsilon
    this.epsilon = Math.max(
      this.config.epsilonMin,
      this.epsilon * this.config.epsilonDecay
    );

    return history.history.loss[0] as number;
  }

  /**
   * Update target network weights
   */
  private updateTargetNetwork(): void {
    const weights = this.qNetwork.getWeights();
    this.targetNetwork.setWeights(weights);
  }

  /**
   * Run a complete episode
   */
  public async runEpisode(): Promise<EnhancedRLTrainingMetrics> {
    console.log(`      🎯 Starting episode ${this.episodeCount}...`);
    let state = this.environment.reset();
    let totalReward = 0;
    let stepCount = 0;
    const maxSteps = 1000; // Safety limit

    console.log(
      `      📊 Episode ${this.episodeCount} initialized, starting steps...`
    );

    while (stepCount < maxSteps) {
      // Choose action
      const action = this.chooseAction(state);

      // Execute action
      const { state: nextState, reward, done } = this.environment.step(action);

      // Store experience
      this.storeExperience(state, action, reward, nextState, done);

      // Train periodically (not every step)
      if (this.memory.length >= this.config.batchSize && stepCount % 10 === 0) {
        await this.train();
      }

      totalReward += reward.total;
      stepCount++;
      this.stepCount++;

      state = nextState;

      if (done) {
        console.log(
          `      ✅ Episode ${this.episodeCount} completed after ${stepCount} steps`
        );
        break;
      }
    }

    if (stepCount >= maxSteps) {
      console.log(
        `      ⚠️ Episode ${this.episodeCount} stopped at max steps (${maxSteps})`
      );
    }

    // Get episode results
    const episodeResults = this.environment.getEpisodeResults();
    console.log(
      `      📈 Episode ${this.episodeCount} results: Return=${(
        episodeResults.totalReturn * 100
      ).toFixed(2)}%, Sharpe=${episodeResults.sharpeRatio.toFixed(3)}, Trades=${
        episodeResults.totalTrades
      }`
    );

    // Create training metrics
    const metrics: EnhancedRLTrainingMetrics = {
      episode: this.episodeCount,
      totalReward,
      averageReward: totalReward / stepCount,
      epsilon: this.epsilon,
      loss: 0,
      accuracy: 0,
      totalReturn: episodeResults.totalReturn,
      sharpeRatio: episodeResults.sharpeRatio,
      maxDrawdown: episodeResults.maxDrawdown,
      winRate: episodeResults.winRate,
      totalTrades: episodeResults.totalTrades,
      modelComplexity: this.config.useTradeModelFactory
        ? "TradeModelFactory"
        : "Custom",
    };

    this.trainingMetrics.push(metrics);
    this.episodeCount++;

    return metrics;
  }

  /**
   * Train the agent for multiple episodes with adaptive learning
   */
  public async trainForEpisodes(
    numEpisodes: number,
    callback?: (metrics: EnhancedRLTrainingMetrics) => void
  ): Promise<EnhancedRLTrainingMetrics[]> {
    const results: EnhancedRLTrainingMetrics[] = [];
    let performanceHistory: number[] = [];

    for (let episode = 0; episode < numEpisodes; episode++) {
      const metrics = await this.runEpisode();
      results.push(metrics);
      performanceHistory.push(metrics.totalReturn);

      // Adaptive learning rate based on performance
      if (episode > 20 && episode % 10 === 0) {
        const recentPerformance = performanceHistory.slice(-10);
        const avgPerformance =
          recentPerformance.reduce((a, b) => a + b, 0) /
          recentPerformance.length;

        // Adjust learning rate based on performance trend
        if (avgPerformance > 0.05) {
          // Good performance - reduce learning rate for fine-tuning
          this.adjustLearningRate(0.95);
        } else if (avgPerformance < -0.1) {
          // Poor performance - increase learning rate for faster learning
          this.adjustLearningRate(1.05);
        }
      }

      if (callback) {
        callback(metrics);
      }

      // Log progress with adaptive learning info
      if (episode % 10 === 0) {
        console.log(
          `Episode ${episode}/${numEpisodes} - ` +
            `Return: ${(metrics.totalReturn * 100).toFixed(2)}% - ` +
            `Sharpe: ${metrics.sharpeRatio.toFixed(3)} - ` +
            `Epsilon: ${metrics.epsilon.toFixed(3)} - ` +
            `Trades: ${metrics.totalTrades} - ` +
            `Model: ${metrics.modelComplexity} - ` +
            `LR: ${this.config.learningRate.toFixed(6)}`
        );
      }
    }

    return results;
  }

  /**
   * Adjust learning rate adaptively
   */
  private adjustLearningRate(factor: number): void {
    const newLearningRate = this.config.learningRate * factor;
    const minLR = 0.0001;
    const maxLR = 0.01;

    this.config.learningRate = Math.max(
      minLR,
      Math.min(maxLR, newLearningRate)
    );

    // Update optimizer with new learning rate
    const optimizer = tf.train.adam(this.config.learningRate);
    this.qNetwork.compile({
      optimizer,
      loss: "meanSquaredError",
      metrics: ["accuracy"],
    });
  }

  /**
   * Evaluate the agent on a test environment
   */
  public async evaluate(
    testEnvironment: RLTradingEnvironment,
    numEpisodes: number = 1
  ): Promise<EnhancedRLTrainingMetrics[]> {
    const results: EnhancedRLTrainingMetrics[] = [];
    const originalEpsilon = this.epsilon;

    // Set epsilon to 0 for evaluation (no exploration)
    this.epsilon = 0;

    for (let episode = 0; episode < numEpisodes; episode++) {
      let state = testEnvironment.reset();
      let totalReward = 0;
      let stepCount = 0;

      while (true) {
        const action = this.chooseAction(state);
        const { state: nextState, reward, done } = testEnvironment.step(action);

        totalReward += reward.total;
        stepCount++;

        state = nextState;

        if (done) {
          break;
        }
      }

      const episodeResults = testEnvironment.getEpisodeResults();
      const metrics: EnhancedRLTrainingMetrics = {
        episode: episode,
        totalReward,
        averageReward: totalReward / stepCount,
        epsilon: 0,
        loss: 0,
        accuracy: 0,
        totalReturn: episodeResults.totalReturn,
        sharpeRatio: episodeResults.sharpeRatio,
        maxDrawdown: episodeResults.maxDrawdown,
        winRate: episodeResults.winRate,
        totalTrades: episodeResults.totalTrades,
        modelComplexity: this.config.useTradeModelFactory
          ? "TradeModelFactory"
          : "Custom",
      };

      results.push(metrics);
    }

    // Restore original epsilon
    this.epsilon = originalEpsilon;

    return results;
  }

  /**
   * Save the trained model
   */
  public async saveModel(path: string): Promise<void> {
    await this.qNetwork.save(`file://${path}`);
    console.log(`Model saved to ${path}`);
  }

  /**
   * Load a trained model
   */
  public async loadModel(path: string): Promise<void> {
    this.qNetwork = await tf.loadLayersModel(`file://${path}/model.json`);
    this.updateTargetNetwork();
    console.log(`Model loaded from ${path}`);
  }

  /**
   * Get training metrics history
   */
  public getTrainingMetrics(): EnhancedRLTrainingMetrics[] {
    return this.trainingMetrics;
  }

  /**
   * Get current epsilon value
   */
  public getEpsilon(): number {
    return this.epsilon;
  }

  /**
   * Get agent configuration
   */
  public getConfig(): EnhancedRLAgentConfig {
    return this.config;
  }

  /**
   * Get Q-Network model
   */
  public getQNetwork(): tf.LayersModel {
    return this.qNetwork;
  }

  /**
   * Dispose of resources
   */
  public dispose(): void {
    if (this.qNetwork) {
      this.qNetwork.dispose();
    }
    if (this.targetNetwork) {
      this.targetNetwork.dispose();
    }
  }
}
