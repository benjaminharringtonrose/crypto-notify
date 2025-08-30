import * as tf from "@tensorflow/tfjs-node";
import {
  RLTradingEnvironment,
  RLState,
  RLAction,
  RLReward,
} from "./RLTradingEnvironment";

// RL Agent Configuration
export interface RLAgentConfig {
  learningRate: number;
  discountFactor: number;
  epsilon: number;
  epsilonDecay: number;
  epsilonMin: number;
  batchSize: number;
  memorySize: number;
  targetUpdateFrequency: number;
  hiddenLayers: number[];
  activationFunction: string;
  optimizer: string;
  lossFunction: string;
  gradientClipping: number;
  experienceReplay: boolean;
  prioritizedReplay: boolean;
  doubleDQN: boolean;
  duelingDQN: boolean;
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
export interface RLTrainingMetrics {
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
}

export class RLTradingAgent {
  private config: RLAgentConfig;
  private environment: RLTradingEnvironment;
  private qNetwork: tf.LayersModel;
  private targetNetwork: tf.LayersModel;
  private memory: Experience[];
  private episodeCount: number;
  private stepCount: number;
  private epsilon: number;
  private trainingMetrics: RLTrainingMetrics[];

  constructor(environment: RLTradingEnvironment, config: RLAgentConfig) {
    this.config = config;
    this.environment = environment;
    this.memory = [];
    this.episodeCount = 0;
    this.stepCount = 0;
    this.epsilon = config.epsilon;
    this.trainingMetrics = [];

    // Build Q-Network
    this.qNetwork = this.buildQNetwork();
    this.targetNetwork = this.buildQNetwork();
    this.updateTargetNetwork();
  }

  /**
   * Build the Q-Network architecture
   */
  private buildQNetwork(): tf.LayersModel {
    const stateSize = this.environment.getStateSpaceSize();
    const actionSize = this.environment.getActionSpaceSize();

    const model = tf.sequential();

    // Input layer
    model.add(
      tf.layers.dense({
        units: this.config.hiddenLayers[0],
        inputShape: [stateSize],
        activation: this.config.activationFunction as any,
        kernelInitializer: "heNormal",
        kernelRegularizer: tf.regularizers.l2({ l2: 0.001 }),
      })
    );

    // Hidden layers
    for (let i = 1; i < this.config.hiddenLayers.length; i++) {
      model.add(
        tf.layers.dense({
          units: this.config.hiddenLayers[i],
          activation: this.config.activationFunction as any,
          kernelInitializer: "heNormal",
          kernelRegularizer: tf.regularizers.l2({ l2: 0.001 }),
        })
      );
      model.add(tf.layers.dropout({ rate: 0.2 }));
    }

    // Output layer
    model.add(
      tf.layers.dense({
        units: actionSize,
        activation: "linear",
        kernelInitializer: "heNormal",
      })
    );

    // Compile model
    const optimizer =
      this.config.optimizer === "adam"
        ? tf.train.adam(this.config.learningRate)
        : tf.train.rmsprop(this.config.learningRate);

    model.compile({
      optimizer,
      loss: this.config.lossFunction,
      metrics: ["accuracy"],
    });

    return model;
  }

  /**
   * Convert state to tensor for neural network input
   */
  private stateToTensor(state: RLState): tf.Tensor {
    const stateArray = [
      ...state.features,
      state.position,
      state.capital / this.environment.getConfig().initialCapital, // Normalize capital
      state.portfolioValue / this.environment.getConfig().initialCapital, // Normalize portfolio
      state.volatility,
      state.momentum,
      state.trendStrength,
      state.timeStep / 1000, // Normalize time step
      // Enhanced market regime encoding
      state.marketRegime === "trending" ? 1 : 0,
      state.marketRegime === "ranging" ? 1 : 0,
      state.marketRegime === "volatile" ? 1 : 0,
      state.marketRegime === "bullish" ? 1 : 0,
      state.marketRegime === "bearish" ? 1 : 0,
      // New state variables
      state.regimeConfidence,
      state.riskScore,
      state.optimalPositionSize,
      // Market condition encoding
      state.marketCondition === "favorable" ? 1 : 0,
      state.marketCondition === "neutral" ? 1 : 0,
      state.marketCondition === "unfavorable" ? 1 : 0,
      // Synthetic data flag
      state.syntheticDataFlag ? 1 : 0,
      // Rule-based and hybrid signals
      state.ruleBasedSignal,
      state.hybridWeight,
    ];

    return tf.tensor2d([stateArray], [1, stateArray.length]);
  }

  /**
   * Choose action using epsilon-greedy policy
   */
  public chooseAction(state: RLState): RLAction {
    if (Math.random() < this.epsilon) {
      // Random action
      return Math.floor(
        Math.random() * this.environment.getActionSpaceSize()
      ) as RLAction;
    } else {
      // Greedy action
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

    // Return action with highest Q-value
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
      // Calculate priority based on reward magnitude
      experience.priority = Math.abs(reward.total) + 1e-6;
    }

    this.memory.push(experience);

    // Remove oldest experience if memory is full
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
  public async runEpisode(): Promise<RLTrainingMetrics> {
    let state = this.environment.reset();
    let totalReward = 0;
    let stepCount = 0;

    while (true) {
      // Choose action
      const action = this.chooseAction(state);

      // Execute action
      const { state: nextState, reward, done } = this.environment.step(action);

      // Store experience
      this.storeExperience(state, action, reward, nextState, done);

      // Train if enough experiences
      await this.train();

      totalReward += reward.total;
      stepCount++;
      this.stepCount++;

      state = nextState;

      if (done) {
        break;
      }
    }

    // Get episode results
    const episodeResults = this.environment.getEpisodeResults();

    // Create training metrics
    const metrics: RLTrainingMetrics = {
      episode: this.episodeCount,
      totalReward,
      averageReward: totalReward / stepCount,
      epsilon: this.epsilon,
      loss: 0, // Will be updated with actual loss
      accuracy: 0, // Will be updated with actual accuracy
      totalReturn: episodeResults.totalReturn,
      sharpeRatio: episodeResults.sharpeRatio,
      maxDrawdown: episodeResults.maxDrawdown,
      winRate: episodeResults.winRate,
      totalTrades: episodeResults.totalTrades,
    };

    this.trainingMetrics.push(metrics);
    this.episodeCount++;

    return metrics;
  }

  /**
   * Train the agent for multiple episodes
   */
  public async trainForEpisodes(
    numEpisodes: number,
    callback?: (metrics: RLTrainingMetrics) => void
  ): Promise<RLTrainingMetrics[]> {
    const results: RLTrainingMetrics[] = [];

    for (let episode = 0; episode < numEpisodes; episode++) {
      const metrics = await this.runEpisode();
      results.push(metrics);

      if (callback) {
        callback(metrics);
      }

      // Log progress
      if (episode % 10 === 0) {
        console.log(
          `Episode ${episode}/${numEpisodes} - ` +
            `Return: ${(metrics.totalReturn * 100).toFixed(2)}% - ` +
            `Sharpe: ${metrics.sharpeRatio.toFixed(3)} - ` +
            `Epsilon: ${metrics.epsilon.toFixed(3)} - ` +
            `Trades: ${metrics.totalTrades}`
        );
      }
    }

    return results;
  }

  /**
   * Evaluate the agent on a test environment
   */
  public async evaluate(
    testEnvironment: RLTradingEnvironment,
    numEpisodes: number = 1
  ): Promise<RLTrainingMetrics[]> {
    const results: RLTrainingMetrics[] = [];
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
      const metrics: RLTrainingMetrics = {
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
  public getTrainingMetrics(): RLTrainingMetrics[] {
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
  public getConfig(): RLAgentConfig {
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
