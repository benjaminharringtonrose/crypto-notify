import * as tf from "@tensorflow/tfjs-node";

export class CurriculumLearningCallback extends tf.Callback {
  private currentEpoch: number = 0;
  private totalEpochs: number;
  private difficultySchedule: number[];
  private dataProcessor: any;

  constructor(totalEpochs: number, dataProcessor: any) {
    super();
    this.totalEpochs = totalEpochs;
    this.dataProcessor = dataProcessor;

    // Define difficulty schedule: start with easier samples, gradually increase difficulty
    this.difficultySchedule = this.generateDifficultySchedule();
  }

  private generateDifficultySchedule(): number[] {
    const schedule: number[] = [];

    // Phase 1 (epochs 0-30): Start with easiest 60% of data
    for (let i = 0; i < 30; i++) {
      schedule.push(0.6);
    }

    // Phase 2 (epochs 31-60): Gradually increase to 80% of data
    for (let i = 30; i < 60; i++) {
      const progress = (i - 30) / 30;
      schedule.push(0.6 + progress * 0.2);
    }

    // Phase 3 (epochs 61-90): Increase to 90% of data
    for (let i = 60; i < 90; i++) {
      const progress = (i - 60) / 30;
      schedule.push(0.8 + progress * 0.1);
    }

    // Phase 4 (epochs 91+): Use full dataset
    for (let i = 90; i < this.totalEpochs; i++) {
      schedule.push(1.0);
    }

    return schedule;
  }

  async onEpochBegin(epoch: number, logs?: tf.Logs): Promise<void> {
    this.currentEpoch = epoch;
    const difficulty =
      this.difficultySchedule[
        Math.min(epoch, this.difficultySchedule.length - 1)
      ];

    // Update data processor with current difficulty level
    if (this.dataProcessor && this.dataProcessor.setDifficultyLevel) {
      this.dataProcessor.setDifficultyLevel(difficulty);
    }

    console.log(
      `Curriculum Learning: Epoch ${epoch + 1}, Difficulty: ${(
        difficulty * 100
      ).toFixed(1)}%`
    );
  }

  async onEpochEnd(epoch: number, logs?: tf.Logs): Promise<void> {
    // Log curriculum progress
    const difficulty =
      this.difficultySchedule[
        Math.min(epoch, this.difficultySchedule.length - 1)
      ];
    console.log(
      `Curriculum Learning: Completed epoch ${epoch + 1}, Final difficulty: ${(
        difficulty * 100
      ).toFixed(1)}%`
    );
  }

  getDifficultyLevel(): number {
    return this.difficultySchedule[
      Math.min(this.currentEpoch, this.difficultySchedule.length - 1)
    ];
  }
}
