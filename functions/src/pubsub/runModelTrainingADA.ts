import dotenv from "dotenv";
import { onSchedule } from "firebase-functions/v2/scheduler";
import { TradeModelTrainer } from "../cardano/TradeModelTrainer";

dotenv.config();

export const runModelTrainingADA = onSchedule(
  {
    schedule: `*/30 * * * *`,
    memory: "4GiB",
    timeoutSeconds: 540,
  },
  async () => {
    console.log("Training started...");
    const trainer = new TradeModelTrainer();
    try {
      await trainer.train();
      console.log("Training completed.");
    } catch (error) {
      console.error("Training failed:", error);
      throw error;
    }
  }
);
