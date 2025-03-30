import dotenv from "dotenv";
import { onSchedule } from "firebase-functions/scheduler";
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
    await trainer.train();
  }
);
