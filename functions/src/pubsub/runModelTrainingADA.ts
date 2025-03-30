import { onSchedule } from "firebase-functions/scheduler";
import { TradeModelTrainer } from "../cardano/TradeModelTrainer";
import { TRAINING_CONFIG } from "../constants";

export const runModelTrainingADA = onSchedule(TRAINING_CONFIG, async () => {
  console.log("Training started...");
  const trainer = new TradeModelTrainer();
  await trainer.train();
});
