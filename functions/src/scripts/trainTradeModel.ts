import { TradeModelTrainer } from "../cardano/TradeModelTrainer";

const trainer = new TradeModelTrainer();
trainer.train().catch(console.error);
