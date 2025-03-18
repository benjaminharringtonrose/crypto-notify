import { Probabilities, Recommendation } from "./types";

export const isAboveThreshold = ({
  prices,
  currentPrice,
}: {
  prices: number[];
  currentPrice: number;
}) => {
  return prices.find((threshold) => currentPrice >= threshold);
};

export const checkCardanoPriceErrorMessage = (error: any) => {
  return `Error: ${JSON.stringify(error)}`;
};

export const priceAlertTextMessage = (
  threshold: number,
  currentPrice: number
) => {
  return `CARDANO ALERT: ADA has risen above $${threshold} and is now at $${currentPrice}`;
};

export const formatAnalysisResults = ({
  cryptoSymbol,
  currentPrice,
  probabilities,
  recommendation,
  metConditions,
}: {
  cryptoSymbol: string;
  currentPrice: number;
  probabilities: Probabilities;
  recommendation: Recommendation;
  metConditions: string[];
}) => {
  const symbol = cryptoSymbol.toUpperCase();
  const price = formatCurrency(currentPrice);
  const buyProb = `${(probabilities.buy * 100).toFixed(3)}%`;
  const sellProb = `${(probabilities.sell * 100).toFixed(3)}%`;
  const holdProb = `${(probabilities.hold * 100).toFixed(3)}%`;
  const rec = recommendation.charAt(0).toUpperCase() + recommendation.slice(1);
  const conditions = metConditions.join(", ");

  return `${symbol}: ${price}\n\nBuy Probability: ${buyProb}\nSell Probability: ${sellProb}\nHold Probability: ${holdProb}\n\nRecommendation: ${rec}\n\nSell conditions met: ${conditions}\n\nThe probability is the model’s confidence in a near-term price drop exceeding 2% over 3 days.\n\nReply with a cryptocurrency to run the analysis again`;
};

export const fetchCardanoPriceErrorMessage = (error: any) => {
  return `Error fetching Cardano price: ${JSON.stringify(error)}`;
};

export const sendSmsErrorMessage = (error: any) => {
  return `Error sending SMS notification: ${JSON.stringify(error)}`;
};

export const formatCurrency = (amount: number): string => {
  return new Intl.NumberFormat("en-US", {
    style: "currency",
    currency: "USD",
    minimumFractionDigits: 2,
  }).format(amount);
};
