import axios from "axios";
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

export const checkBitcoinPriceErrorMessage = (error: any) => {
  return `Error: ${JSON.stringify(error)}`;
};

export const priceAlertTextMessage = (
  threshold: number,
  currentPrice: number
) => {
  return `BITCOIN ALERT: BTC has risen above $${threshold} and is now at $${currentPrice}`;
};

export const formatAnalysisResults = ({
  cryptoSymbol,
  currentPrice,
  probabilities,
  recommendation,
}: {
  cryptoSymbol: string;
  currentPrice: number;
  probabilities: Probabilities;
  recommendation: Recommendation;
}) => {
  const symbol = cryptoSymbol.toUpperCase();
  const price = formatCurrency(currentPrice);
  const buyProb = `${(probabilities.buy * 100).toFixed(3)}%`;
  const sellProb = `${(probabilities.sell * 100).toFixed(3)}%`;
  const holdProb = `${(probabilities.hold * 100).toFixed(3)}%`;
  const rec = recommendation.charAt(0).toUpperCase() + recommendation.slice(1);

  return `${symbol}: ${price}\n\nBuy Probability: ${buyProb}\nSell Probability: ${sellProb}\nHold Probability: ${holdProb}\n\nRecommendation: ${rec}\n\nThe probability is the model’s confidence in a near-term price change exceeding 5% over 7 days.\n\nReply with a cryptocurrency to run the analysis again`;
};

export const fetchBitcoinPriceErrorMessage = (error: any) => {
  return `Error fetching Bitcoin price: ${JSON.stringify(error)}`;
};

export const sendSmsErrorMessage = (error: any) => {
  return `Error sending SMS notification: ${JSON.stringify(error)}`;
};

export const formatCurrency = (amount: number): string => {
  return new Intl.NumberFormat("en-US", {
    style: "currency",
    currency: "USD",
    minimumFractionDigits: 6,
  }).format(amount);
};

export const sendSMS = async (message: string) => {
  try {
    await axios.post("https://textbelt.com/text", {
      phone: process.env.PHONE_NUMBER,
      message,
      key: process.env.TEXTBELT_API_KEY,
      replyWebhookUrl: process.env.WEBHOOK_URL,
    });
  } catch (error) {
    console.log(sendSmsErrorMessage(error));
    throw error;
  }
};

// Add to utils.ts
export const formatPredictionExplanation = ({
  recommendation,
  buyProb,
  sellProb,
  confidence,
  momentum,
  trendSlope,
  atr,
}: {
  recommendation: Recommendation;
  buyProb: number;
  sellProb: number;
  confidence: number;
  momentum: number;
  trendSlope: number;
  atr: number;
}) => {
  const rec = recommendation.charAt(0).toUpperCase() + recommendation.slice(1);
  let explanation = `Reason for ${rec} Recommendation:\n`;

  if (recommendation === Recommendation.Buy) {
    explanation += `- High buy confidence (${(buyProb * 100).toFixed(
      1
    )}%) suggests upward potential.\n`;
    if (momentum > 0) {
      explanation += `- Positive momentum (${(momentum * 100).toFixed(
        2
      )}%) indicates rising prices.\n`;
    }
    if (trendSlope > 0) {
      explanation += `- Upward trend slope supports price growth.\n`;
    }
  } else if (recommendation === Recommendation.Sell) {
    explanation += `- High sell confidence (${(sellProb * 100).toFixed(
      1
    )}%) suggests downward risk.\n`;
    if (momentum < 0) {
      explanation += `- Negative momentum (${(momentum * 100).toFixed(
        2
      )}%) indicates falling prices.\n`;
    }
    if (trendSlope < 0) {
      explanation += `- Downward trend slope signals potential decline.\n`;
    }
  } else {
    explanation += `- Balanced probabilities (Buy: ${(buyProb * 100).toFixed(
      1
    )}%, Sell: ${(sellProb * 100).toFixed(1)}%) suggest no clear trend.\n`;
    explanation += `- Low volatility (ATR: ${atr.toFixed(
      4
    )}) or mixed signals favor holding.\n`;
  }

  explanation += `- Overall confidence: ${(confidence * 100).toFixed(1)}%.`;
  return explanation;
};
