import axios, { AxiosResponse } from "axios";
import { COINGECKO_API_URL } from "../constants";

export const getHistoricalData = async (
  coin: string,
  days: number
): Promise<{ prices: number[]; volumes: number[] }> => {
  const response: AxiosResponse = await axios.get(
    `${COINGECKO_API_URL}/coins/${coin}/market_chart?vs_currency=usd&days=${days}`
  );
  const prices = response.data.prices.map(
    ([_, price]: [number, number]) => price
  );
  const volumes = response.data.total_volumes.map(
    ([_, vol]: [number, number]) => vol
  );
  return { prices, volumes };
};
