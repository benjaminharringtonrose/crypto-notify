import axios from "axios";
import { CoinGeckoMarketChartResponse } from "../types";
import {
  BITCOIN_30_DAY_HISTORICAL_URL,
  CARDANO_30_DAY_HISTORICAL_URL,
} from "../constants";

export const getHistoricalPricesAndVolumes = async () => {
  const historicalAdaResponse = await axios.get<CoinGeckoMarketChartResponse>(
    CARDANO_30_DAY_HISTORICAL_URL
  );

  const historicalBtcResponse = await axios.get<CoinGeckoMarketChartResponse>(
    BITCOIN_30_DAY_HISTORICAL_URL
  );

  const adaPrices = historicalAdaResponse.data.prices.map(
    ([_, price]) => price
  );
  const adaVolumes = historicalAdaResponse.data.total_volumes.map(
    ([_, vol]) => vol
  );
  const btcPrices = historicalBtcResponse.data.prices.map(
    ([_, price]) => price
  );
  const btcVolumes = historicalBtcResponse.data.total_volumes.map(
    ([_, vol]) => vol
  );

  return {
    adaPrices,
    adaVolumes,
    btcPrices,
    btcVolumes,
  };
};
