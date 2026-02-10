import axios from 'axios';

const api = axios.create({ baseURL: '/api' });

export interface Position {
  position_id: string;
  symbol: string;
  option_type: string;
  strike: number;
  expiry: string;
  quantity: number;
  avg_price: number;
  multiplier: number;
}

export interface PositionGreeks {
  position: Position;
  theo_price: number;
  iv: number;
  delta: number;
  gamma: number;
  vega: number;
  theta: number;
  rho: number;
  vanna: number;
  volga: number;
  charm: number;
  delta_usd: number;
  gamma_usd: number;
  vega_usd: number;
  theta_usd: number;
  rho_usd: number;
  vanna_usd: number;
  volga_usd: number;
  charm_usd: number;
  market_value: number;
  unrealized_pnl: number;
}

export interface PortfolioSummary {
  total_delta_usd: number;
  total_gamma_usd: number;
  total_vega_usd: number;
  total_theta_usd: number;
  total_rho_usd: number;
  total_vanna_usd: number;
  total_volga_usd: number;
  total_charm_usd: number;
  total_market_value: number;
  total_unrealized_pnl: number;
  position_count: number;
}

export interface PortfolioResponse {
  positions: PositionGreeks[];
  summary: PortfolioSummary;
  by_underlying: Record<string, Record<string, number>>;
  by_expiry: Record<string, Record<string, number>>;
}

export interface Order {
  order_id: string;
  cl_ord_id: string;
  symbol: string;
  instrument_type: string;
  option_type: string | null;
  strike: number | null;
  expiry: string | null;
  side: string;
  quantity: number;
  filled_quantity: number;
  remaining_quantity: number;
  order_type: string;
  limit_price: number | null;
  avg_fill_price: number;
  status: string;
  reject_reason: string | null;
  created_at: string;
  updated_at: string;
}

export interface VolSurfaceData {
  strikes: number[];
  expiries: string[];
  vols: number[][];
  model_type: string;
  params: Record<string, unknown>[];
  fit_errors: number[];
}

export const fetchPortfolio = () => api.get<PortfolioResponse>('/portfolio');

export const submitOrder = (order: {
  symbol: string;
  instrument_type?: string;
  option_type?: string;
  strike?: number;
  expiry?: string;
  side: string;
  quantity: number;
  order_type: string;
  limit_price?: number;
}) => api.post<Order>('/orders/submit', order);

export const cancelOrder = (orderId: string) =>
  api.post('/orders/cancel', { order_id: orderId });

export const fetchOrders = () => api.get<Order[]>('/orders');

export const calibrateVolSurface = (req: {
  symbol: string;
  spot: number;
  rate?: number;
  dividend_yield?: number;
  quotes: {
    strike: number;
    expiry: string;
    bid_vol?: number;
    ask_vol?: number;
    mid_vol?: number;
    option_type?: string;
  }[];
  model?: string;
  beta?: number;
}) => api.post<VolSurfaceData>('/volsurface/calibrate', req);

export const priceOption = (req: {
  contract: {
    symbol: string;
    option_type: string;
    strike: number;
    expiry: string;
    exercise_style?: string;
  };
  market: {
    spot: number;
    rate?: number;
    dividend_yield?: number;
    volatility: number;
  };
}) => api.post('/pricing/price', req);

