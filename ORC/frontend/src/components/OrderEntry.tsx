import { useState } from 'react';
import { submitOrder } from '../api';

export default function OrderEntry() {
  const [symbol, setSymbol] = useState('AAPL');
  const [optType, setOptType] = useState('call');
  const [strike, setStrike] = useState(185);
  const [expiry, setExpiry] = useState(() => {
    const d = new Date(); d.setDate(d.getDate() + 30);
    return d.toISOString().split('T')[0];
  });
  const [qty, setQty] = useState(1);
  const [orderType, setOrderType] = useState('limit');
  const [limitPrice, setLimitPrice] = useState(5.00);
  const [lastResult, setLastResult] = useState<string | null>(null);

  const send = async (side: 'buy' | 'sell') => {
    try {
      const res = await submitOrder({
        symbol, instrument_type: 'option', option_type: optType,
        strike, expiry, side, quantity: qty,
        order_type: orderType,
        limit_price: orderType === 'limit' ? limitPrice : undefined,
      });
      const o = res.data;
      setLastResult(`${o.status.toUpperCase()} | ${o.order_id} | ${side.toUpperCase()} ${qty}x ${symbol} ${optType} ${strike} @ ${o.avg_fill_price || limitPrice}`);
    } catch (e: any) {
      setLastResult(`ERROR: ${e.response?.data?.detail || e.message}`);
    }
  };

  return (
    <div className="panel" style={{ height: 'fit-content' }}>
      <div className="panel-header">Order Entry</div>
      <div style={{ padding: 16 }}>
        <div className="form-row">
          <div className="form-group">
            <label>Symbol</label>
            <input value={symbol} onChange={e => setSymbol(e.target.value.toUpperCase())} />
          </div>
          <div className="form-group">
            <label>Option Type</label>
            <select value={optType} onChange={e => setOptType(e.target.value)}>
              <option value="call">Call</option>
              <option value="put">Put</option>
            </select>
          </div>
        </div>
        <div className="form-row">
          <div className="form-group">
            <label>Strike</label>
            <input type="number" value={strike} onChange={e => setStrike(+e.target.value)} step="0.5" />
          </div>
          <div className="form-group">
            <label>Expiry</label>
            <input type="date" value={expiry} onChange={e => setExpiry(e.target.value)} />
          </div>
        </div>
        <div className="form-row">
          <div className="form-group">
            <label>Quantity</label>
            <input type="number" value={qty} onChange={e => setQty(+e.target.value)} min={1} />
          </div>
          <div className="form-group">
            <label>Order Type</label>
            <select value={orderType} onChange={e => setOrderType(e.target.value)}>
              <option value="limit">Limit</option>
              <option value="market">Market</option>
              <option value="ioc">IOC</option>
            </select>
          </div>
        </div>
        {orderType === 'limit' && (
          <div className="form-group">
            <label>Limit Price</label>
            <input type="number" value={limitPrice} onChange={e => setLimitPrice(+e.target.value)} step="0.01" />
          </div>
        )}
        <div style={{ display: 'flex', gap: 12, marginTop: 16 }}>
          <button className="btn btn-buy" style={{ flex: 1 }} onClick={() => send('buy')}>
            BUY
          </button>
          <button className="btn btn-sell" style={{ flex: 1 }} onClick={() => send('sell')}>
            SELL
          </button>
        </div>
        {lastResult && (
          <div style={{
            marginTop: 12, padding: 10, background: 'var(--bg-tertiary)',
            borderRadius: 4, fontSize: 12, fontFamily: 'var(--font-mono)',
            color: lastResult.startsWith('ERROR') ? 'var(--red)' : 'var(--green)',
          }}>
            {lastResult}
          </div>
        )}
      </div>
    </div>
  );
}

