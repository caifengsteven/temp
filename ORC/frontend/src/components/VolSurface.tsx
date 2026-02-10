import { useState } from 'react';
import Plot from 'react-plotly.js';
import { calibrateVolSurface, type VolSurfaceData } from '../api';

// Generate sample vol quotes for demo
function generateSampleQuotes(spot: number) {
  const today = new Date();
  const quotes: any[] = [];
  const expiries = [30, 60, 90, 120, 180].map(d => {
    const dt = new Date(today);
    dt.setDate(dt.getDate() + d);
    return dt.toISOString().split('T')[0];
  });
  const moneyness = [0.8, 0.85, 0.9, 0.95, 0.97, 1.0, 1.03, 1.05, 1.1, 1.15, 1.2];

  for (const exp of expiries) {
    for (const m of moneyness) {
      const strike = Math.round(spot * m * 100) / 100;
      const daysToExp = (new Date(exp).getTime() - today.getTime()) / 86400000;
      const baseVol = 0.20 + 0.05 * Math.pow(1.0 - m, 2) * 20;
      const termAdj = 0.02 * Math.sqrt(daysToExp / 365);
      const midVol = Math.round((baseVol + termAdj + (Math.random() - 0.5) * 0.01) * 10000) / 10000;
      const spread = 0.005 + Math.random() * 0.01;
      quotes.push({
        strike, expiry: exp,
        bid_vol: Math.round((midVol - spread / 2) * 10000) / 10000,
        ask_vol: Math.round((midVol + spread / 2) * 10000) / 10000,
        mid_vol: midVol,
        option_type: 'call',
      });
    }
  }
  return quotes;
}

export default function VolSurface() {
  const [surface, setSurface] = useState<VolSurfaceData | null>(null);
  const [model, setModel] = useState('sabr');
  const [symbol, setSymbol] = useState('SPY');
  const [spot, setSpot] = useState(502);
  const [loading, setLoading] = useState(false);
  const [errors, setErrors] = useState<number[]>([]);

  const calibrate = async () => {
    setLoading(true);
    try {
      const quotes = generateSampleQuotes(spot);
      const res = await calibrateVolSurface({
        symbol, spot, rate: 0.053, dividend_yield: 0.013,
        quotes, model, beta: 0.5,
      });
      setSurface(res.data);
      setErrors(res.data.fit_errors);
    } catch (e) { console.error(e); }
    setLoading(false);
  };

  return (
    <div style={{ height: '100%', display: 'flex', flexDirection: 'column' }}>
      <div className="panel" style={{ padding: 16, marginBottom: 16 }}>
        <div style={{ display: 'flex', gap: 16, alignItems: 'flex-end', flexWrap: 'wrap' }}>
          <div className="form-group" style={{ marginBottom: 0 }}>
            <label>Symbol</label>
            <input value={symbol} onChange={e => setSymbol(e.target.value)} style={{ width: 100 }} />
          </div>
          <div className="form-group" style={{ marginBottom: 0 }}>
            <label>Spot Price</label>
            <input type="number" value={spot} onChange={e => setSpot(+e.target.value)} style={{ width: 100 }} />
          </div>
          <div className="form-group" style={{ marginBottom: 0 }}>
            <label>Model</label>
            <select value={model} onChange={e => setModel(e.target.value)} style={{ width: 120 }}>
              <option value="sabr">SABR</option>
              <option value="svi">SVI</option>
            </select>
          </div>
          <button className="btn btn-primary" onClick={calibrate} disabled={loading}>
            {loading ? 'Calibrating...' : '⚡ Calibrate Surface'}
          </button>
          {errors.length > 0 && (
            <span style={{ color: 'var(--text-secondary)', fontSize: 12 }}>
              RMSE: {errors.map(e => (e * 100).toFixed(2) + '%').join(' | ')}
            </span>
          )}
        </div>
      </div>
      <div className="panel" style={{ flex: 1, minHeight: 500, padding: 8 }}>
        {surface ? (
          <Plot
            data={[{
              type: 'surface' as const,
              x: surface.strikes,
              y: surface.expiries,
              z: surface.vols.map(row => row.map(v => v * 100)),
              colorscale: 'Viridis' as const,
              colorbar: { title: { text: 'Vol %', side: 'right' as const } },
            }]}
            layout={{
              autosize: true,
              paper_bgcolor: '#161b22',
              plot_bgcolor: '#161b22',
              font: { color: '#e6edf3', size: 11 },
              scene: {
                xaxis: { title: { text: 'Strike' }, gridcolor: '#30363d' },
                yaxis: { title: { text: 'Expiry' }, gridcolor: '#30363d' },
                zaxis: { title: { text: 'Vol (%)' }, gridcolor: '#30363d' },
                bgcolor: '#0d1117',
              },
              margin: { l: 0, r: 0, t: 30, b: 0 },
              title: { text: `${symbol} Implied Volatility Surface (${model.toUpperCase()})`, font: { size: 14 } },
            }}
            useResizeHandler
            style={{ width: '100%', height: '100%' }}
          />
        ) : (
          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center',
                        height: '100%', color: 'var(--text-secondary)' }}>
            Click "Calibrate Surface" to fit vol model to market quotes
          </div>
        )}
      </div>
    </div>
  );
}

