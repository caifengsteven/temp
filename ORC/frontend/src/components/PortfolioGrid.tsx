import { useEffect, useState } from 'react';
import { AgGridReact } from 'ag-grid-react';
import { AllCommunityModule, ModuleRegistry, type ColDef } from 'ag-grid-community';
import { fetchPortfolio, type PortfolioResponse, type PositionGreeks } from '../api';

ModuleRegistry.registerModules([AllCommunityModule]);

const usdFmt = (v: number | null | undefined) =>
  v != null ? v.toLocaleString('en-US', { style: 'currency', currency: 'USD', minimumFractionDigits: 0 }) : '';

const numFmt = (decimals: number) => (p: { value: number | null }) =>
  p.value != null ? p.value.toFixed(decimals) : '';

const colorCell = (p: { value: number }) => ({
  color: p.value > 0 ? 'var(--green)' : p.value < 0 ? 'var(--red)' : 'var(--text-secondary)',
});

const columns: ColDef<PositionGreeks>[] = [
  { headerName: 'Symbol', field: 'position.symbol' as any, width: 80, pinned: 'left' },
  { headerName: 'Type', field: 'position.option_type' as any, width: 60 },
  { headerName: 'Strike', field: 'position.strike' as any, width: 80, valueFormatter: numFmt(1) as any },
  { headerName: 'Expiry', field: 'position.expiry' as any, width: 110 },
  { headerName: 'Qty', field: 'position.quantity' as any, width: 60, cellStyle: colorCell as any },
  { headerName: 'Theo', field: 'theo_price', width: 80, valueFormatter: numFmt(4) as any },
  { headerName: 'IV', field: 'iv', width: 70, valueFormatter: (p: any) => p.value ? (p.value * 100).toFixed(1) + '%' : '' },
  { headerName: 'Delta', field: 'delta', width: 75, valueFormatter: numFmt(4) as any, cellStyle: colorCell as any },
  { headerName: 'Gamma', field: 'gamma', width: 75, valueFormatter: numFmt(6) as any },
  { headerName: 'Vega', field: 'vega', width: 75, valueFormatter: numFmt(4) as any },
  { headerName: 'Theta', field: 'theta', width: 75, valueFormatter: numFmt(4) as any, cellStyle: colorCell as any },
  { headerName: 'Rho', field: 'rho', width: 70, valueFormatter: numFmt(4) as any },
  { headerName: 'Vanna', field: 'vanna', width: 75, valueFormatter: numFmt(4) as any },
  { headerName: 'Volga', field: 'volga', width: 75, valueFormatter: numFmt(6) as any },
  { headerName: 'Δ USD', field: 'delta_usd', width: 100, valueFormatter: (p: any) => usdFmt(p.value), cellStyle: colorCell as any },
  { headerName: 'Γ USD', field: 'gamma_usd', width: 100, valueFormatter: (p: any) => usdFmt(p.value) },
  { headerName: 'V USD', field: 'vega_usd', width: 100, valueFormatter: (p: any) => usdFmt(p.value) },
  { headerName: 'Θ USD', field: 'theta_usd', width: 100, valueFormatter: (p: any) => usdFmt(p.value), cellStyle: colorCell as any },
  { headerName: 'ρ USD', field: 'rho_usd', width: 90, valueFormatter: (p: any) => usdFmt(p.value) },
  { headerName: 'MktVal', field: 'market_value', width: 110, valueFormatter: (p: any) => usdFmt(p.value) },
  { headerName: 'P&L', field: 'unrealized_pnl', width: 110, valueFormatter: (p: any) => usdFmt(p.value), cellStyle: colorCell as any },
];

export default function PortfolioGrid() {
  const [data, setData] = useState<PortfolioResponse | null>(null);
  const [loading, setLoading] = useState(true);

  const load = async () => {
    setLoading(true);
    try {
      const res = await fetchPortfolio();
      setData(res.data);
    } catch (e) { console.error(e); }
    setLoading(false);
  };

  useEffect(() => { load(); }, []);

  const s = data?.summary;
  return (
    <div style={{ height: '100%', display: 'flex', flexDirection: 'column' }}>
      {s && (
        <div className="summary-bar">
          <SummaryCard label="Positions" value={s.position_count.toString()} />
          <SummaryCard label="Delta USD" value={usdFmt(s.total_delta_usd)} color={s.total_delta_usd} />
          <SummaryCard label="Gamma USD" value={usdFmt(s.total_gamma_usd)} color={s.total_gamma_usd} />
          <SummaryCard label="Vega USD" value={usdFmt(s.total_vega_usd)} color={s.total_vega_usd} />
          <SummaryCard label="Theta USD" value={usdFmt(s.total_theta_usd)} color={s.total_theta_usd} />
          <SummaryCard label="Mkt Value" value={usdFmt(s.total_market_value)} color={s.total_market_value} />
          <SummaryCard label="Unreal P&L" value={usdFmt(s.total_unrealized_pnl)} color={s.total_unrealized_pnl} />
          <button className="btn btn-primary" onClick={load} style={{ alignSelf: 'center' }}>
            ↻ Refresh
          </button>
        </div>
      )}
      <div className="ag-theme-alpine-dark" style={{ flex: 1, minHeight: 400 }}>
        <AgGridReact<PositionGreeks>
          rowData={data?.positions ?? []}
          columnDefs={columns}
          defaultColDef={{ sortable: true, resizable: true, filter: true }}
          animateRows
          loading={loading}
        />
      </div>
    </div>
  );
}

function SummaryCard({ label, value, color }: { label: string; value: string; color?: number }) {
  const cls = color != null ? (color > 0 ? 'positive' : color < 0 ? 'negative' : '') : '';
  return (
    <div className="summary-card">
      <div className="label">{label}</div>
      <div className={`value ${cls}`}>{value}</div>
    </div>
  );
}

