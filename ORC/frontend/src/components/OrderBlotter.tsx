import { useEffect, useState } from 'react';
import { AgGridReact } from 'ag-grid-react';
import { AllCommunityModule, ModuleRegistry, type ColDef } from 'ag-grid-community';
import { fetchOrders, cancelOrder, type Order } from '../api';

ModuleRegistry.registerModules([AllCommunityModule]);

const statusColor = (status: string) => {
  switch (status) {
    case 'filled': return 'var(--green)';
    case 'partial': return 'var(--yellow)';
    case 'cancelled':
    case 'rejected': return 'var(--red)';
    default: return 'var(--text-primary)';
  }
};

const columns: ColDef<Order>[] = [
  { headerName: 'ID', field: 'order_id', width: 80 },
  { headerName: 'Symbol', field: 'symbol', width: 75 },
  { headerName: 'Side', field: 'side', width: 60,
    cellStyle: (p: any) => ({ color: p.value === 'buy' ? 'var(--green)' : 'var(--red)', fontWeight: 600 }) },
  { headerName: 'Type', field: 'option_type', width: 55 },
  { headerName: 'Strike', field: 'strike', width: 70 },
  { headerName: 'Expiry', field: 'expiry', width: 100 },
  { headerName: 'Qty', field: 'quantity', width: 55 },
  { headerName: 'Filled', field: 'filled_quantity', width: 60 },
  { headerName: 'OrdType', field: 'order_type', width: 70 },
  { headerName: 'Limit', field: 'limit_price', width: 70,
    valueFormatter: (p: any) => p.value?.toFixed(2) ?? '-' },
  { headerName: 'AvgPx', field: 'avg_fill_price', width: 70,
    valueFormatter: (p: any) => p.value ? p.value.toFixed(4) : '-' },
  { headerName: 'Status', field: 'status', width: 85,
    cellStyle: (p: any) => ({ color: statusColor(p.value), fontWeight: 600, textTransform: 'uppercase' }) },
  { headerName: 'Time', field: 'created_at', width: 160,
    valueFormatter: (p: any) => p.value ? new Date(p.value).toLocaleTimeString() : '' },
];

export default function OrderBlotter() {
  const [orders, setOrders] = useState<Order[]>([]);
  const [selectedId, setSelectedId] = useState<string | null>(null);

  const load = async () => {
    try {
      const res = await fetchOrders();
      setOrders(res.data);
    } catch (e) { console.error(e); }
  };

  useEffect(() => {
    load();
    const interval = setInterval(load, 3000);
    return () => clearInterval(interval);
  }, []);

  const handleCancel = async () => {
    if (selectedId) {
      await cancelOrder(selectedId);
      load();
    }
  };

  return (
    <div className="panel" style={{ display: 'flex', flexDirection: 'column', height: '100%' }}>
      <div className="panel-header" style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <span>Order Blotter ({orders.length} orders)</span>
        <div style={{ display: 'flex', gap: 8 }}>
          <button className="btn btn-cancel" onClick={handleCancel}
                  disabled={!selectedId} style={{ padding: '4px 12px', fontSize: 11 }}>
            Cancel Order
          </button>
          <button className="btn btn-primary" onClick={load} style={{ padding: '4px 12px', fontSize: 11 }}>
            ↻
          </button>
        </div>
      </div>
      <div className="ag-theme-alpine-dark" style={{ flex: 1, minHeight: 300 }}>
        <AgGridReact<Order>
          rowData={orders}
          columnDefs={columns}
          defaultColDef={{ sortable: true, resizable: true }}
          rowSelection="single"
          onRowClicked={(e) => setSelectedId(e.data?.order_id ?? null)}
          animateRows
        />
      </div>
    </div>
  );
}

