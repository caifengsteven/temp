import { useState } from 'react';
import PortfolioGrid from './components/PortfolioGrid';
import VolSurface from './components/VolSurface';
import OrderEntry from './components/OrderEntry';
import OrderBlotter from './components/OrderBlotter';
import './App.css';

type Tab = 'portfolio' | 'volsurface' | 'trading';

export default function App() {
  const [activeTab, setActiveTab] = useState<Tab>('portfolio');

  return (
    <div className="app">
      <header className="app-header">
        <div className="logo">
          <span className="logo-icon">◈</span>
          <h1>ORC Trading Platform</h1>
        </div>
        <nav className="tabs">
          <button className={activeTab === 'portfolio' ? 'active' : ''}
                  onClick={() => setActiveTab('portfolio')}>
            Portfolio &amp; Greeks
          </button>
          <button className={activeTab === 'volsurface' ? 'active' : ''}
                  onClick={() => setActiveTab('volsurface')}>
            Vol Surface
          </button>
          <button className={activeTab === 'trading' ? 'active' : ''}
                  onClick={() => setActiveTab('trading')}>
            Trading
          </button>
        </nav>
      </header>
      <main className="app-main">
        {activeTab === 'portfolio' && <PortfolioGrid />}
        {activeTab === 'volsurface' && <VolSurface />}
        {activeTab === 'trading' && (
          <div className="trading-layout">
            <OrderEntry />
            <OrderBlotter />
          </div>
        )}
      </main>
    </div>
  );
}

