"""
Visualizer

Comprehensive visualization tools for LSTM-BEKK trading system analysis.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from typing import Dict, List, Optional, Tuple, Union
import logging


class Visualizer:
    """
    Comprehensive visualization toolkit for LSTM-BEKK trading system.
    
    Provides plotting functions for model outputs, portfolio performance,
    risk analysis, and strategy comparison.
    """
    
    def __init__(self, style: str = "seaborn-v0_8", figsize: Tuple[int, int] = (12, 8)):
        """
        Initialize visualizer.
        
        Args:
            style: Matplotlib style
            figsize: Default figure size
        """
        self.style = style
        self.figsize = figsize
        self.logger = logging.getLogger(__name__)
        
        # Set style
        plt.style.use(style)
        sns.set_palette("husl")
    
    def plot_volatility_forecasts(self, realized_vol: pd.Series,
                                 predicted_vol: pd.Series,
                                 title: str = "Volatility Forecasts vs Realized") -> go.Figure:
        """
        Plot volatility forecasts against realized volatility.
        
        Args:
            realized_vol: Realized volatility series
            predicted_vol: Predicted volatility series
            title: Plot title
            
        Returns:
            Plotly figure
        """
        fig = go.Figure()
        
        # Align series
        aligned_real, aligned_pred = realized_vol.align(predicted_vol, join='inner')
        
        fig.add_trace(go.Scatter(
            x=aligned_real.index,
            y=aligned_real.values,
            mode='lines',
            name='Realized Volatility',
            line=dict(color='blue', width=2)
        ))
        
        fig.add_trace(go.Scatter(
            x=aligned_pred.index,
            y=aligned_pred.values,
            mode='lines',
            name='LSTM-BEKK Forecast',
            line=dict(color='red', width=2, dash='dash')
        ))
        
        fig.update_layout(
            title=title,
            xaxis_title='Date',
            yaxis_title='Volatility',
            hovermode='x unified',
            template='plotly_white'
        )
        
        return fig
    
    def plot_correlation_heatmap(self, correlation_matrix: np.ndarray,
                               asset_names: List[str],
                               title: str = "Asset Correlation Matrix") -> go.Figure:
        """
        Plot correlation matrix heatmap.
        
        Args:
            correlation_matrix: Correlation matrix
            asset_names: Asset names
            title: Plot title
            
        Returns:
            Plotly figure
        """
        fig = go.Figure(data=go.Heatmap(
            z=correlation_matrix,
            x=asset_names,
            y=asset_names,
            colorscale='RdBu',
            zmid=0,
            text=np.round(correlation_matrix, 2),
            texttemplate="%{text}",
            textfont={"size": 10},
            hoverongaps=False
        ))
        
        fig.update_layout(
            title=title,
            template='plotly_white',
            width=600,
            height=600
        )
        
        return fig
    
    def plot_portfolio_performance(self, strategy_returns: Dict[str, pd.Series],
                                 title: str = "Portfolio Performance Comparison") -> go.Figure:
        """
        Plot cumulative performance of multiple strategies.
        
        Args:
            strategy_returns: Dictionary of strategy name -> return series
            title: Plot title
            
        Returns:
            Plotly figure
        """
        fig = go.Figure()
        
        colors = px.colors.qualitative.Set1
        
        for i, (strategy_name, returns) in enumerate(strategy_returns.items()):
            cumulative_returns = (1 + returns).cumprod()
            
            fig.add_trace(go.Scatter(
                x=cumulative_returns.index,
                y=cumulative_returns.values,
                mode='lines',
                name=strategy_name,
                line=dict(color=colors[i % len(colors)], width=2)
            ))
        
        fig.update_layout(
            title=title,
            xaxis_title='Date',
            yaxis_title='Cumulative Return',
            hovermode='x unified',
            template='plotly_white',
            legend=dict(x=0.02, y=0.98)
        )
        
        return fig
    
    def plot_drawdown_analysis(self, returns: pd.Series,
                             title: str = "Drawdown Analysis") -> go.Figure:
        """
        Plot drawdown analysis.
        
        Args:
            returns: Return series
            title: Plot title
            
        Returns:
            Plotly figure
        """
        # Calculate cumulative returns and drawdowns
        cumulative_returns = (1 + returns).cumprod()
        running_max = cumulative_returns.expanding().max()
        drawdowns = (cumulative_returns - running_max) / running_max
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=('Cumulative Returns', 'Drawdown'),
            vertical_spacing=0.1
        )
        
        # Cumulative returns
        fig.add_trace(
            go.Scatter(x=cumulative_returns.index, y=cumulative_returns.values,
                      mode='lines', name='Cumulative Returns',
                      line=dict(color='blue', width=2)),
            row=1, col=1
        )
        
        # Drawdown
        fig.add_trace(
            go.Scatter(x=drawdowns.index, y=drawdowns.values,
                      mode='lines', name='Drawdown', fill='tonexty',
                      line=dict(color='red', width=1),
                      fillcolor='rgba(255,0,0,0.3)'),
            row=2, col=1
        )
        
        fig.update_layout(
            title=title,
            template='plotly_white',
            showlegend=False
        )
        
        fig.update_xaxes(title_text="Date", row=2, col=1)
        fig.update_yaxes(title_text="Cumulative Return", row=1, col=1)
        fig.update_yaxes(title_text="Drawdown", row=2, col=1)
        
        return fig
    
    def plot_risk_metrics_comparison(self, risk_metrics: pd.DataFrame,
                                   title: str = "Risk Metrics Comparison") -> go.Figure:
        """
        Plot risk metrics comparison across strategies.
        
        Args:
            risk_metrics: DataFrame with risk metrics
            title: Plot title
            
        Returns:
            Plotly figure
        """
        # Select key risk metrics
        key_metrics = ['Annualized Volatility', 'Max Drawdown', 'VaR 5%', 'ES 5%']
        available_metrics = [m for m in key_metrics if m in risk_metrics.columns]
        
        if not available_metrics:
            self.logger.warning("No risk metrics available for plotting")
            return go.Figure()
        
        fig = go.Figure()
        
        strategies = risk_metrics.index.tolist()
        
        for metric in available_metrics:
            fig.add_trace(go.Bar(
                name=metric,
                x=strategies,
                y=risk_metrics[metric].values,
                text=np.round(risk_metrics[metric].values, 3),
                textposition='auto'
            ))
        
        fig.update_layout(
            title=title,
            xaxis_title='Strategy',
            yaxis_title='Risk Metric Value',
            barmode='group',
            template='plotly_white'
        )
        
        return fig
    
    def plot_weight_evolution(self, weights_history: np.ndarray,
                            asset_names: List[str],
                            dates: pd.DatetimeIndex,
                            title: str = "Portfolio Weight Evolution") -> go.Figure:
        """
        Plot evolution of portfolio weights over time.
        
        Args:
            weights_history: Array of weights over time (T x N)
            asset_names: Asset names
            dates: Date index
            title: Plot title
            
        Returns:
            Plotly figure
        """
        fig = go.Figure()
        
        colors = px.colors.qualitative.Set3
        
        for i, asset in enumerate(asset_names):
            fig.add_trace(go.Scatter(
                x=dates,
                y=weights_history[:, i],
                mode='lines',
                name=asset,
                line=dict(color=colors[i % len(colors)], width=2),
                stackgroup='one'
            ))
        
        fig.update_layout(
            title=title,
            xaxis_title='Date',
            yaxis_title='Portfolio Weight',
            hovermode='x unified',
            template='plotly_white'
        )
        
        return fig
    
    def plot_model_training_history(self, training_history: List[Dict],
                                  title: str = "Model Training History") -> go.Figure:
        """
        Plot model training history.
        
        Args:
            training_history: Training history from model
            title: Plot title
            
        Returns:
            Plotly figure
        """
        if not training_history:
            return go.Figure()
        
        epochs = [h['epoch'] for h in training_history]
        train_loss = [h['train_loss'] for h in training_history]
        val_loss = [h.get('val_loss') for h in training_history if h.get('val_loss') is not None]
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=epochs,
            y=train_loss,
            mode='lines',
            name='Training Loss',
            line=dict(color='blue', width=2)
        ))
        
        if val_loss and len(val_loss) == len(epochs):
            fig.add_trace(go.Scatter(
                x=epochs,
                y=val_loss,
                mode='lines',
                name='Validation Loss',
                line=dict(color='red', width=2)
            ))
        
        fig.update_layout(
            title=title,
            xaxis_title='Epoch',
            yaxis_title='Loss',
            template='plotly_white'
        )
        
        return fig
    
    def create_performance_dashboard(self, backtest_results: Dict) -> Dict[str, go.Figure]:
        """
        Create comprehensive performance dashboard.
        
        Args:
            backtest_results: Backtest results dictionary
            
        Returns:
            Dictionary of figures for dashboard
        """
        figures = {}
        
        try:
            # Strategy performance comparison
            strategy_returns = {}
            for name, result in backtest_results['strategy_results'].items():
                if 'returns' in result and len(result['returns']) > 0:
                    strategy_returns[name] = result['returns']
            
            if strategy_returns:
                figures['performance_comparison'] = self.plot_portfolio_performance(
                    strategy_returns, "Strategy Performance Comparison"
                )
            
            # Risk metrics comparison
            if 'performance_comparison' in backtest_results:
                figures['risk_comparison'] = self.plot_risk_metrics_comparison(
                    backtest_results['performance_comparison'],
                    "Risk Metrics Comparison"
                )
            
            # LSTM-BEKK specific plots
            if 'lstm_bekk' in backtest_results['strategy_results']:
                lstm_result = backtest_results['strategy_results']['lstm_bekk']
                
                # Drawdown analysis
                if 'returns' in lstm_result:
                    figures['drawdown_analysis'] = self.plot_drawdown_analysis(
                        lstm_result['returns'], "LSTM-BEKK Drawdown Analysis"
                    )
                
                # Training history
                if 'training_history' in lstm_result:
                    figures['training_history'] = self.plot_model_training_history(
                        lstm_result['training_history'], "LSTM-BEKK Training History"
                    )
            
        except Exception as e:
            self.logger.error(f"Error creating dashboard: {e}")
        
        return figures
