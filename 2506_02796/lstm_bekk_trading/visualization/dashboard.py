"""
Dashboard

Interactive dashboard for monitoring LSTM-BEKK trading system.
"""

import dash
from dash import dcc, html, Input, Output, callback
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from typing import Dict, List, Optional
import logging


class Dashboard:
    """
    Interactive dashboard for LSTM-BEKK trading system monitoring.
    """
    
    def __init__(self, backtest_results: Optional[Dict] = None):
        """
        Initialize dashboard.
        
        Args:
            backtest_results: Backtest results to display
        """
        self.backtest_results = backtest_results
        self.logger = logging.getLogger(__name__)
        
        # Initialize Dash app
        self.app = dash.Dash(__name__)
        self.setup_layout()
        self.setup_callbacks()
    
    def setup_layout(self):
        """Setup dashboard layout."""
        self.app.layout = html.Div([
            html.H1("LSTM-BEKK Trading System Dashboard", 
                   style={'textAlign': 'center', 'marginBottom': 30}),
            
            # Summary cards
            html.Div([
                html.Div([
                    html.H3("Total Return"),
                    html.H2(id="total-return", children="--")
                ], className="summary-card", style={'width': '23%', 'display': 'inline-block', 'margin': '1%'}),
                
                html.Div([
                    html.H3("Sharpe Ratio"),
                    html.H2(id="sharpe-ratio", children="--")
                ], className="summary-card", style={'width': '23%', 'display': 'inline-block', 'margin': '1%'}),
                
                html.Div([
                    html.H3("Max Drawdown"),
                    html.H2(id="max-drawdown", children="--")
                ], className="summary-card", style={'width': '23%', 'display': 'inline-block', 'margin': '1%'}),
                
                html.Div([
                    html.H3("Volatility"),
                    html.H2(id="volatility", children="--")
                ], className="summary-card", style={'width': '23%', 'display': 'inline-block', 'margin': '1%'})
            ], style={'marginBottom': 30}),
            
            # Strategy selector
            html.Div([
                html.Label("Select Strategy:"),
                dcc.Dropdown(
                    id='strategy-dropdown',
                    options=[],
                    value=None,
                    style={'width': '300px'}
                )
            ], style={'marginBottom': 20}),
            
            # Main charts
            html.Div([
                # Performance chart
                html.Div([
                    dcc.Graph(id='performance-chart')
                ], style={'width': '50%', 'display': 'inline-block'}),
                
                # Drawdown chart
                html.Div([
                    dcc.Graph(id='drawdown-chart')
                ], style={'width': '50%', 'display': 'inline-block'})
            ]),
            
            html.Div([
                # Risk metrics chart
                html.Div([
                    dcc.Graph(id='risk-metrics-chart')
                ], style={'width': '50%', 'display': 'inline-block'}),
                
                # Correlation heatmap
                html.Div([
                    dcc.Graph(id='correlation-heatmap')
                ], style={'width': '50%', 'display': 'inline-block'})
            ]),
            
            # Performance table
            html.Div([
                html.H3("Performance Comparison"),
                html.Div(id='performance-table')
            ], style={'marginTop': 30})
        ])
    
    def setup_callbacks(self):
        """Setup dashboard callbacks."""
        
        @self.app.callback(
            [Output('strategy-dropdown', 'options'),
             Output('strategy-dropdown', 'value')],
            [Input('strategy-dropdown', 'id')]
        )
        def update_strategy_options(_):
            if not self.backtest_results:
                return [], None
            
            strategies = list(self.backtest_results.get('strategy_results', {}).keys())
            options = [{'label': s, 'value': s} for s in strategies]
            default_value = 'lstm_bekk' if 'lstm_bekk' in strategies else strategies[0] if strategies else None
            
            return options, default_value
        
        @self.app.callback(
            [Output('total-return', 'children'),
             Output('sharpe-ratio', 'children'),
             Output('max-drawdown', 'children'),
             Output('volatility', 'children')],
            [Input('strategy-dropdown', 'value')]
        )
        def update_summary_cards(selected_strategy):
            if not self.backtest_results or not selected_strategy:
                return "--", "--", "--", "--"
            
            try:
                comparison = self.backtest_results.get('performance_comparison', pd.DataFrame())
                if selected_strategy in comparison.index:
                    metrics = comparison.loc[selected_strategy]
                    
                    total_return = f"{metrics['Total Return']:.2%}"
                    sharpe_ratio = f"{metrics['Sharpe Ratio']:.2f}"
                    max_drawdown = f"{metrics['Max Drawdown']:.2%}"
                    volatility = f"{metrics['Annualized Volatility']:.2%}"
                    
                    return total_return, sharpe_ratio, max_drawdown, volatility
            except Exception as e:
                self.logger.error(f"Error updating summary cards: {e}")
            
            return "--", "--", "--", "--"
        
        @self.app.callback(
            Output('performance-chart', 'figure'),
            [Input('strategy-dropdown', 'value')]
        )
        def update_performance_chart(selected_strategy):
            if not self.backtest_results:
                return go.Figure()
            
            try:
                strategy_results = self.backtest_results.get('strategy_results', {})
                
                fig = go.Figure()
                
                for name, result in strategy_results.items():
                    if 'returns' in result and len(result['returns']) > 0:
                        returns = result['returns']
                        cumulative_returns = (1 + returns).cumprod()
                        
                        line_width = 3 if name == selected_strategy else 1
                        opacity = 1.0 if name == selected_strategy else 0.6
                        
                        fig.add_trace(go.Scatter(
                            x=cumulative_returns.index,
                            y=cumulative_returns.values,
                            mode='lines',
                            name=name,
                            line=dict(width=line_width),
                            opacity=opacity
                        ))
                
                fig.update_layout(
                    title="Cumulative Performance",
                    xaxis_title="Date",
                    yaxis_title="Cumulative Return",
                    template='plotly_white'
                )
                
                return fig
                
            except Exception as e:
                self.logger.error(f"Error updating performance chart: {e}")
                return go.Figure()
        
        @self.app.callback(
            Output('drawdown-chart', 'figure'),
            [Input('strategy-dropdown', 'value')]
        )
        def update_drawdown_chart(selected_strategy):
            if not self.backtest_results or not selected_strategy:
                return go.Figure()
            
            try:
                strategy_results = self.backtest_results.get('strategy_results', {})
                
                if selected_strategy in strategy_results:
                    returns = strategy_results[selected_strategy].get('returns')
                    
                    if returns is not None and len(returns) > 0:
                        cumulative_returns = (1 + returns).cumprod()
                        running_max = cumulative_returns.expanding().max()
                        drawdowns = (cumulative_returns - running_max) / running_max
                        
                        fig = go.Figure()
                        fig.add_trace(go.Scatter(
                            x=drawdowns.index,
                            y=drawdowns.values,
                            mode='lines',
                            fill='tonexty',
                            name='Drawdown',
                            line=dict(color='red'),
                            fillcolor='rgba(255,0,0,0.3)'
                        ))
                        
                        fig.update_layout(
                            title=f"Drawdown Analysis - {selected_strategy}",
                            xaxis_title="Date",
                            yaxis_title="Drawdown",
                            template='plotly_white'
                        )
                        
                        return fig
                
            except Exception as e:
                self.logger.error(f"Error updating drawdown chart: {e}")
            
            return go.Figure()
        
        @self.app.callback(
            Output('risk-metrics-chart', 'figure'),
            [Input('strategy-dropdown', 'value')]
        )
        def update_risk_metrics_chart(selected_strategy):
            if not self.backtest_results:
                return go.Figure()
            
            try:
                comparison = self.backtest_results.get('performance_comparison', pd.DataFrame())
                
                if not comparison.empty:
                    metrics = ['Annualized Volatility', 'Max Drawdown', 'VaR 5%']
                    available_metrics = [m for m in metrics if m in comparison.columns]
                    
                    if available_metrics:
                        fig = go.Figure()
                        
                        for metric in available_metrics:
                            colors = ['red' if idx == selected_strategy else 'lightblue' 
                                    for idx in comparison.index]
                            
                            fig.add_trace(go.Bar(
                                name=metric,
                                x=comparison.index,
                                y=comparison[metric].abs().values,  # Use absolute values
                                marker_color=colors
                            ))
                        
                        fig.update_layout(
                            title="Risk Metrics Comparison",
                            xaxis_title="Strategy",
                            yaxis_title="Risk Metric Value",
                            barmode='group',
                            template='plotly_white'
                        )
                        
                        return fig
                
            except Exception as e:
                self.logger.error(f"Error updating risk metrics chart: {e}")
            
            return go.Figure()
        
        @self.app.callback(
            Output('performance-table', 'children'),
            [Input('strategy-dropdown', 'value')]
        )
        def update_performance_table(selected_strategy):
            if not self.backtest_results:
                return html.Div("No data available")
            
            try:
                comparison = self.backtest_results.get('performance_comparison', pd.DataFrame())
                
                if not comparison.empty:
                    # Format the dataframe for display
                    display_df = comparison.round(4)
                    
                    # Create HTML table
                    table_header = [html.Tr([html.Th("Strategy")] + [html.Th(col) for col in display_df.columns])]
                    
                    table_rows = []
                    for idx, row in display_df.iterrows():
                        style = {'backgroundColor': '#f0f0f0'} if idx == selected_strategy else {}
                        table_rows.append(html.Tr([html.Td(idx, style=style)] + 
                                                [html.Td(f"{val:.4f}", style=style) for val in row]))
                    
                    table = html.Table(table_header + table_rows, 
                                     style={'width': '100%', 'textAlign': 'center'})
                    
                    return table
                
            except Exception as e:
                self.logger.error(f"Error updating performance table: {e}")
            
            return html.Div("Error loading performance data")
    
    def run(self, host: str = '127.0.0.1', port: int = 8050, debug: bool = True):
        """
        Run the dashboard.
        
        Args:
            host: Host address
            port: Port number
            debug: Debug mode
        """
        self.logger.info(f"Starting dashboard at http://{host}:{port}")
        self.app.run_server(host=host, port=port, debug=debug)
