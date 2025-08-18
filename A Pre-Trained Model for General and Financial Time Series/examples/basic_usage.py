"""
Basic usage example for Delphyne model
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from delphyne import DelphyneModel, DelphyneConfig
from delphyne.data import WaveletDataGenerator, GARCHDataGenerator

def basic_forecasting_example():
    """Demonstrate basic forecasting with Delphyne."""
    print("=== Basic Forecasting Example ===")
    
    # Create a smaller model for demonstration
    config = DelphyneConfig(
        num_layers=4,
        hidden_size=256,
        num_attention_heads=8,
        intermediate_size=1024,
        patch_size=16,
        max_sequence_length=512,
        num_mixture_components=2
    )
    
    # Initialize model
    model = DelphyneModel(config)
    model.eval()
    
    print(f"Model created with {model.get_num_parameters():,} parameters")
    
    # Generate synthetic time series
    generator = WaveletDataGenerator(seed=42)
    time_series, metadata = generator.generate(
        batch_size=4,
        seq_len=128,
        num_variates=1
    )
    
    print(f"Input time series shape: {time_series.shape}")
    
    # Generate forecasts
    with torch.no_grad():
        forecasts = model.generate_forecasts(
            time_series=time_series,
            forecast_length=32,
            num_samples=100
        )
    
    print(f"Forecast samples shape: {forecasts['samples'].shape}")
    print(f"Mean forecast shape: {forecasts['mean'].shape}")
    print(f"Quantiles shape: {forecasts['quantiles'].shape}")
    
    # Plot results for first sample
    plt.figure(figsize=(12, 6))
    
    # Historical data
    historical = time_series[0].numpy()
    plt.plot(range(len(historical)), historical, 'b-', label='Historical', linewidth=2)
    
    # Forecast mean
    forecast_mean = forecasts['mean'][0].numpy()
    forecast_start = len(historical)
    forecast_range = range(forecast_start, forecast_start + len(forecast_mean))
    plt.plot(forecast_range, forecast_mean, 'r-', label='Forecast Mean', linewidth=2)
    
    # Forecast quantiles
    quantiles = forecasts['quantiles'][0].numpy()  # [seq_len, num_quantiles]
    quantile_levels = forecasts['quantile_levels'].numpy()
    
    # Plot confidence intervals
    for i, q in enumerate(quantile_levels):
        if q in [0.1, 0.9]:  # 80% confidence interval
            alpha = 0.3 if q == 0.1 else 0.3
            color = 'red'
            plt.plot(forecast_range, quantiles[:, i], '--', color=color, alpha=alpha)
            if q == 0.1:
                plt.fill_between(
                    forecast_range, 
                    quantiles[:, i], 
                    quantiles[:, -1-i],  # corresponding upper quantile
                    alpha=0.2, 
                    color=color,
                    label='80% Confidence'
                )
    
    plt.axvline(x=forecast_start, color='gray', linestyle='--', alpha=0.7, label='Forecast Start')
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.title('Delphyne Time Series Forecasting')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('basic_forecasting_example.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print("✓ Basic forecasting example completed!")
    print("Plot saved as 'basic_forecasting_example.png'")


def multivariate_example():
    """Demonstrate multivariate time series forecasting."""
    print("\n=== Multivariate Forecasting Example ===")
    
    config = DelphyneConfig(
        num_layers=2,
        hidden_size=128,
        num_attention_heads=4,
        intermediate_size=512,
        patch_size=8,
        max_sequence_length=256
    )
    
    model = DelphyneModel(config)
    model.eval()
    
    # Generate multivariate time series
    generator = WaveletDataGenerator(seed=123)
    time_series, metadata = generator.generate(
        batch_size=2,
        seq_len=64,
        num_variates=3,
        correlated=True
    )
    
    print(f"Multivariate time series shape: {time_series.shape}")
    
    # Create variate IDs
    batch_size, num_variates, seq_len = time_series.shape
    variate_ids = torch.arange(num_variates).unsqueeze(1).repeat(1, seq_len)
    variate_ids = variate_ids.flatten().unsqueeze(0).repeat(batch_size, 1)
    
    # Forward pass
    with torch.no_grad():
        outputs = model(
            time_series=time_series,
            variate_ids=variate_ids,
            return_dict=True
        )
    
    print(f"Model output distribution: {type(outputs['distribution'])}")
    print(f"Hidden states shape: {outputs['last_hidden_state'].shape}")
    
    # Sample from the distribution
    samples = outputs['distribution'].sample((10,))
    print(f"Generated samples shape: {samples.shape}")
    
    print("✓ Multivariate example completed!")


def garch_data_example():
    """Demonstrate forecasting with GARCH data."""
    print("\n=== GARCH Data Forecasting Example ===")
    
    config = DelphyneConfig(
        num_layers=2,
        hidden_size=128,
        num_attention_heads=4,
        patch_size=8
    )
    
    model = DelphyneModel(config)
    model.eval()
    
    # Generate GARCH time series
    generator = GARCHDataGenerator(
        omega=0.01,
        alpha=0.1,
        beta=0.8,
        seed=456
    )
    
    time_series, metadata = generator.generate(
        batch_size=3,
        seq_len=100,
        num_variates=1
    )
    
    print(f"GARCH time series shape: {time_series.shape}")
    print(f"GARCH parameters: ω={metadata['omega']}, α={metadata['alpha']}, β={metadata['beta']}")
    
    # Generate forecasts
    with torch.no_grad():
        forecasts = model.generate_forecasts(
            time_series=time_series,
            forecast_length=20,
            num_samples=50
        )
    
    print(f"GARCH forecast samples shape: {forecasts['samples'].shape}")
    
    # Plot volatility clustering
    plt.figure(figsize=(12, 4))
    
    for i in range(3):
        plt.subplot(1, 3, i+1)
        data = time_series[i].numpy()
        plt.plot(data, 'b-', alpha=0.7)
        plt.title(f'GARCH Series {i+1}')
        plt.xlabel('Time')
        plt.ylabel('Returns')
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('garch_data_example.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print("✓ GARCH data example completed!")
    print("Plot saved as 'garch_data_example.png'")


def probabilistic_forecasting_example():
    """Demonstrate probabilistic forecasting capabilities."""
    print("\n=== Probabilistic Forecasting Example ===")
    
    config = DelphyneConfig(
        num_layers=2,
        hidden_size=128,
        num_attention_heads=4,
        num_mixture_components=3  # Multiple components for richer distributions
    )
    
    model = DelphyneModel(config)
    model.eval()
    
    # Generate test data
    generator = WaveletDataGenerator(seed=789)
    time_series, _ = generator.generate(batch_size=1, seq_len=80, num_variates=1)
    
    # Generate many samples for probabilistic analysis
    with torch.no_grad():
        forecasts = model.generate_forecasts(
            time_series=time_series,
            forecast_length=20,
            num_samples=1000  # Many samples for good statistics
        )
    
    # Analyze forecast distribution
    samples = forecasts['samples'][0, 0, :].numpy()  # [num_samples, forecast_length]
    
    plt.figure(figsize=(15, 5))
    
    # Plot 1: Sample trajectories
    plt.subplot(1, 3, 1)
    historical = time_series[0].numpy()
    plt.plot(range(len(historical)), historical, 'b-', label='Historical', linewidth=2)
    
    forecast_start = len(historical)
    forecast_range = range(forecast_start, forecast_start + samples.shape[1])
    
    # Plot some sample trajectories
    for i in range(0, 1000, 50):  # Every 50th sample
        plt.plot(forecast_range, samples[i], 'r-', alpha=0.1)
    
    plt.axvline(x=forecast_start, color='gray', linestyle='--', alpha=0.7)
    plt.title('Sample Forecast Trajectories')
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 2: Forecast distribution at specific time points
    plt.subplot(1, 3, 2)
    time_points = [0, 9, 19]  # Beginning, middle, end of forecast
    for t in time_points:
        values = samples[:, t]
        plt.hist(values, bins=50, alpha=0.6, label=f't+{t+1}', density=True)
    
    plt.title('Forecast Distribution at Different Times')
    plt.xlabel('Value')
    plt.ylabel('Density')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 3: Uncertainty evolution
    plt.subplot(1, 3, 3)
    mean_forecast = samples.mean(axis=0)
    std_forecast = samples.std(axis=0)
    
    plt.plot(forecast_range, mean_forecast, 'r-', label='Mean', linewidth=2)
    plt.fill_between(
        forecast_range,
        mean_forecast - 2*std_forecast,
        mean_forecast + 2*std_forecast,
        alpha=0.3,
        label='±2σ'
    )
    plt.fill_between(
        forecast_range,
        mean_forecast - std_forecast,
        mean_forecast + std_forecast,
        alpha=0.5,
        label='±1σ'
    )
    
    plt.title('Forecast Uncertainty Evolution')
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('probabilistic_forecasting_example.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print(f"Mean forecast uncertainty (std): {std_forecast.mean():.4f}")
    print(f"Uncertainty range: {std_forecast.min():.4f} - {std_forecast.max():.4f}")
    print("✓ Probabilistic forecasting example completed!")
    print("Plot saved as 'probabilistic_forecasting_example.png'")


if __name__ == "__main__":
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    print("Delphyne Model Examples")
    print("=" * 50)
    
    try:
        basic_forecasting_example()
        multivariate_example()
        garch_data_example()
        probabilistic_forecasting_example()
        
        print("\n*** All examples completed successfully! ***")
        
    except ImportError as e:
        if "matplotlib" in str(e):
            print("Note: Matplotlib not available, skipping visualization examples")
            multivariate_example()  # This one doesn't require matplotlib
        else:
            raise e
