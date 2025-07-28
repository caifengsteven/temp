# DC Generator High-Frequency Trading Backtesting System

A high-performance C++ backtesting framework for testing Directional Change (DC) Generator algorithms on high-frequency trading data.

## Features

- **High-Performance C++ Implementation**: Optimized for processing large volumes of tick-by-tick market data
- **SQLite Database Integration**: Reads orderbook and trades data from SQLite databases
- **DC Generator Algorithm**: Implements the Directional Change event detection algorithm
- **Realistic Order Management**: Simulates order execution with latency and fees
- **Comprehensive Performance Metrics**: Calculates Sharpe ratio, maximum drawdown, win rate, and more
- **Multiple Threshold Testing**: Automatically tests multiple DC thresholds to find optimal parameters
- **Data Quality Checking**: Validates input data quality and reports issues

## Project Structure

```
cpp_backtesting/
├── CMakeLists.txt          # CMake build configuration
├── build.bat               # Windows build script
├── build.sh                # Linux/Mac build script
├── README.md               # This file
├── include/                # Header files
│   ├── types.h             # Common type definitions
│   ├── dc_generator.h      # DC algorithm implementation
│   ├── database_reader.h   # SQLite database interface
│   ├── order_manager.h     # Order management system
│   ├── backtesting_engine.h # Main backtesting engine
│   ├── performance_metrics.h # Performance calculation
│   └── market_data.h       # Market data processing
└── src/                    # Source files
    ├── main.cpp            # Main application
    ├── dc_generator.cpp    # DC algorithm implementation
    ├── database_reader.cpp # Database reading logic
    ├── order_manager.cpp   # Order management
    ├── backtesting_engine.cpp # Backtesting engine
    ├── performance_metrics.cpp # Performance calculations
    └── market_data.cpp     # Market data processing
```

## Prerequisites

- **C++17 compatible compiler** (GCC 7+, Clang 5+, MSVC 2017+)
- **CMake 3.16+**
- **SQLite3 development libraries**

### Installing Dependencies

#### Windows (with vcpkg)
```bash
vcpkg install sqlite3:x64-windows
```

#### Ubuntu/Debian
```bash
sudo apt-get install libsqlite3-dev cmake build-essential
```

#### macOS
```bash
brew install sqlite cmake
```

## Building

### Windows
```bash
cd cpp_backtesting
build.bat
```

### Linux/macOS
```bash
cd cpp_backtesting
chmod +x build.sh
./build.sh
```

## Usage

### Basic Usage
```bash
./bin/DCGeneratorBacktesting --orderbook-db /path/to/orderbook.db --trades-db /path/to/trades.db --symbol BTCUSDT
```

### Command Line Options

- `--orderbook-db <path>`: Path to orderbook SQLite database
- `--trades-db <path>`: Path to trades SQLite database  
- `--symbol <symbol>`: Trading symbol to backtest (e.g., BTCUSDT)
- `--start-time <timestamp>`: Start timestamp in nanoseconds
- `--end-time <timestamp>`: End timestamp in nanoseconds
- `--dc-threshold <value>`: DC threshold (default: 0.001 = 0.1%)
- `--capital <amount>`: Initial capital (default: 100000)
- `--verbose`: Enable verbose output
- `--help`: Show help message

### Example with Your Data
```bash
./bin/DCGeneratorBacktesting \
  --orderbook-db "J:/fenbi/cpp_implementation/sqlite_databases" \
  --trades-db "I:/zhubi/cpp_implementation/sqlite_databases" \
  --symbol BTCUSDT \
  --dc-threshold 0.001 \
  --capital 100000 \
  --verbose
```

## Database Schema Requirements

The system expects the following SQLite table schemas:

### Orderbook Table
```sql
CREATE TABLE orderbook (
    timestamp INTEGER,
    symbol TEXT,
    bid_prices TEXT,    -- JSON or comma-separated values
    bid_quantities TEXT,
    ask_prices TEXT,
    ask_quantities TEXT
);
```

### Trades Table
```sql
CREATE TABLE trades (
    timestamp INTEGER,
    symbol TEXT,
    price REAL,
    quantity REAL,
    side TEXT           -- 'buy', 'sell', 'BUY', 'SELL', or '1'/'0'
);
```

## Algorithm Details

### Directional Change (DC) Generator

The DC algorithm identifies significant price movements by detecting when the price changes by a specified threshold from a local extreme:

1. **Upturn Detection**: Price rises by threshold % from the lowest point
2. **Downturn Detection**: Price falls by threshold % from the highest point

### Trading Strategy

The implemented strategy uses a simple approach:
- **Buy Signal**: When an "end downturn" event occurs (market turns up)
- **Sell Signal**: When an "end upturn" event occurs (market turns down)

### Performance Metrics

The system calculates comprehensive performance metrics:
- **Total Return**: Overall percentage return
- **Sharpe Ratio**: Risk-adjusted return measure
- **Maximum Drawdown**: Largest peak-to-trough decline
- **Win Rate**: Percentage of profitable trades
- **Average Trade Duration**: Mean time between entry and exit
- **Total Fees**: Transaction costs

## Customization

### Creating Custom Strategies

Implement the `TradingStrategy` interface:

```cpp
class MyCustomStrategy : public TradingStrategy {
public:
    void onDCEvent(DCEvent event, Price price, Timestamp timestamp, 
                   OrderManager& order_manager) override {
        // Your trading logic here
    }
    
    void onMarketUpdate(const OrderBookSnapshot& orderbook, Timestamp timestamp,
                       OrderManager& order_manager) override {
        // Optional: React to orderbook updates
    }
};
```

### Modifying DC Parameters

You can test different DC thresholds by modifying the `thresholds` vector in `main.cpp`:

```cpp
std::vector<double> thresholds = {0.0001, 0.0005, 0.001, 0.002, 0.005, 0.01, 0.02};
```

## Performance Optimization

- **Compiled with O3 optimization** for maximum performance
- **Memory-efficient data structures** for handling large datasets
- **Minimal memory allocations** in the main processing loop
- **SQLite prepared statements** for efficient database queries

## Expected Output

The system will output:
1. **Data loading progress** and statistics
2. **Data quality report** (if verbose mode enabled)
3. **Trading signals** as they occur
4. **Performance metrics** for each DC threshold tested
5. **Comparison table** showing results across all thresholds
6. **Best performing configuration**

## Troubleshooting

### Common Issues

1. **SQLite not found**: Install SQLite development libraries
2. **Database connection failed**: Check database paths and permissions
3. **No data loaded**: Verify database schema and symbol names
4. **Build errors**: Ensure C++17 compiler and CMake 3.16+

### Performance Tips

- Use SSD storage for database files
- Ensure sufficient RAM for large datasets
- Consider data preprocessing for very large time ranges
- Use release build for maximum performance

## License

This project is released under the MIT License (same as the original DC Generator Python implementation).
