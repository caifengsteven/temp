#include "backtesting_engine.h"
#include "performance_metrics.h"
#include <iostream>
#include <string>
#include <vector>
#include <chrono>

using namespace dcbacktest;

void printUsage(const char* program_name) {
    std::cout << "Usage: " << program_name << " [options]" << std::endl;
    std::cout << "Options:" << std::endl;
    std::cout << "  --orderbook-db <path>    Path to orderbook SQLite database" << std::endl;
    std::cout << "  --trades-db <path>       Path to trades SQLite database" << std::endl;
    std::cout << "  --symbol <symbol>        Trading symbol to backtest" << std::endl;
    std::cout << "  --start-time <timestamp> Start timestamp (nanoseconds)" << std::endl;
    std::cout << "  --end-time <timestamp>   End timestamp (nanoseconds)" << std::endl;
    std::cout << "  --dc-threshold <value>   DC threshold (default: 0.001)" << std::endl;
    std::cout << "  --capital <amount>       Initial capital (default: 100000)" << std::endl;
    std::cout << "  --verbose                Enable verbose output" << std::endl;
    std::cout << "  --help                   Show this help message" << std::endl;
}

int main(int argc, char* argv[]) {
    // Default configuration - using a specific database file
    std::string orderbook_db_path = ""; // No orderbook data for now
    std::string trades_db_path = "I:/zhubi/cpp_implementation/sqlite_databases/2018/2018_01.db";
    std::string symbol = "BTCUSD"; // Default symbol - adjust based on your data
    
    BacktestConfig config;
    config.symbol = symbol;
    config.initial_capital = 100000.0;
    config.dc_threshold = 0.001; // 0.1%
    config.maker_fee_bps = 1.0;
    config.taker_fee_bps = 2.0;
    config.use_orderbook_data = false; // We only have trades data
    config.use_trades_data = true;
    
    // Set default time range (last 24 hours worth of nanoseconds)
    auto now = std::chrono::system_clock::now();
    auto yesterday = now - std::chrono::hours(24);
    config.end_time = std::chrono::duration_cast<std::chrono::nanoseconds>(
        now.time_since_epoch());
    config.start_time = std::chrono::duration_cast<std::chrono::nanoseconds>(
        yesterday.time_since_epoch());
    
    bool verbose = false;
    
    // Parse command line arguments
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        
        if (arg == "--help") {
            printUsage(argv[0]);
            return 0;
        } else if (arg == "--orderbook-db" && i + 1 < argc) {
            orderbook_db_path = argv[++i];
        } else if (arg == "--trades-db" && i + 1 < argc) {
            trades_db_path = argv[++i];
        } else if (arg == "--symbol" && i + 1 < argc) {
            config.symbol = argv[++i];
        } else if (arg == "--start-time" && i + 1 < argc) {
            config.start_time = Timestamp(std::stoull(argv[++i]));
        } else if (arg == "--end-time" && i + 1 < argc) {
            config.end_time = Timestamp(std::stoull(argv[++i]));
        } else if (arg == "--dc-threshold" && i + 1 < argc) {
            config.dc_threshold = std::stod(argv[++i]);
        } else if (arg == "--capital" && i + 1 < argc) {
            config.initial_capital = std::stod(argv[++i]);
        } else if (arg == "--verbose") {
            verbose = true;
        } else {
            std::cerr << "Unknown argument: " << arg << std::endl;
            printUsage(argv[0]);
            return 1;
        }
    }
    
    std::cout << "=== DC Generator High-Frequency Trading Backtest ===" << std::endl;
    std::cout << "Orderbook DB: " << orderbook_db_path << std::endl;
    std::cout << "Trades DB: " << trades_db_path << std::endl;
    std::cout << "Symbol: " << config.symbol << std::endl;
    std::cout << "DC Threshold: " << config.dc_threshold * 100 << "%" << std::endl;
    std::cout << "Initial Capital: $" << config.initial_capital << std::endl;
    
    try {
        // Create backtesting engine
        BacktestingEngine engine(orderbook_db_path, trades_db_path);
        engine.setVerbose(verbose);
        
        // Test multiple DC thresholds
        std::vector<double> thresholds = {0.0005, 0.001, 0.002, 0.005, 0.01};
        std::vector<PerformanceMetrics> results;
        
        std::cout << "\n=== Testing Multiple DC Thresholds ===" << std::endl;
        
        for (double threshold : thresholds) {
            std::cout << "\n--- Testing DC Threshold: " << threshold * 100 << "% ---" << std::endl;
            
            BacktestConfig test_config = config;
            test_config.dc_threshold = threshold;
            
            // Create strategy
            auto strategy = std::make_unique<SimpleDCStrategy>(0.95); // Use 95% of capital
            
            // Run backtest
            auto metrics = engine.runBacktest(test_config, std::move(strategy));
            results.push_back(metrics);
            
            // Print summary
            std::cout << "Total Return: " << std::fixed << std::setprecision(2) 
                      << metrics.total_return << "%" << std::endl;
            std::cout << "Sharpe Ratio: " << metrics.sharpe_ratio << std::endl;
            std::cout << "Max Drawdown: " << metrics.max_drawdown << "%" << std::endl;
            std::cout << "Total Trades: " << metrics.total_trades << std::endl;
        }
        
        // Find best threshold
        auto best_it = std::max_element(results.begin(), results.end(),
                                       [](const PerformanceMetrics& a, const PerformanceMetrics& b) {
                                           return a.total_return < b.total_return;
                                       });
        
        if (best_it != results.end()) {
            size_t best_index = std::distance(results.begin(), best_it);
            double best_threshold = thresholds[best_index];
            
            std::cout << "\n=== BEST RESULTS ===" << std::endl;
            std::cout << "Best DC Threshold: " << best_threshold * 100 << "%" << std::endl;
            performance_utils::printPerformanceReport(*best_it, {});
        }
        
        // Summary comparison
        std::cout << "\n=== THRESHOLD COMPARISON ===" << std::endl;
        std::cout << std::setw(12) << "Threshold" 
                  << std::setw(12) << "Return %" 
                  << std::setw(12) << "Sharpe" 
                  << std::setw(12) << "Max DD %" 
                  << std::setw(12) << "Trades" << std::endl;
        std::cout << std::string(60, '-') << std::endl;
        
        for (size_t i = 0; i < thresholds.size() && i < results.size(); ++i) {
            std::cout << std::fixed << std::setprecision(3)
                      << std::setw(12) << thresholds[i] * 100
                      << std::setw(12) << results[i].total_return
                      << std::setw(12) << results[i].sharpe_ratio
                      << std::setw(12) << results[i].max_drawdown
                      << std::setw(12) << results[i].total_trades << std::endl;
        }
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    std::cout << "\nBacktest completed successfully!" << std::endl;
    return 0;
}
