#pragma once

#include "types.h"
#include "dc_generator.h"
#include "database_reader.h"
#include "order_manager.h"
#include "performance_metrics.h"
#include <memory>
#include <string>

namespace dcbacktest {

struct BacktestConfig {
    std::string symbol;
    Timestamp start_time;
    Timestamp end_time;
    double initial_capital;
    double dc_threshold;
    double maker_fee_bps;
    double taker_fee_bps;
    std::chrono::nanoseconds order_latency;
    std::chrono::nanoseconds cancel_latency;
    bool use_orderbook_data;
    bool use_trades_data;
    
    BacktestConfig() 
        : initial_capital(100000.0)
        , dc_threshold(0.001)
        , maker_fee_bps(1.0)
        , taker_fee_bps(2.0)
        , order_latency(std::chrono::microseconds(100))
        , cancel_latency(std::chrono::microseconds(50))
        , use_orderbook_data(true)
        , use_trades_data(true) {}
};

class TradingStrategy {
public:
    virtual ~TradingStrategy() = default;
    
    // Called when a new DC event occurs
    virtual void onDCEvent(DCEvent event, Price price, Timestamp timestamp, 
                          OrderManager& order_manager) = 0;
    
    // Called on market data update
    virtual void onMarketUpdate(const OrderBookSnapshot& orderbook, Timestamp timestamp,
                               OrderManager& order_manager) {}
    
    // Called on trade update
    virtual void onTrade(const Trade& trade, Timestamp timestamp,
                        OrderManager& order_manager) {}
    
    // Called at the start of backtesting
    virtual void onStart(const BacktestConfig& config) {}
    
    // Called at the end of backtesting
    virtual void onEnd(const PerformanceMetrics& metrics) {}
};

class SimpleDCStrategy : public TradingStrategy {
public:
    SimpleDCStrategy(double position_size_pct = 0.95);
    
    void onDCEvent(DCEvent event, Price price, Timestamp timestamp, 
                   OrderManager& order_manager) override;
    
    void onStart(const BacktestConfig& config) override;

private:
    double position_size_pct_;
    double initial_capital_;
    bool has_position_;
    OrderId current_order_id_;
};

class BacktestingEngine {
public:
    BacktestingEngine(const std::string& orderbook_db_path, 
                     const std::string& trades_db_path);
    ~BacktestingEngine() = default;

    // Run backtest with given configuration and strategy
    PerformanceMetrics runBacktest(const BacktestConfig& config, 
                                  std::unique_ptr<TradingStrategy> strategy);
    
    // Set verbose output
    void setVerbose(bool verbose) { verbose_ = verbose; }

private:
    std::unique_ptr<OrderBookReader> orderbook_reader_;
    std::unique_ptr<TradesReader> trades_reader_;
    std::unique_ptr<DCGenerator> dc_generator_;
    std::unique_ptr<DCEventHistory> dc_history_;
    std::unique_ptr<OrderManager> order_manager_;
    std::unique_ptr<PerformanceCalculator> perf_calculator_;
    bool verbose_;
    
    // Helper methods
    void processMarketData(const BacktestConfig& config, TradingStrategy& strategy);
    void processTradesData(const BacktestConfig& config, TradingStrategy& strategy);
    void logProgress(Timestamp current_time, Timestamp start_time, Timestamp end_time);
};

} // namespace dcbacktest
