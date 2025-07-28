#pragma once

#include "types.h"
#include <vector>
#include <memory>

namespace dcbacktest {

struct PortfolioSnapshot {
    Timestamp timestamp;
    double cash;
    double position_value;
    double total_value;
    Price current_price;
    
    PortfolioSnapshot(Timestamp ts, double c, double pv, double tv, Price p)
        : timestamp(ts), cash(c), position_value(pv), total_value(tv), current_price(p) {}
};

struct TradeRecord {
    Timestamp entry_time;
    Timestamp exit_time;
    Price entry_price;
    Price exit_price;
    Quantity quantity;
    Side side;
    double pnl;
    double fees;
    
    TradeRecord() = default;
    TradeRecord(Timestamp et, Timestamp xt, Price ep, Price xp, Quantity q, Side s, double p, double f)
        : entry_time(et), exit_time(xt), entry_price(ep), exit_price(xp), 
          quantity(q), side(s), pnl(p), fees(f) {}
};

class PerformanceCalculator {
public:
    PerformanceCalculator(double initial_capital);
    ~PerformanceCalculator() = default;

    // Update portfolio state
    void updatePortfolio(Timestamp timestamp, double cash, double position_value, 
                        Price current_price);
    
    // Record a completed trade
    void recordTrade(const TradeRecord& trade);
    
    // Calculate final performance metrics
    PerformanceMetrics calculateMetrics() const;
    
    // Get portfolio history
    const std::vector<PortfolioSnapshot>& getPortfolioHistory() const { return portfolio_history_; }
    
    // Get trade history
    const std::vector<TradeRecord>& getTradeHistory() const { return trade_history_; }
    
    // Reset calculator
    void reset();

private:
    double initial_capital_;
    std::vector<PortfolioSnapshot> portfolio_history_;
    std::vector<TradeRecord> trade_history_;
    
    // Helper methods for calculations
    double calculateTotalReturn() const;
    double calculateSharpeRatio() const;
    double calculateMaxDrawdown() const;
    double calculateWinRate() const;
    double calculateAverageTradeDuration() const;
    std::vector<double> calculateReturns() const;
    double calculateVolatility(const std::vector<double>& returns) const;
};

// Utility functions for performance analysis
namespace performance_utils {
    
    // Convert timestamp to string for reporting
    std::string timestampToString(Timestamp timestamp);
    
    // Calculate Sharpe ratio from returns
    double calculateSharpeRatio(const std::vector<double>& returns, double risk_free_rate = 0.0);
    
    // Calculate maximum drawdown from portfolio values
    double calculateMaxDrawdown(const std::vector<double>& portfolio_values);
    
    // Calculate Calmar ratio
    double calculateCalmarRatio(double total_return, double max_drawdown);
    
    // Calculate Sortino ratio
    double calculateSortinoRatio(const std::vector<double>& returns, double target_return = 0.0);
    
    // Print performance report
    void printPerformanceReport(const PerformanceMetrics& metrics, 
                               const std::vector<TradeRecord>& trades);
}

} // namespace dcbacktest
