#include "performance_metrics.h"
#include <algorithm>
#include <numeric>
#include <cmath>
#include <iostream>
#include <iomanip>
#include <sstream>

namespace dcbacktest {

PerformanceCalculator::PerformanceCalculator(double initial_capital) 
    : initial_capital_(initial_capital) {}

void PerformanceCalculator::updatePortfolio(Timestamp timestamp, double cash, 
                                           double position_value, Price current_price) {
    double total_value = cash + position_value;
    portfolio_history_.emplace_back(timestamp, cash, position_value, total_value, current_price);
}

void PerformanceCalculator::recordTrade(const TradeRecord& trade) {
    trade_history_.push_back(trade);
}

PerformanceMetrics PerformanceCalculator::calculateMetrics() const {
    PerformanceMetrics metrics;
    
    if (portfolio_history_.empty()) {
        return metrics;
    }
    
    metrics.start_time = portfolio_history_.front().timestamp;
    metrics.end_time = portfolio_history_.back().timestamp;
    metrics.total_return = calculateTotalReturn();
    metrics.sharpe_ratio = calculateSharpeRatio();
    metrics.max_drawdown = calculateMaxDrawdown();
    metrics.win_rate = calculateWinRate();
    metrics.total_trades = static_cast<int>(trade_history_.size());
    metrics.avg_trade_duration_ms = calculateAverageTradeDuration();
    metrics.total_fees = std::accumulate(trade_history_.begin(), trade_history_.end(), 0.0,
                                        [](double sum, const TradeRecord& trade) {
                                            return sum + trade.fees;
                                        });
    
    return metrics;
}

void PerformanceCalculator::reset() {
    portfolio_history_.clear();
    trade_history_.clear();
}

double PerformanceCalculator::calculateTotalReturn() const {
    if (portfolio_history_.empty()) {
        return 0.0;
    }
    
    double final_value = portfolio_history_.back().total_value;
    return (final_value - initial_capital_) / initial_capital_ * 100.0;
}

double PerformanceCalculator::calculateSharpeRatio() const {
    auto returns = calculateReturns();
    if (returns.empty()) {
        return 0.0;
    }
    
    double mean_return = std::accumulate(returns.begin(), returns.end(), 0.0) / returns.size();
    double volatility = calculateVolatility(returns);
    
    return (volatility > 0) ? mean_return / volatility : 0.0;
}

double PerformanceCalculator::calculateMaxDrawdown() const {
    if (portfolio_history_.size() < 2) {
        return 0.0;
    }
    
    double max_value = portfolio_history_[0].total_value;
    double max_drawdown = 0.0;
    
    for (size_t i = 1; i < portfolio_history_.size(); ++i) {
        double current_value = portfolio_history_[i].total_value;
        max_value = std::max(max_value, current_value);
        
        double drawdown = (max_value - current_value) / max_value;
        max_drawdown = std::max(max_drawdown, drawdown);
    }
    
    return max_drawdown * 100.0; // Return as percentage
}

double PerformanceCalculator::calculateWinRate() const {
    if (trade_history_.empty()) {
        return 0.0;
    }
    
    int winning_trades = std::count_if(trade_history_.begin(), trade_history_.end(),
                                      [](const TradeRecord& trade) {
                                          return trade.pnl > 0;
                                      });
    
    return static_cast<double>(winning_trades) / trade_history_.size() * 100.0;
}

double PerformanceCalculator::calculateAverageTradeDuration() const {
    if (trade_history_.empty()) {
        return 0.0;
    }
    
    double total_duration = 0.0;
    for (const auto& trade : trade_history_) {
        auto duration = trade.exit_time - trade.entry_time;
        total_duration += std::chrono::duration<double, std::milli>(duration).count();
    }
    
    return total_duration / trade_history_.size();
}

std::vector<double> PerformanceCalculator::calculateReturns() const {
    std::vector<double> returns;
    
    if (portfolio_history_.size() < 2) {
        return returns;
    }
    
    for (size_t i = 1; i < portfolio_history_.size(); ++i) {
        double prev_value = portfolio_history_[i-1].total_value;
        double curr_value = portfolio_history_[i].total_value;
        
        if (prev_value > 0) {
            double return_pct = (curr_value - prev_value) / prev_value;
            returns.push_back(return_pct);
        }
    }
    
    return returns;
}

double PerformanceCalculator::calculateVolatility(const std::vector<double>& returns) const {
    if (returns.size() < 2) {
        return 0.0;
    }
    
    double mean = std::accumulate(returns.begin(), returns.end(), 0.0) / returns.size();
    
    double variance = 0.0;
    for (double ret : returns) {
        variance += (ret - mean) * (ret - mean);
    }
    variance /= (returns.size() - 1);
    
    return std::sqrt(variance);
}

// Utility functions implementation

namespace performance_utils {

std::string timestampToString(Timestamp timestamp) {
    auto time_point = std::chrono::system_clock::time_point(
        std::chrono::duration_cast<std::chrono::system_clock::duration>(timestamp));
    auto time_t = std::chrono::system_clock::to_time_t(time_point);
    
    std::stringstream ss;
    ss << std::put_time(std::localtime(&time_t), "%Y-%m-%d %H:%M:%S");
    return ss.str();
}

double calculateSharpeRatio(const std::vector<double>& returns, double risk_free_rate) {
    if (returns.empty()) {
        return 0.0;
    }
    
    double mean_return = std::accumulate(returns.begin(), returns.end(), 0.0) / returns.size();
    double excess_return = mean_return - risk_free_rate;
    
    if (returns.size() < 2) {
        return 0.0;
    }
    
    double variance = 0.0;
    for (double ret : returns) {
        variance += (ret - mean_return) * (ret - mean_return);
    }
    variance /= (returns.size() - 1);
    double volatility = std::sqrt(variance);
    
    return (volatility > 0) ? excess_return / volatility : 0.0;
}

double calculateMaxDrawdown(const std::vector<double>& portfolio_values) {
    if (portfolio_values.size() < 2) {
        return 0.0;
    }
    
    double max_value = portfolio_values[0];
    double max_drawdown = 0.0;
    
    for (size_t i = 1; i < portfolio_values.size(); ++i) {
        max_value = std::max(max_value, portfolio_values[i]);
        double drawdown = (max_value - portfolio_values[i]) / max_value;
        max_drawdown = std::max(max_drawdown, drawdown);
    }
    
    return max_drawdown;
}

double calculateCalmarRatio(double total_return, double max_drawdown) {
    return (max_drawdown > 0) ? total_return / max_drawdown : 0.0;
}

double calculateSortinoRatio(const std::vector<double>& returns, double target_return) {
    if (returns.empty()) {
        return 0.0;
    }
    
    double mean_return = std::accumulate(returns.begin(), returns.end(), 0.0) / returns.size();
    double excess_return = mean_return - target_return;
    
    // Calculate downside deviation
    double downside_variance = 0.0;
    int downside_count = 0;
    
    for (double ret : returns) {
        if (ret < target_return) {
            downside_variance += (ret - target_return) * (ret - target_return);
            downside_count++;
        }
    }
    
    if (downside_count == 0) {
        return 0.0;
    }
    
    downside_variance /= downside_count;
    double downside_deviation = std::sqrt(downside_variance);
    
    return (downside_deviation > 0) ? excess_return / downside_deviation : 0.0;
}

void printPerformanceReport(const PerformanceMetrics& metrics, 
                           const std::vector<TradeRecord>& trades) {
    std::cout << "\n=== PERFORMANCE REPORT ===" << std::endl;
    std::cout << std::fixed << std::setprecision(2);
    
    std::cout << "Period: " << timestampToString(metrics.start_time) 
              << " to " << timestampToString(metrics.end_time) << std::endl;
    
    std::cout << "Total Return: " << metrics.total_return << "%" << std::endl;
    std::cout << "Sharpe Ratio: " << metrics.sharpe_ratio << std::endl;
    std::cout << "Max Drawdown: " << metrics.max_drawdown << "%" << std::endl;
    std::cout << "Win Rate: " << metrics.win_rate << "%" << std::endl;
    std::cout << "Total Trades: " << metrics.total_trades << std::endl;
    std::cout << "Avg Trade Duration: " << metrics.avg_trade_duration_ms << " ms" << std::endl;
    std::cout << "Total Fees: $" << metrics.total_fees << std::endl;
    
    if (!trades.empty()) {
        double total_pnl = std::accumulate(trades.begin(), trades.end(), 0.0,
                                          [](double sum, const TradeRecord& trade) {
                                              return sum + trade.pnl;
                                          });
        std::cout << "Total P&L: $" << total_pnl << std::endl;
        
        auto winning_trades = std::count_if(trades.begin(), trades.end(),
                                           [](const TradeRecord& trade) {
                                               return trade.pnl > 0;
                                           });
        std::cout << "Winning Trades: " << winning_trades << "/" << trades.size() << std::endl;
    }
    
    std::cout << "=========================" << std::endl;
}

} // namespace performance_utils

} // namespace dcbacktest
