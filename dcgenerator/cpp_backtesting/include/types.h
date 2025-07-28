#pragma once

#include <string>
#include <vector>
#include <chrono>
#include <cstdint>

namespace dcbacktest {

using Timestamp = std::chrono::nanoseconds;
using Price = double;
using Quantity = double;
using OrderId = std::uint64_t;

enum class Side {
    BUY,
    SELL
};

enum class OrderType {
    MARKET,
    LIMIT
};

enum class OrderStatus {
    PENDING,
    FILLED,
    PARTIALLY_FILLED,
    CANCELLED,
    REJECTED
};

enum class DCEvent {
    NONE,
    START_UPTURN,
    END_UPTURN,
    START_DOWNTURN,
    END_DOWNTURN,
    START_UPWARD_OS,
    END_UPWARD_OS,
    START_DOWNWARD_OS,
    END_DOWNWARD_OS
};

struct Tick {
    Timestamp timestamp;
    Price price;
    Quantity volume;
    
    Tick() = default;
    Tick(Timestamp ts, Price p, Quantity v) : timestamp(ts), price(p), volume(v) {}
};

struct OrderBookLevel {
    Price price;
    Quantity quantity;
    
    OrderBookLevel() = default;
    OrderBookLevel(Price p, Quantity q) : price(p), quantity(q) {}
};

struct OrderBookSnapshot {
    Timestamp timestamp;
    std::vector<OrderBookLevel> bids;
    std::vector<OrderBookLevel> asks;
    
    Price getBestBid() const { return bids.empty() ? 0.0 : bids[0].price; }
    Price getBestAsk() const { return asks.empty() ? 0.0 : asks[0].price; }
    Price getMidPrice() const { 
        auto bid = getBestBid();
        auto ask = getBestAsk();
        return (bid > 0 && ask > 0) ? (bid + ask) / 2.0 : 0.0;
    }
};

struct Trade {
    Timestamp timestamp;
    Price price;
    Quantity quantity;
    Side side;
    
    Trade() = default;
    Trade(Timestamp ts, Price p, Quantity q, Side s) 
        : timestamp(ts), price(p), quantity(q), side(s) {}
};

struct Order {
    OrderId id;
    Timestamp timestamp;
    Side side;
    OrderType type;
    Price price;
    Quantity quantity;
    Quantity filled_quantity;
    OrderStatus status;
    
    Order() = default;
    Order(OrderId id, Timestamp ts, Side s, OrderType t, Price p, Quantity q)
        : id(id), timestamp(ts), side(s), type(t), price(p), quantity(q), 
          filled_quantity(0), status(OrderStatus::PENDING) {}
};

struct DCState {
    DCEvent current_event;
    Price highest_price;
    Price lowest_price;
    bool is_upturn;
    
    DCState() : current_event(DCEvent::NONE), highest_price(0), lowest_price(0), is_upturn(true) {}
};

struct PerformanceMetrics {
    double total_return;
    double sharpe_ratio;
    double max_drawdown;
    double win_rate;
    int total_trades;
    double avg_trade_duration_ms;
    double total_fees;
    Timestamp start_time;
    Timestamp end_time;
    
    PerformanceMetrics() : total_return(0), sharpe_ratio(0), max_drawdown(0), 
                          win_rate(0), total_trades(0), avg_trade_duration_ms(0), 
                          total_fees(0) {}
};

} // namespace dcbacktest
