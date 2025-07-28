#pragma once

#include "types.h"
#include <unordered_map>
#include <queue>
#include <memory>

namespace dcbacktest {

class OrderManager {
public:
    OrderManager();
    ~OrderManager() = default;

    // Submit a new order
    OrderId submitOrder(Side side, OrderType type, Price price, Quantity quantity, Timestamp timestamp);
    
    // Cancel an order
    bool cancelOrder(OrderId order_id, Timestamp timestamp);
    
    // Process market data update (for order matching)
    void processMarketUpdate(const OrderBookSnapshot& orderbook, Timestamp timestamp);
    
    // Process trade update (for order matching)
    void processTrade(const Trade& trade, Timestamp timestamp);
    
    // Get order by ID
    const Order* getOrder(OrderId order_id) const;
    
    // Get all orders
    const std::unordered_map<OrderId, Order>& getAllOrders() const { return orders_; }
    
    // Get pending orders
    std::vector<OrderId> getPendingOrders() const;
    
    // Get filled orders
    std::vector<OrderId> getFilledOrders() const;
    
    // Clear all orders
    void clear();
    
    // Set trading fees (in basis points)
    void setTradingFees(double maker_fee_bps, double taker_fee_bps);
    
    // Get total fees paid
    double getTotalFees() const { return total_fees_; }

private:
    std::unordered_map<OrderId, Order> orders_;
    OrderId next_order_id_;
    double maker_fee_bps_;
    double taker_fee_bps_;
    double total_fees_;
    
    // Latency simulation
    std::chrono::nanoseconds order_latency_;
    std::chrono::nanoseconds cancel_latency_;
    
    // Helper methods
    void fillOrder(OrderId order_id, Price fill_price, Quantity fill_quantity, Timestamp timestamp);
    double calculateFee(Price price, Quantity quantity, bool is_maker) const;
    bool canFillOrder(const Order& order, const OrderBookSnapshot& orderbook) const;
    std::pair<Price, Quantity> getExecutionDetails(const Order& order, const OrderBookSnapshot& orderbook) const;
};

} // namespace dcbacktest
