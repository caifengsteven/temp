#include "order_manager.h"
#include <algorithm>
#include <iostream>

namespace dcbacktest {

OrderManager::OrderManager() 
    : next_order_id_(1)
    , maker_fee_bps_(1.0)
    , taker_fee_bps_(2.0)
    , total_fees_(0.0)
    , order_latency_(std::chrono::microseconds(100))
    , cancel_latency_(std::chrono::microseconds(50)) {}

OrderId OrderManager::submitOrder(Side side, OrderType type, Price price, 
                                 Quantity quantity, Timestamp timestamp) {
    OrderId order_id = next_order_id_++;
    
    Order order(order_id, timestamp, side, type, price, quantity);
    orders_[order_id] = order;
    
    return order_id;
}

bool OrderManager::cancelOrder(OrderId order_id, Timestamp timestamp) {
    auto it = orders_.find(order_id);
    if (it == orders_.end()) {
        return false;
    }
    
    Order& order = it->second;
    if (order.status != OrderStatus::PENDING && order.status != OrderStatus::PARTIALLY_FILLED) {
        return false;
    }
    
    order.status = OrderStatus::CANCELLED;
    return true;
}

void OrderManager::processMarketUpdate(const OrderBookSnapshot& orderbook, Timestamp timestamp) {
    // Check if any pending orders can be filled
    for (auto& [order_id, order] : orders_) {
        if (order.status == OrderStatus::PENDING || order.status == OrderStatus::PARTIALLY_FILLED) {
            if (canFillOrder(order, orderbook)) {
                auto [fill_price, fill_quantity] = getExecutionDetails(order, orderbook);
                fillOrder(order_id, fill_price, fill_quantity, timestamp);
            }
        }
    }
}

void OrderManager::processTrade(const Trade& trade, Timestamp timestamp) {
    // For market orders, we can use trade data for immediate execution
    for (auto& [order_id, order] : orders_) {
        if (order.status == OrderStatus::PENDING && order.type == OrderType::MARKET) {
            // Market orders execute at the trade price
            Quantity fill_quantity = std::min(order.quantity - order.filled_quantity, trade.quantity);
            fillOrder(order_id, trade.price, fill_quantity, timestamp);
        }
    }
}

const Order* OrderManager::getOrder(OrderId order_id) const {
    auto it = orders_.find(order_id);
    return (it != orders_.end()) ? &it->second : nullptr;
}

std::vector<OrderId> OrderManager::getPendingOrders() const {
    std::vector<OrderId> pending;
    for (const auto& [order_id, order] : orders_) {
        if (order.status == OrderStatus::PENDING || order.status == OrderStatus::PARTIALLY_FILLED) {
            pending.push_back(order_id);
        }
    }
    return pending;
}

std::vector<OrderId> OrderManager::getFilledOrders() const {
    std::vector<OrderId> filled;
    for (const auto& [order_id, order] : orders_) {
        if (order.status == OrderStatus::FILLED) {
            filled.push_back(order_id);
        }
    }
    return filled;
}

void OrderManager::clear() {
    orders_.clear();
    next_order_id_ = 1;
    total_fees_ = 0.0;
}

void OrderManager::setTradingFees(double maker_fee_bps, double taker_fee_bps) {
    maker_fee_bps_ = maker_fee_bps;
    taker_fee_bps_ = taker_fee_bps;
}

void OrderManager::fillOrder(OrderId order_id, Price fill_price, Quantity fill_quantity, Timestamp timestamp) {
    auto it = orders_.find(order_id);
    if (it == orders_.end()) {
        return;
    }
    
    Order& order = it->second;
    
    // Update filled quantity
    order.filled_quantity += fill_quantity;
    
    // Calculate fees (assuming taker for simplicity)
    double fee = calculateFee(fill_price, fill_quantity, false);
    total_fees_ += fee;
    
    // Update order status
    if (order.filled_quantity >= order.quantity) {
        order.status = OrderStatus::FILLED;
    } else {
        order.status = OrderStatus::PARTIALLY_FILLED;
    }
    
    // Log the fill (optional)
    // std::cout << "Order " << order_id << " filled: " << fill_quantity 
    //           << " @ " << fill_price << " (fee: " << fee << ")" << std::endl;
}

double OrderManager::calculateFee(Price price, Quantity quantity, bool is_maker) const {
    double notional = price * quantity;
    double fee_rate = is_maker ? maker_fee_bps_ : taker_fee_bps_;
    return notional * fee_rate / 10000.0; // Convert basis points to decimal
}

bool OrderManager::canFillOrder(const Order& order, const OrderBookSnapshot& orderbook) const {
    if (order.type == OrderType::MARKET) {
        return true; // Market orders can always be filled (assuming liquidity)
    }
    
    // For limit orders, check if price crosses the spread
    if (order.side == Side::BUY) {
        Price best_ask = orderbook.getBestAsk();
        return best_ask > 0 && order.price >= best_ask;
    } else {
        Price best_bid = orderbook.getBestBid();
        return best_bid > 0 && order.price <= best_bid;
    }
}

std::pair<Price, Quantity> OrderManager::getExecutionDetails(const Order& order, 
                                                            const OrderBookSnapshot& orderbook) const {
    Price fill_price;
    Quantity remaining_quantity = order.quantity - order.filled_quantity;
    
    if (order.type == OrderType::MARKET) {
        // Market order executes at best available price
        fill_price = (order.side == Side::BUY) ? orderbook.getBestAsk() : orderbook.getBestBid();
    } else {
        // Limit order executes at limit price or better
        if (order.side == Side::BUY) {
            fill_price = std::min(order.price, orderbook.getBestAsk());
        } else {
            fill_price = std::max(order.price, orderbook.getBestBid());
        }
    }
    
    // For simplicity, assume we can fill the entire remaining quantity
    // In reality, you'd need to walk the order book
    Quantity fill_quantity = remaining_quantity;
    
    return {fill_price, fill_quantity};
}

} // namespace dcbacktest
