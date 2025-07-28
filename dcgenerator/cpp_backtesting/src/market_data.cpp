#include "market_data.h"
#include <algorithm>
#include <numeric>
#include <iostream>

namespace dcbacktest {

std::vector<MarketDataProcessor::MarketEvent> 
MarketDataProcessor::mergeMarketData(const std::vector<OrderBookSnapshot>& orderbook_data,
                                    const std::vector<Trade>& trades_data) {
    std::vector<MarketEvent> events;
    
    // Convert orderbook snapshots to events
    for (const auto& snapshot : orderbook_data) {
        if (isValidOrderBook(snapshot)) {
            events.emplace_back(snapshot.timestamp, 
                               std::make_shared<OrderBookSnapshot>(snapshot));
        }
    }
    
    // Convert trades to events
    for (const auto& trade : trades_data) {
        if (isValidTrade(trade)) {
            events.emplace_back(trade.timestamp, 
                               std::make_shared<Trade>(trade));
        }
    }
    
    // Sort by timestamp
    std::sort(events.begin(), events.end(), 
              [](const MarketEvent& a, const MarketEvent& b) {
                  return a.timestamp < b.timestamp;
              });
    
    return events;
}

std::vector<Tick> MarketDataProcessor::extractPriceSeries(const std::vector<MarketEvent>& events) {
    std::vector<Tick> ticks;
    
    for (const auto& event : events) {
        if (event.type == MarketEvent::TRADE && event.trade) {
            ticks.emplace_back(event.timestamp, event.trade->price, event.trade->quantity);
        } else if (event.type == MarketEvent::ORDERBOOK_UPDATE && event.orderbook) {
            Price mid_price = event.orderbook->getMidPrice();
            if (mid_price > 0) {
                ticks.emplace_back(event.timestamp, mid_price, 0.0); // No volume for mid-price
            }
        }
    }
    
    return ticks;
}

std::vector<Tick> MarketDataProcessor::resampleToInterval(const std::vector<Tick>& ticks, 
                                                         std::chrono::nanoseconds interval) {
    if (ticks.empty()) {
        return {};
    }
    
    std::vector<Tick> resampled;
    
    Timestamp current_bucket = ticks[0].timestamp;
    Timestamp next_bucket = current_bucket + interval;
    
    std::vector<Tick> bucket_ticks;
    
    for (const auto& tick : ticks) {
        if (tick.timestamp < next_bucket) {
            bucket_ticks.push_back(tick);
        } else {
            // Process current bucket
            if (!bucket_ticks.empty()) {
                // Use OHLC logic - for simplicity, use last price and total volume
                Price last_price = bucket_ticks.back().price;
                Quantity total_volume = std::accumulate(bucket_ticks.begin(), bucket_ticks.end(), 0.0,
                                                       [](Quantity sum, const Tick& t) {
                                                           return sum + t.volume;
                                                       });
                resampled.emplace_back(current_bucket, last_price, total_volume);
            }
            
            // Start new bucket
            current_bucket = next_bucket;
            next_bucket = current_bucket + interval;
            bucket_ticks.clear();
            bucket_ticks.push_back(tick);
        }
    }
    
    // Process final bucket
    if (!bucket_ticks.empty()) {
        Price last_price = bucket_ticks.back().price;
        Quantity total_volume = std::accumulate(bucket_ticks.begin(), bucket_ticks.end(), 0.0,
                                               [](Quantity sum, const Tick& t) {
                                                   return sum + t.volume;
                                               });
        resampled.emplace_back(current_bucket, last_price, total_volume);
    }
    
    return resampled;
}

double MarketDataProcessor::calculateVWAP(const std::vector<Trade>& trades, 
                                         Timestamp start_time, 
                                         Timestamp end_time) {
    double total_value = 0.0;
    double total_volume = 0.0;
    
    for (const auto& trade : trades) {
        if (trade.timestamp >= start_time && trade.timestamp <= end_time) {
            total_value += trade.price * trade.quantity;
            total_volume += trade.quantity;
        }
    }
    
    return (total_volume > 0) ? total_value / total_volume : 0.0;
}

MarketDataProcessor::SpreadStats 
MarketDataProcessor::calculateSpreadStats(const std::vector<OrderBookSnapshot>& orderbook_data) {
    SpreadStats stats;
    std::vector<double> spreads;
    
    for (const auto& snapshot : orderbook_data) {
        if (isValidOrderBook(snapshot)) {
            Price bid = snapshot.getBestBid();
            Price ask = snapshot.getBestAsk();
            
            if (bid > 0 && ask > 0 && ask > bid) {
                double spread = ask - bid;
                spreads.push_back(spread);
            }
        }
    }
    
    if (spreads.empty()) {
        return stats;
    }
    
    // Calculate statistics
    std::sort(spreads.begin(), spreads.end());
    
    stats.mean_spread = std::accumulate(spreads.begin(), spreads.end(), 0.0) / spreads.size();
    stats.median_spread = spreads[spreads.size() / 2];
    stats.min_spread = spreads.front();
    stats.max_spread = spreads.back();
    
    // Calculate standard deviation
    double variance = 0.0;
    for (double spread : spreads) {
        variance += (spread - stats.mean_spread) * (spread - stats.mean_spread);
    }
    variance /= spreads.size();
    stats.std_spread = std::sqrt(variance);
    
    return stats;
}

bool MarketDataProcessor::isValidOrderBook(const OrderBookSnapshot& orderbook) const {
    return orderbook.timestamp.count() > 0 && 
           orderbook.getBestBid() > 0 && 
           orderbook.getBestAsk() > 0 &&
           orderbook.getBestAsk() > orderbook.getBestBid();
}

bool MarketDataProcessor::isValidTrade(const Trade& trade) const {
    return trade.timestamp.count() > 0 && 
           trade.price > 0 && 
           trade.quantity > 0;
}

// DataQualityChecker implementation

DataQualityChecker::QualityReport 
DataQualityChecker::checkDataQuality(const std::vector<OrderBookSnapshot>& orderbook_data,
                                     const std::vector<Trade>& trades_data) {
    QualityReport report;
    
    report.total_orderbook_updates = orderbook_data.size();
    report.total_trades = trades_data.size();
    
    // Check orderbook data quality
    Timestamp last_ob_timestamp{0};
    for (const auto& snapshot : orderbook_data) {
        if (!isValidTimestamp(snapshot.timestamp)) {
            report.missing_timestamps++;
        } else if (snapshot.timestamp < last_ob_timestamp) {
            report.out_of_order_timestamps++;
        }
        
        if (!isValidPrice(snapshot.getBestBid()) || 
            !isValidPrice(snapshot.getBestAsk()) ||
            snapshot.getBestAsk() <= snapshot.getBestBid()) {
            report.invalid_orderbook_updates++;
        }
        
        last_ob_timestamp = snapshot.timestamp;
    }
    
    // Check trades data quality
    Timestamp last_trade_timestamp{0};
    for (const auto& trade : trades_data) {
        if (!isValidTimestamp(trade.timestamp)) {
            report.missing_timestamps++;
        } else if (trade.timestamp < last_trade_timestamp) {
            report.out_of_order_timestamps++;
        }
        
        if (!isValidPrice(trade.price) || !isValidQuantity(trade.quantity)) {
            report.invalid_trades++;
        }
        
        last_trade_timestamp = trade.timestamp;
    }
    
    // Calculate quality score
    size_t total_records = report.total_orderbook_updates + report.total_trades;
    size_t total_issues = report.invalid_orderbook_updates + report.invalid_trades + 
                         report.missing_timestamps + report.out_of_order_timestamps;
    
    if (total_records > 0) {
        report.data_quality_score = 1.0 - (static_cast<double>(total_issues) / total_records);
    }
    
    return report;
}

void DataQualityChecker::printQualityReport(const QualityReport& report) {
    std::cout << "\n=== DATA QUALITY REPORT ===" << std::endl;
    std::cout << "Total Orderbook Updates: " << report.total_orderbook_updates << std::endl;
    std::cout << "Total Trades: " << report.total_trades << std::endl;
    std::cout << "Invalid Orderbook Updates: " << report.invalid_orderbook_updates << std::endl;
    std::cout << "Invalid Trades: " << report.invalid_trades << std::endl;
    std::cout << "Missing Timestamps: " << report.missing_timestamps << std::endl;
    std::cout << "Out-of-Order Timestamps: " << report.out_of_order_timestamps << std::endl;
    std::cout << "Data Quality Score: " << std::fixed << std::setprecision(2) 
              << report.data_quality_score * 100 << "%" << std::endl;
    std::cout << "===========================" << std::endl;
}

bool DataQualityChecker::isValidPrice(Price price) const {
    return price > 0.0 && std::isfinite(price);
}

bool DataQualityChecker::isValidQuantity(Quantity quantity) const {
    return quantity > 0.0 && std::isfinite(quantity);
}

bool DataQualityChecker::isValidTimestamp(Timestamp timestamp) const {
    return timestamp.count() > 0;
}

} // namespace dcbacktest
