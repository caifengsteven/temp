#pragma once

#include "types.h"
#include <vector>
#include <memory>
#include <algorithm>

namespace dcbacktest {

class MarketDataProcessor {
public:
    MarketDataProcessor() = default;
    ~MarketDataProcessor() = default;

    // Merge orderbook and trades data chronologically
    struct MarketEvent {
        enum Type { ORDERBOOK_UPDATE, TRADE };
        
        Timestamp timestamp;
        Type type;
        std::shared_ptr<OrderBookSnapshot> orderbook;
        std::shared_ptr<Trade> trade;
        
        MarketEvent(Timestamp ts, std::shared_ptr<OrderBookSnapshot> ob)
            : timestamp(ts), type(ORDERBOOK_UPDATE), orderbook(ob) {}
            
        MarketEvent(Timestamp ts, std::shared_ptr<Trade> t)
            : timestamp(ts), type(TRADE), trade(t) {}
    };
    
    // Merge and sort market data events
    std::vector<MarketEvent> mergeMarketData(
        const std::vector<OrderBookSnapshot>& orderbook_data,
        const std::vector<Trade>& trades_data);
    
    // Extract price series from market data
    std::vector<Tick> extractPriceSeries(const std::vector<MarketEvent>& events);
    
    // Resample data to specific intervals
    std::vector<Tick> resampleToInterval(const std::vector<Tick>& ticks, 
                                        std::chrono::nanoseconds interval);
    
    // Calculate VWAP over a time window
    double calculateVWAP(const std::vector<Trade>& trades, 
                        Timestamp start_time, 
                        Timestamp end_time);
    
    // Calculate bid-ask spread statistics
    struct SpreadStats {
        double mean_spread;
        double median_spread;
        double std_spread;
        double min_spread;
        double max_spread;
    };
    
    SpreadStats calculateSpreadStats(const std::vector<OrderBookSnapshot>& orderbook_data);

private:
    // Helper methods
    bool isValidOrderBook(const OrderBookSnapshot& orderbook) const;
    bool isValidTrade(const Trade& trade) const;
};

class DataQualityChecker {
public:
    struct QualityReport {
        size_t total_orderbook_updates;
        size_t total_trades;
        size_t invalid_orderbook_updates;
        size_t invalid_trades;
        size_t missing_timestamps;
        size_t out_of_order_timestamps;
        double data_quality_score; // 0-1, where 1 is perfect
        
        QualityReport() : total_orderbook_updates(0), total_trades(0), 
                         invalid_orderbook_updates(0), invalid_trades(0),
                         missing_timestamps(0), out_of_order_timestamps(0),
                         data_quality_score(0.0) {}
    };
    
    // Check data quality
    QualityReport checkDataQuality(const std::vector<OrderBookSnapshot>& orderbook_data,
                                  const std::vector<Trade>& trades_data);
    
    // Print quality report
    void printQualityReport(const QualityReport& report);

private:
    bool isValidPrice(Price price) const;
    bool isValidQuantity(Quantity quantity) const;
    bool isValidTimestamp(Timestamp timestamp) const;
};

} // namespace dcbacktest
