#pragma once

#include "types.h"
#include <sqlite3.h>
#include <string>
#include <vector>
#include <memory>
#include <functional>

namespace dcbacktest {

class DatabaseReader {
public:
    explicit DatabaseReader(const std::string& db_path);
    ~DatabaseReader();

    // Open database connection
    bool open();
    
    // Close database connection
    void close();
    
    // Check if database is open
    bool isOpen() const { return db_ != nullptr; }

protected:
    std::string db_path_;
    sqlite3* db_;
    
    // Helper method to execute SQL queries
    bool executeQuery(const std::string& query, 
                     std::function<void(sqlite3_stmt*)> row_callback);
};

class OrderBookReader : public DatabaseReader {
public:
    explicit OrderBookReader(const std::string& db_path);
    
    // Read orderbook snapshots within time range
    std::vector<OrderBookSnapshot> readOrderBook(Timestamp start_time, Timestamp end_time);
    
    // Read orderbook snapshots for a specific symbol
    std::vector<OrderBookSnapshot> readOrderBookForSymbol(const std::string& symbol,
                                                          Timestamp start_time, 
                                                          Timestamp end_time);
    
    // Get available symbols
    std::vector<std::string> getAvailableSymbols();
    
    // Get time range for a symbol
    std::pair<Timestamp, Timestamp> getTimeRange(const std::string& symbol);

private:
    OrderBookSnapshot parseOrderBookRow(sqlite3_stmt* stmt);
};

class TradesReader : public DatabaseReader {
public:
    explicit TradesReader(const std::string& db_path);
    
    // Read trades within time range
    std::vector<Trade> readTrades(Timestamp start_time, Timestamp end_time);
    
    // Read trades for a specific symbol
    std::vector<Trade> readTradesForSymbol(const std::string& symbol,
                                          Timestamp start_time, 
                                          Timestamp end_time);
    
    // Get available symbols
    std::vector<std::string> getAvailableSymbols();
    
    // Get time range for a symbol
    std::pair<Timestamp, Timestamp> getTimeRange(const std::string& symbol);
    
    // Read trades and convert to price ticks
    std::vector<Tick> readPriceTicks(const std::string& symbol,
                                    Timestamp start_time, 
                                    Timestamp end_time);

private:
    Trade parseTradeRow(sqlite3_stmt* stmt);
};

} // namespace dcbacktest
