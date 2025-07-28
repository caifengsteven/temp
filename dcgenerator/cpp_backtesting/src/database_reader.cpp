#include "database_reader.h"
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <chrono>
#include <iomanip>

namespace dcbacktest {

// Helper function to parse date and time strings to timestamp
Timestamp parseDateTime(const std::string& date_str, const std::string& time_str) {
    try {
        // Assuming date format: YYYY-MM-DD and time format: HH:MM:SS or HH:MM:SS.mmm
        std::tm tm = {};
        std::istringstream date_stream(date_str);
        date_stream >> std::get_time(&tm, "%Y-%m-%d");

        std::istringstream time_stream(time_str);
        int hour, minute, second, millisecond = 0;
        char colon1, colon2, dot;

        if (time_str.find('.') != std::string::npos) {
            time_stream >> hour >> colon1 >> minute >> colon2 >> second >> dot >> millisecond;
        } else {
            time_stream >> hour >> colon1 >> minute >> colon2 >> second;
        }

        tm.tm_hour = hour;
        tm.tm_min = minute;
        tm.tm_sec = second;

        auto time_point = std::chrono::system_clock::from_time_t(std::mktime(&tm));
        auto nanoseconds = std::chrono::duration_cast<std::chrono::nanoseconds>(
            time_point.time_since_epoch()) + std::chrono::milliseconds(millisecond);

        return nanoseconds;
    } catch (...) {
        // If parsing fails, return current time
        return std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::system_clock::now().time_since_epoch());
    }
}

// DatabaseReader implementation

DatabaseReader::DatabaseReader(const std::string& db_path) 
    : db_path_(db_path), db_(nullptr) {}

DatabaseReader::~DatabaseReader() {
    close();
}

bool DatabaseReader::open() {
    if (db_) {
        return true; // Already open
    }
    
    int rc = sqlite3_open(db_path_.c_str(), &db_);
    if (rc != SQLITE_OK) {
        std::cerr << "Cannot open database: " << sqlite3_errmsg(db_) << std::endl;
        sqlite3_close(db_);
        db_ = nullptr;
        return false;
    }
    
    return true;
}

void DatabaseReader::close() {
    if (db_) {
        sqlite3_close(db_);
        db_ = nullptr;
    }
}

bool DatabaseReader::executeQuery(const std::string& query, 
                                 std::function<void(sqlite3_stmt*)> row_callback) {
    if (!db_) {
        std::cerr << "Database not open" << std::endl;
        return false;
    }
    
    sqlite3_stmt* stmt;
    int rc = sqlite3_prepare_v2(db_, query.c_str(), -1, &stmt, nullptr);
    
    if (rc != SQLITE_OK) {
        std::cerr << "SQL error: " << sqlite3_errmsg(db_) << std::endl;
        return false;
    }
    
    while ((rc = sqlite3_step(stmt)) == SQLITE_ROW) {
        row_callback(stmt);
    }
    
    sqlite3_finalize(stmt);
    
    if (rc != SQLITE_DONE) {
        std::cerr << "SQL execution error: " << sqlite3_errmsg(db_) << std::endl;
        return false;
    }
    
    return true;
}

// OrderBookReader implementation

OrderBookReader::OrderBookReader(const std::string& db_path) 
    : DatabaseReader(db_path) {}

std::vector<OrderBookSnapshot> OrderBookReader::readOrderBook(Timestamp start_time, Timestamp end_time) {
    std::vector<OrderBookSnapshot> snapshots;
    
    if (!open()) {
        return snapshots;
    }
    
    // This is a generic query - you'll need to adapt it to your actual schema
    std::string query = R"(
        SELECT timestamp, bid_prices, bid_quantities, ask_prices, ask_quantities 
        FROM orderbook 
        WHERE timestamp >= ? AND timestamp <= ? 
        ORDER BY timestamp
    )";
    
    sqlite3_stmt* stmt;
    int rc = sqlite3_prepare_v2(db_, query.c_str(), -1, &stmt, nullptr);
    
    if (rc == SQLITE_OK) {
        sqlite3_bind_int64(stmt, 1, start_time.count());
        sqlite3_bind_int64(stmt, 2, end_time.count());
        
        while ((rc = sqlite3_step(stmt)) == SQLITE_ROW) {
            snapshots.push_back(parseOrderBookRow(stmt));
        }
        
        sqlite3_finalize(stmt);
    }
    
    return snapshots;
}

std::vector<OrderBookSnapshot> OrderBookReader::readOrderBookForSymbol(
    const std::string& symbol, Timestamp start_time, Timestamp end_time) {
    
    std::vector<OrderBookSnapshot> snapshots;
    
    if (!open()) {
        return snapshots;
    }
    
    std::string query = R"(
        SELECT timestamp, bid_prices, bid_quantities, ask_prices, ask_quantities 
        FROM orderbook 
        WHERE symbol = ? AND timestamp >= ? AND timestamp <= ? 
        ORDER BY timestamp
    )";
    
    sqlite3_stmt* stmt;
    int rc = sqlite3_prepare_v2(db_, query.c_str(), -1, &stmt, nullptr);
    
    if (rc == SQLITE_OK) {
        sqlite3_bind_text(stmt, 1, symbol.c_str(), -1, SQLITE_STATIC);
        sqlite3_bind_int64(stmt, 2, start_time.count());
        sqlite3_bind_int64(stmt, 3, end_time.count());
        
        while ((rc = sqlite3_step(stmt)) == SQLITE_ROW) {
            snapshots.push_back(parseOrderBookRow(stmt));
        }
        
        sqlite3_finalize(stmt);
    }
    
    return snapshots;
}

std::vector<std::string> OrderBookReader::getAvailableSymbols() {
    std::vector<std::string> symbols;
    
    if (!open()) {
        return symbols;
    }
    
    std::string query = "SELECT DISTINCT symbol FROM orderbook ORDER BY symbol";
    
    executeQuery(query, [&symbols](sqlite3_stmt* stmt) {
        const char* symbol = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 0));
        if (symbol) {
            symbols.emplace_back(symbol);
        }
    });
    
    return symbols;
}

std::pair<Timestamp, Timestamp> OrderBookReader::getTimeRange(const std::string& symbol) {
    Timestamp min_time{0}, max_time{0};
    
    if (!open()) {
        return {min_time, max_time};
    }
    
    std::string query = R"(
        SELECT MIN(timestamp), MAX(timestamp) 
        FROM orderbook 
        WHERE symbol = ?
    )";
    
    sqlite3_stmt* stmt;
    int rc = sqlite3_prepare_v2(db_, query.c_str(), -1, &stmt, nullptr);
    
    if (rc == SQLITE_OK) {
        sqlite3_bind_text(stmt, 1, symbol.c_str(), -1, SQLITE_STATIC);
        
        if (sqlite3_step(stmt) == SQLITE_ROW) {
            min_time = Timestamp(sqlite3_column_int64(stmt, 0));
            max_time = Timestamp(sqlite3_column_int64(stmt, 1));
        }
        
        sqlite3_finalize(stmt);
    }
    
    return {min_time, max_time};
}

OrderBookSnapshot OrderBookReader::parseOrderBookRow(sqlite3_stmt* stmt) {
    OrderBookSnapshot snapshot;
    
    // Parse timestamp
    snapshot.timestamp = Timestamp(sqlite3_column_int64(stmt, 0));
    
    // Parse bid and ask data - this is simplified and needs to be adapted
    // to your actual database schema
    // For now, assuming simplified format with best bid/ask only
    
    // You'll need to implement proper parsing based on your schema
    // This is just a placeholder implementation
    
    return snapshot;
}

// TradesReader implementation

TradesReader::TradesReader(const std::string& db_path)
    : DatabaseReader(db_path) {}

std::vector<Trade> TradesReader::readTrades(Timestamp start_time, Timestamp end_time) {
    std::vector<Trade> trades;

    if (!open()) {
        return trades;
    }

    // Updated query to match your actual schema
    std::string query = R"(
        SELECT date, time, price, volume, buysell
        FROM trade
        ORDER BY date, time
    )";

    sqlite3_stmt* stmt;
    int rc = sqlite3_prepare_v2(db_, query.c_str(), -1, &stmt, nullptr);

    if (rc == SQLITE_OK) {
        while ((rc = sqlite3_step(stmt)) == SQLITE_ROW) {
            trades.push_back(parseTradeRow(stmt));
        }

        sqlite3_finalize(stmt);
    }

    return trades;
}

std::vector<Trade> TradesReader::readTradesForSymbol(
    const std::string& symbol, Timestamp start_time, Timestamp end_time) {

    std::vector<Trade> trades;

    if (!open()) {
        return trades;
    }

    // Updated query to match your actual schema
    std::string query = R"(
        SELECT date, time, price, volume, buysell
        FROM trade
        WHERE symbol = ?
        ORDER BY date, time
    )";

    sqlite3_stmt* stmt;
    int rc = sqlite3_prepare_v2(db_, query.c_str(), -1, &stmt, nullptr);

    if (rc == SQLITE_OK) {
        sqlite3_bind_text(stmt, 1, symbol.c_str(), -1, SQLITE_STATIC);

        while ((rc = sqlite3_step(stmt)) == SQLITE_ROW) {
            trades.push_back(parseTradeRow(stmt));
        }

        sqlite3_finalize(stmt);
    }

    return trades;
}

std::vector<std::string> TradesReader::getAvailableSymbols() {
    std::vector<std::string> symbols;

    if (!open()) {
        return symbols;
    }

    std::string query = "SELECT DISTINCT symbol FROM trade ORDER BY symbol";

    executeQuery(query, [&symbols](sqlite3_stmt* stmt) {
        const char* symbol = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 0));
        if (symbol) {
            symbols.emplace_back(symbol);
        }
    });

    return symbols;
}

std::pair<Timestamp, Timestamp> TradesReader::getTimeRange(const std::string& symbol) {
    Timestamp min_time{0}, max_time{0};

    if (!open()) {
        return {min_time, max_time};
    }

    std::string query = R"(
        SELECT MIN(date || ' ' || time), MAX(date || ' ' || time)
        FROM trade
        WHERE symbol = ?
    )";

    sqlite3_stmt* stmt;
    int rc = sqlite3_prepare_v2(db_, query.c_str(), -1, &stmt, nullptr);

    if (rc == SQLITE_OK) {
        sqlite3_bind_text(stmt, 1, symbol.c_str(), -1, SQLITE_STATIC);

        if (sqlite3_step(stmt) == SQLITE_ROW) {
            // For now, we'll use dummy timestamps since we need to parse the date/time strings
            min_time = Timestamp(0);
            max_time = Timestamp(std::chrono::duration_cast<std::chrono::nanoseconds>(
                std::chrono::system_clock::now().time_since_epoch()));
        }

        sqlite3_finalize(stmt);
    }

    return {min_time, max_time};
}

std::vector<Tick> TradesReader::readPriceTicks(const std::string& symbol,
                                              Timestamp start_time,
                                              Timestamp end_time) {
    std::vector<Tick> ticks;
    auto trades = readTradesForSymbol(symbol, start_time, end_time);

    for (const auto& trade : trades) {
        ticks.emplace_back(trade.timestamp, trade.price, trade.quantity);
    }

    return ticks;
}

Trade TradesReader::parseTradeRow(sqlite3_stmt* stmt) {
    Trade trade;

    // Parse date and time (columns 0 and 1)
    const char* date_str = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 0));
    const char* time_str = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 1));

    if (date_str && time_str) {
        trade.timestamp = parseDateTime(std::string(date_str), std::string(time_str));
    } else {
        trade.timestamp = std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::system_clock::now().time_since_epoch());
    }

    // Parse price (column 2)
    trade.price = sqlite3_column_double(stmt, 2);

    // Parse volume/quantity (column 3)
    trade.quantity = static_cast<double>(sqlite3_column_int64(stmt, 3));

    // Parse buysell (column 4)
    const char* side_str = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 4));
    if (side_str) {
        std::string side(side_str);
        if (side == "buy" || side == "BUY" || side == "b" || side == "B") {
            trade.side = Side::BUY;
        } else {
            trade.side = Side::SELL;
        }
    } else {
        trade.side = Side::BUY; // Default
    }

    return trade;
}

} // namespace dcbacktest
