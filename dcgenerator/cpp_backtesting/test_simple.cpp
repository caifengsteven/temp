#include <iostream>
#include <sqlite3.h>
#include <string>
#include <vector>

struct SimpleTradeData {
    std::string date;
    std::string time;
    std::string symbol;
    double price;
    int volume;
    std::string buysell;
};

int main() {
    const std::string db_path = "I:/zhubi/cpp_implementation/sqlite_databases/2018/2018_01.db";
    
    std::cout << "Testing database connection and data reading..." << std::endl;
    std::cout << "Database: " << db_path << std::endl;
    
    sqlite3* db;
    int rc = sqlite3_open(db_path.c_str(), &db);
    
    if (rc != SQLITE_OK) {
        std::cerr << "Cannot open database: " << sqlite3_errmsg(db) << std::endl;
        sqlite3_close(db);
        return 1;
    }
    
    std::cout << "Database opened successfully!" << std::endl;
    
    // Get available symbols
    const char* symbols_query = "SELECT DISTINCT symbol FROM trade LIMIT 10;";
    sqlite3_stmt* stmt;
    
    rc = sqlite3_prepare_v2(db, symbols_query, -1, &stmt, nullptr);
    if (rc != SQLITE_OK) {
        std::cerr << "Failed to prepare statement: " << sqlite3_errmsg(db) << std::endl;
        sqlite3_close(db);
        return 1;
    }
    
    std::cout << "\nAvailable symbols:" << std::endl;
    std::vector<std::string> symbols;
    while ((rc = sqlite3_step(stmt)) == SQLITE_ROW) {
        const char* symbol = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 0));
        if (symbol) {
            symbols.push_back(std::string(symbol));
            std::cout << "  " << symbol << std::endl;
        }
    }
    sqlite3_finalize(stmt);
    
    if (symbols.empty()) {
        std::cout << "No symbols found!" << std::endl;
        sqlite3_close(db);
        return 1;
    }
    
    // Get sample data for the first symbol
    std::string test_symbol = symbols[0];
    std::cout << "\nSample data for symbol: " << test_symbol << std::endl;
    
    std::string sample_query = "SELECT date, time, price, volume, buysell FROM trade WHERE symbol = ? LIMIT 10;";
    rc = sqlite3_prepare_v2(db, sample_query.c_str(), -1, &stmt, nullptr);
    
    if (rc != SQLITE_OK) {
        std::cerr << "Failed to prepare sample query: " << sqlite3_errmsg(db) << std::endl;
        sqlite3_close(db);
        return 1;
    }
    
    sqlite3_bind_text(stmt, 1, test_symbol.c_str(), -1, SQLITE_STATIC);
    
    std::cout << "Date       | Time     | Price    | Volume | Side" << std::endl;
    std::cout << "-----------|----------|----------|--------|-----" << std::endl;
    
    std::vector<SimpleTradeData> trades;
    while ((rc = sqlite3_step(stmt)) == SQLITE_ROW) {
        SimpleTradeData trade;
        
        const char* date = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 0));
        const char* time = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 1));
        
        trade.date = date ? std::string(date) : "";
        trade.time = time ? std::string(time) : "";
        trade.price = sqlite3_column_double(stmt, 2);
        trade.volume = sqlite3_column_int(stmt, 3);
        
        const char* buysell = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 4));
        trade.buysell = buysell ? std::string(buysell) : "";
        
        trades.push_back(trade);
        
        printf("%-10s | %-8s | %8.2f | %6d | %s\n", 
               trade.date.c_str(), trade.time.c_str(), trade.price, trade.volume, trade.buysell.c_str());
    }
    
    sqlite3_finalize(stmt);
    
    // Get total count for this symbol
    std::string count_query = "SELECT COUNT(*) FROM trade WHERE symbol = ?;";
    rc = sqlite3_prepare_v2(db, count_query.c_str(), -1, &stmt, nullptr);
    
    if (rc == SQLITE_OK) {
        sqlite3_bind_text(stmt, 1, test_symbol.c_str(), -1, SQLITE_STATIC);
        if (sqlite3_step(stmt) == SQLITE_ROW) {
            int total_count = sqlite3_column_int(stmt, 0);
            std::cout << "\nTotal trades for " << test_symbol << ": " << total_count << std::endl;
        }
        sqlite3_finalize(stmt);
    }
    
    // Test DC Generator on this data
    std::cout << "\nTesting DC Generator with threshold 0.1%..." << std::endl;
    
    if (!trades.empty()) {
        double threshold = 0.001; // 0.1%
        double highest_price = trades[0].price;
        double lowest_price = trades[0].price;
        bool is_upturn = true;
        int dc_events = 0;
        
        for (size_t i = 1; i < trades.size(); ++i) {
            double current_price = trades[i].price;
            
            if (is_upturn) {
                if (current_price <= highest_price * (1.0 - threshold)) {
                    // End of upturn
                    std::cout << "DC Event: End upturn at " << current_price 
                              << " (from peak " << highest_price << ")" << std::endl;
                    is_upturn = false;
                    lowest_price = current_price;
                    dc_events++;
                } else if (current_price > highest_price) {
                    highest_price = current_price;
                }
            } else {
                if (current_price >= lowest_price * (1.0 + threshold)) {
                    // End of downturn
                    std::cout << "DC Event: End downturn at " << current_price 
                              << " (from trough " << lowest_price << ")" << std::endl;
                    is_upturn = true;
                    highest_price = current_price;
                    dc_events++;
                } else if (current_price < lowest_price) {
                    lowest_price = current_price;
                }
            }
        }
        
        std::cout << "Total DC events detected: " << dc_events << std::endl;
    }
    
    sqlite3_close(db);
    std::cout << "\nTest completed successfully!" << std::endl;
    
    return 0;
}
