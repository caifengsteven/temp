#include <iostream>
#include <vector>
#include <string>
#include <windows.h>
#include <iomanip>
#include <algorithm>
#include <cmath>

// SQLite function pointers
typedef struct sqlite3 sqlite3;
typedef struct sqlite3_stmt sqlite3_stmt;
typedef int (*sqlite3_open_func)(const char*, sqlite3**);
typedef int (*sqlite3_close_func)(sqlite3*);
typedef int (*sqlite3_prepare_v2_func)(sqlite3*, const char*, int, sqlite3_stmt**, const char**);
typedef int (*sqlite3_step_func)(sqlite3_stmt*);
typedef int (*sqlite3_finalize_func)(sqlite3_stmt*);
typedef const unsigned char* (*sqlite3_column_text_func)(sqlite3_stmt*, int);
typedef double (*sqlite3_column_double_func)(sqlite3_stmt*, int);
typedef int (*sqlite3_bind_text_func)(sqlite3_stmt*, int, const char*, int, void(*)(void*));
typedef const char* (*sqlite3_errmsg_func)(sqlite3*);

#define SQLITE_OK           0
#define SQLITE_ROW          100

// Global function pointers
sqlite3_open_func sqlite3_open_ptr = nullptr;
sqlite3_close_func sqlite3_close_ptr = nullptr;
sqlite3_prepare_v2_func sqlite3_prepare_v2_ptr = nullptr;
sqlite3_step_func sqlite3_step_ptr = nullptr;
sqlite3_finalize_func sqlite3_finalize_ptr = nullptr;
sqlite3_column_text_func sqlite3_column_text_ptr = nullptr;
sqlite3_column_double_func sqlite3_column_double_ptr = nullptr;
sqlite3_bind_text_func sqlite3_bind_text_ptr = nullptr;
sqlite3_errmsg_func sqlite3_errmsg_ptr = nullptr;

bool loadSQLite() {
    HMODULE hModule = LoadLibraryA("C:/sqlite3/sqlite3.dll");
    if (!hModule) {
        hModule = LoadLibraryA("sqlite3.dll");
    }
    if (!hModule) {
        std::cout << "Could not load SQLite library" << std::endl;
        return false;
    }
    
    sqlite3_open_ptr = (sqlite3_open_func)GetProcAddress(hModule, "sqlite3_open");
    sqlite3_close_ptr = (sqlite3_close_func)GetProcAddress(hModule, "sqlite3_close");
    sqlite3_prepare_v2_ptr = (sqlite3_prepare_v2_func)GetProcAddress(hModule, "sqlite3_prepare_v2");
    sqlite3_step_ptr = (sqlite3_step_func)GetProcAddress(hModule, "sqlite3_step");
    sqlite3_finalize_ptr = (sqlite3_finalize_func)GetProcAddress(hModule, "sqlite3_finalize");
    sqlite3_column_text_ptr = (sqlite3_column_text_func)GetProcAddress(hModule, "sqlite3_column_text");
    sqlite3_column_double_ptr = (sqlite3_column_double_func)GetProcAddress(hModule, "sqlite3_column_double");
    sqlite3_bind_text_ptr = (sqlite3_bind_text_func)GetProcAddress(hModule, "sqlite3_bind_text");
    sqlite3_errmsg_ptr = (sqlite3_errmsg_func)GetProcAddress(hModule, "sqlite3_errmsg");
    
    return (sqlite3_open_ptr && sqlite3_close_ptr && sqlite3_prepare_v2_ptr && 
            sqlite3_step_ptr && sqlite3_finalize_ptr);
}

// Test loading CL (Crude Oil) data
void testCLData() {
    std::cout << "=== Testing CL (Crude Oil) Futures Data ===" << std::endl;
    
    std::string db_path = "F:\\database\\us futures 1mins\\us_fut_1min.db";
    std::cout << "Database: " << db_path << std::endl;
    
    sqlite3* db;
    int rc = sqlite3_open_ptr(db_path.c_str(), &db);
    
    if (rc != SQLITE_OK) {
        std::cout << "Cannot open database: " << sqlite3_errmsg_ptr(db) << std::endl;
        if (db) sqlite3_close_ptr(db);
        return;
    }
    
    std::cout << "Database opened successfully!" << std::endl;
    
    // First, check how many CL records exist
    std::string count_sql = "SELECT COUNT(*) FROM futures_data WHERE symbol = 'CL';";
    sqlite3_stmt* stmt;
    
    rc = sqlite3_prepare_v2_ptr(db, count_sql.c_str(), -1, &stmt, nullptr);
    if (rc != SQLITE_OK) {
        std::cout << "Count SQL error: " << sqlite3_errmsg_ptr(db) << std::endl;
        sqlite3_close_ptr(db);
        return;
    }
    
    if (sqlite3_step_ptr(stmt) == SQLITE_ROW) {
        const char* count_str = reinterpret_cast<const char*>(sqlite3_column_text_ptr(stmt, 0));
        if (count_str) {
            std::cout << "Total CL records: " << count_str << std::endl;
        }
    }
    sqlite3_finalize_ptr(stmt);
    
    // Get sample CL data
    std::string sample_sql = "SELECT datetime, close FROM futures_data WHERE symbol = 'CL' ORDER BY datetime LIMIT 10;";
    rc = sqlite3_prepare_v2_ptr(db, sample_sql.c_str(), -1, &stmt, nullptr);
    
    if (rc == SQLITE_OK) {
        std::cout << "\nSample CL data:" << std::endl;
        std::cout << "DateTime\t\tClose Price" << std::endl;
        std::cout << std::string(40, '-') << std::endl;
        
        while (sqlite3_step_ptr(stmt) == SQLITE_ROW) {
            const char* datetime = reinterpret_cast<const char*>(sqlite3_column_text_ptr(stmt, 0));
            double close_price = sqlite3_column_double_ptr(stmt, 1);
            
            if (datetime) {
                std::cout << datetime << "\t$" << std::fixed << std::setprecision(2) << close_price << std::endl;
            }
        }
        sqlite3_finalize_ptr(stmt);
    }
    
    // Load price data for DC analysis
    std::string data_sql = "SELECT close FROM futures_data WHERE symbol = 'CL' ORDER BY datetime LIMIT 10000;";
    rc = sqlite3_prepare_v2_ptr(db, data_sql.c_str(), -1, &stmt, nullptr);
    
    if (rc != SQLITE_OK) {
        std::cout << "Data SQL error: " << sqlite3_errmsg_ptr(db) << std::endl;
        sqlite3_close_ptr(db);
        return;
    }
    
    std::vector<double> prices;
    int count = 0;
    
    std::cout << "\nLoading CL price data..." << std::endl;
    
    while ((rc = sqlite3_step_ptr(stmt)) == SQLITE_ROW) {
        double price = sqlite3_column_double_ptr(stmt, 0);
        if (price > 0 && std::isfinite(price)) {
            prices.push_back(price);
            count++;
            
            if (count % 1000 == 0) {
                std::cout << "  Loaded " << count << " points..." << std::endl;
            }
        }
    }
    sqlite3_finalize_ptr(stmt);
    sqlite3_close_ptr(db);
    
    std::cout << "Total CL prices loaded: " << prices.size() << std::endl;
    
    if (prices.size() > 0) {
        double min_price = *std::min_element(prices.begin(), prices.end());
        double max_price = *std::max_element(prices.begin(), prices.end());
        double price_range_pct = ((max_price - min_price) / min_price) * 100.0;
        
        std::cout << "\nCL Price Analysis:" << std::endl;
        std::cout << "  Price range: $" << std::fixed << std::setprecision(2) << min_price << " to $" << max_price << std::endl;
        std::cout << "  Price range %: " << std::setprecision(1) << price_range_pct << "%" << std::endl;
        std::cout << "  First price: $" << std::setprecision(2) << prices.front() << std::endl;
        std::cout << "  Last price: $" << prices.back() << std::endl;
        
        if (prices.size() >= 1000) {
            std::cout << "\n✅ SUCCESS: CL data loaded successfully!" << std::endl;
            std::cout << "The futures database is working correctly." << std::endl;
        } else {
            std::cout << "\n⚠️  WARNING: Limited CL data (" << prices.size() << " points)" << std::endl;
        }
    } else {
        std::cout << "\n❌ ERROR: No CL price data found!" << std::endl;
    }
}

int main() {
    std::cout << "=== Single Futures Symbol Test ===" << std::endl;
    
    if (!loadSQLite()) {
        std::cout << "Failed to load SQLite library" << std::endl;
        return 1;
    }
    
    testCLData();
    
    return 0;
}
