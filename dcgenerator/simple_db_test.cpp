#include <iostream>
#include <windows.h>

// SQLite function pointers
typedef struct sqlite3 sqlite3;
typedef struct sqlite3_stmt sqlite3_stmt;
typedef int (*sqlite3_open_func)(const char*, sqlite3**);
typedef int (*sqlite3_close_func)(sqlite3*);
typedef int (*sqlite3_prepare_v2_func)(sqlite3*, const char*, int, sqlite3_stmt**, const char**);
typedef int (*sqlite3_step_func)(sqlite3_stmt*);
typedef int (*sqlite3_finalize_func)(sqlite3_stmt*);
typedef const unsigned char* (*sqlite3_column_text_func)(sqlite3_stmt*, int);
typedef int (*sqlite3_column_count_func)(sqlite3_stmt*);
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
sqlite3_column_count_func sqlite3_column_count_ptr = nullptr;
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
    sqlite3_column_count_ptr = (sqlite3_column_count_func)GetProcAddress(hModule, "sqlite3_column_count");
    sqlite3_errmsg_ptr = (sqlite3_errmsg_func)GetProcAddress(hModule, "sqlite3_errmsg");
    
    return (sqlite3_open_ptr && sqlite3_close_ptr && sqlite3_prepare_v2_ptr && 
            sqlite3_step_ptr && sqlite3_finalize_ptr);
}

void testDatabase(const std::string& db_path) {
    std::cout << "\n=== Testing: " << db_path << " ===" << std::endl;
    
    sqlite3* db;
    int rc = sqlite3_open_ptr(db_path.c_str(), &db);
    
    if (rc != SQLITE_OK) {
        std::cout << "Cannot open database: " << sqlite3_errmsg_ptr(db) << std::endl;
        if (db) sqlite3_close_ptr(db);
        return;
    }
    
    std::cout << "Database opened successfully!" << std::endl;
    
    // Get first table
    const char* tables_sql = "SELECT name FROM sqlite_master WHERE type='table' LIMIT 1;";
    sqlite3_stmt* stmt;
    
    rc = sqlite3_prepare_v2_ptr(db, tables_sql, -1, &stmt, nullptr);
    if (rc != SQLITE_OK) {
        std::cout << "SQL error: " << sqlite3_errmsg_ptr(db) << std::endl;
        sqlite3_close_ptr(db);
        return;
    }
    
    std::string first_table;
    if (sqlite3_step_ptr(stmt) == SQLITE_ROW) {
        const char* table_name = reinterpret_cast<const char*>(sqlite3_column_text_ptr(stmt, 0));
        if (table_name) {
            first_table = std::string(table_name);
            std::cout << "First table: " << first_table << std::endl;
        }
    }
    sqlite3_finalize_ptr(stmt);
    
    if (first_table.empty()) {
        std::cout << "No tables found!" << std::endl;
        sqlite3_close_ptr(db);
        return;
    }
    
    // Get sample data from first table
    std::string sample_sql = "SELECT * FROM " + first_table + " LIMIT 5;";
    rc = sqlite3_prepare_v2_ptr(db, sample_sql.c_str(), -1, &stmt, nullptr);
    
    if (rc == SQLITE_OK) {
        std::cout << "Sample data from " << first_table << ":" << std::endl;
        
        int row_count = 0;
        while ((rc = sqlite3_step_ptr(stmt)) == SQLITE_ROW && row_count < 5) {
            std::cout << "  Row " << (row_count + 1) << ": ";
            
            int col_count = sqlite3_column_count_ptr(stmt);
            for (int col = 0; col < col_count; ++col) {
                const char* value = reinterpret_cast<const char*>(sqlite3_column_text_ptr(stmt, col));
                std::cout << (value ? value : "NULL");
                if (col < col_count - 1) std::cout << " | ";
            }
            std::cout << std::endl;
            row_count++;
        }
        sqlite3_finalize_ptr(stmt);
    } else {
        std::cout << "Failed to query sample data: " << sqlite3_errmsg_ptr(db) << std::endl;
    }
    
    // Check for AAPL specifically
    std::string aapl_sql = "SELECT COUNT(*) FROM " + first_table + " WHERE symbol = 'AAPL';";
    rc = sqlite3_prepare_v2_ptr(db, aapl_sql.c_str(), -1, &stmt, nullptr);
    
    if (rc == SQLITE_OK) {
        if (sqlite3_step_ptr(stmt) == SQLITE_ROW) {
            const char* count_str = reinterpret_cast<const char*>(sqlite3_column_text_ptr(stmt, 0));
            std::cout << "AAPL records in " << first_table << ": " << (count_str ? count_str : "0") << std::endl;
        }
        sqlite3_finalize_ptr(stmt);
    } else {
        std::cout << "Failed to check AAPL: " << sqlite3_errmsg_ptr(db) << std::endl;
    }
    
    sqlite3_close_ptr(db);
}

int main() {
    std::cout << "=== Simple Database Test ===" << std::endl;
    
    if (!loadSQLite()) {
        std::cout << "Failed to load SQLite library" << std::endl;
        return 1;
    }
    
    // Test both databases
    testDatabase("F:\\BaiduNetdiskDownload\\US stock ane etf 1mins\\US_ETF_1min.db");
    testDatabase("F:\\BaiduNetdiskDownload\\US stock ane etf 1mins\\US_stock_1min.db");
    
    return 0;
}
