#include <iostream>
#include <vector>
#include <string>
#include <windows.h>
#include <iomanip>

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

void diagnoseFuturesDatabase() {
    std::cout << "=== US Futures Database Diagnostic Tool ===" << std::endl;
    
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
    
    // Get ALL tables
    const char* tables_sql = "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name;";
    sqlite3_stmt* stmt;
    
    rc = sqlite3_prepare_v2_ptr(db, tables_sql, -1, &stmt, nullptr);
    if (rc != SQLITE_OK) {
        std::cout << "SQL error: " << sqlite3_errmsg_ptr(db) << std::endl;
        sqlite3_close_ptr(db);
        return;
    }
    
    std::cout << "\nALL tables found:" << std::endl;
    std::vector<std::string> tables;
    while ((rc = sqlite3_step_ptr(stmt)) == SQLITE_ROW) {
        const char* table_name = reinterpret_cast<const char*>(sqlite3_column_text_ptr(stmt, 0));
        if (table_name) {
            tables.push_back(std::string(table_name));
            std::cout << "  - " << table_name << std::endl;
        }
    }
    sqlite3_finalize_ptr(stmt);
    
    std::cout << "Total tables: " << tables.size() << std::endl;
    
    // Explore first few tables in detail
    for (size_t i = 0; i < std::min(tables.size(), size_t(3)); ++i) {
        std::string table_name = tables[i];
        std::cout << "\n" << std::string(50, '=') << std::endl;
        std::cout << "=== DETAILED ANALYSIS: " << table_name << " ===" << std::endl;
        std::cout << std::string(50, '=') << std::endl;
        
        // Get column info
        std::string pragma_sql = "PRAGMA table_info(\"" + table_name + "\");";
        rc = sqlite3_prepare_v2_ptr(db, pragma_sql.c_str(), -1, &stmt, nullptr);
        
        if (rc == SQLITE_OK) {
            std::cout << "\nCOLUMNS:" << std::endl;
            std::vector<std::string> columns;
            while ((rc = sqlite3_step_ptr(stmt)) == SQLITE_ROW) {
                const char* col_name = reinterpret_cast<const char*>(sqlite3_column_text_ptr(stmt, 1));
                const char* col_type = reinterpret_cast<const char*>(sqlite3_column_text_ptr(stmt, 2));
                if (col_name && col_type) {
                    columns.push_back(std::string(col_name));
                    std::cout << "  " << col_name << " (" << col_type << ")" << std::endl;
                }
            }
            sqlite3_finalize_ptr(stmt);
            
            // Get total row count
            std::string count_sql = "SELECT COUNT(*) FROM \"" + table_name + "\";";
            rc = sqlite3_prepare_v2_ptr(db, count_sql.c_str(), -1, &stmt, nullptr);
            
            if (rc == SQLITE_OK && sqlite3_step_ptr(stmt) == SQLITE_ROW) {
                const char* count_str = reinterpret_cast<const char*>(sqlite3_column_text_ptr(stmt, 0));
                if (count_str) {
                    std::cout << "\nTOTAL ROWS: " << count_str << std::endl;
                }
                sqlite3_finalize_ptr(stmt);
            }
            
            // Get sample data with ALL columns
            std::string sample_sql = "SELECT * FROM \"" + table_name + "\" LIMIT 5;";
            rc = sqlite3_prepare_v2_ptr(db, sample_sql.c_str(), -1, &stmt, nullptr);
            
            if (rc == SQLITE_OK) {
                std::cout << "\nSAMPLE DATA:" << std::endl;
                int row_count = 0;
                while ((rc = sqlite3_step_ptr(stmt)) == SQLITE_ROW && row_count < 5) {
                    std::cout << "Row " << (row_count + 1) << ":" << std::endl;
                    int col_count = sqlite3_column_count_ptr ? sqlite3_column_count_ptr(stmt) : columns.size();
                    for (int col = 0; col < col_count && col < (int)columns.size(); ++col) {
                        const char* value = reinterpret_cast<const char*>(sqlite3_column_text_ptr(stmt, col));
                        std::cout << "  " << columns[col] << ": " << (value ? value : "NULL") << std::endl;
                    }
                    std::cout << std::endl;
                    row_count++;
                }
                sqlite3_finalize_ptr(stmt);
            }
            
            // Get unique symbols (try different possible column names)
            std::vector<std::string> symbol_columns = {"symbol", "Symbol", "SYMBOL", "ticker", "Ticker", "contract", "Contract"};
            
            for (const auto& col : symbol_columns) {
                std::string symbols_sql = "SELECT DISTINCT " + col + " FROM \"" + table_name + "\" ORDER BY " + col + " LIMIT 10;";
                rc = sqlite3_prepare_v2_ptr(db, symbols_sql.c_str(), -1, &stmt, nullptr);
                
                if (rc == SQLITE_OK) {
                    std::cout << "UNIQUE " << col << " VALUES:" << std::endl;
                    int symbol_count = 0;
                    while ((rc = sqlite3_step_ptr(stmt)) == SQLITE_ROW && symbol_count < 10) {
                        const char* symbol = reinterpret_cast<const char*>(sqlite3_column_text_ptr(stmt, 0));
                        if (symbol) {
                            std::cout << "  " << symbol << std::endl;
                            symbol_count++;
                        }
                    }
                    sqlite3_finalize_ptr(stmt);
                    
                    if (symbol_count > 0) {
                        std::cout << "*** FOUND SYMBOLS IN COLUMN: " << col << " ***" << std::endl;
                        break;
                    }
                } else {
                    // Column doesn't exist, continue to next
                }
            }
        }
    }
    
    sqlite3_close_ptr(db);
}

int main() {
    std::cout << "=== US Futures Database Diagnostic ===" << std::endl;
    
    if (!loadSQLite()) {
        std::cout << "Failed to load SQLite library" << std::endl;
        return 1;
    }
    
    diagnoseFuturesDatabase();
    
    std::cout << "\n" << std::string(60, '=') << std::endl;
    std::cout << "DIAGNOSTIC COMPLETED" << std::endl;
    std::cout << std::string(60, '=') << std::endl;
    std::cout << "\nThis diagnostic will help identify:" << std::endl;
    std::cout << "1. Correct table structure" << std::endl;
    std::cout << "2. Actual column names for price data" << std::endl;
    std::cout << "3. How symbols are stored" << std::endl;
    std::cout << "4. Sample data format" << std::endl;
    
    return 0;
}
