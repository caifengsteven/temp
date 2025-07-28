#include <iostream>
#include <vector>
#include <string>
#include <set>
#include <fstream>
#include <iomanip>
#include <chrono>
#include <ctime>
#include <algorithm>
#include <windows.h>

// SQLite function pointers (same as before)
typedef struct sqlite3 sqlite3;
typedef struct sqlite3_stmt sqlite3_stmt;
typedef int (*sqlite3_open_func)(const char*, sqlite3**);
typedef int (*sqlite3_close_func)(sqlite3*);
typedef int (*sqlite3_prepare_v2_func)(sqlite3*, const char*, int, sqlite3_stmt**, const char**);
typedef int (*sqlite3_step_func)(sqlite3_stmt*);
typedef int (*sqlite3_finalize_func)(sqlite3_stmt*);
typedef const unsigned char* (*sqlite3_column_text_func)(sqlite3_stmt*, int);
typedef int (*sqlite3_column_int_func)(sqlite3_stmt*, int);
typedef const char* (*sqlite3_errmsg_func)(sqlite3*);

#define SQLITE_OK           0
#define SQLITE_ROW          100
#define SQLITE_DONE         101

// Global function pointers
sqlite3_open_func sqlite3_open_ptr = nullptr;
sqlite3_close_func sqlite3_close_ptr = nullptr;
sqlite3_prepare_v2_func sqlite3_prepare_v2_ptr = nullptr;
sqlite3_step_func sqlite3_step_ptr = nullptr;
sqlite3_finalize_func sqlite3_finalize_ptr = nullptr;
sqlite3_column_text_func sqlite3_column_text_ptr = nullptr;
sqlite3_column_int_func sqlite3_column_int_ptr = nullptr;
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
    sqlite3_column_int_ptr = (sqlite3_column_int_func)GetProcAddress(hModule, "sqlite3_column_int");
    sqlite3_errmsg_ptr = (sqlite3_errmsg_func)GetProcAddress(hModule, "sqlite3_errmsg");
    
    return (sqlite3_open_ptr && sqlite3_close_ptr && sqlite3_prepare_v2_ptr && 
            sqlite3_step_ptr && sqlite3_finalize_ptr);
}

// Get all database files
std::vector<std::string> getAllDatabaseFiles() {
    std::vector<std::string> db_files;
    const std::string base_path = "I:/zhubi/cpp_implementation/sqlite_databases/";
    
    for (int year = 2018; year <= 2025; year++) {
        for (int month = 1; month <= 12; month++) {
            std::string month_str = (month < 10) ? "0" + std::to_string(month) : std::to_string(month);
            std::string db_path = base_path + std::to_string(year) + "/" + std::to_string(year) + "_" + month_str + ".db";
            
            HANDLE hFile = CreateFileA(db_path.c_str(), GENERIC_READ, FILE_SHARE_READ, NULL, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, NULL);
            if (hFile != INVALID_HANDLE_VALUE) {
                CloseHandle(hFile);
                db_files.push_back(db_path);
            }
        }
    }
    
    return db_files;
}

// Scan all databases to find unique symbols
std::set<std::string> scanAllSymbols() {
    std::set<std::string> all_symbols;
    auto db_files = getAllDatabaseFiles();
    
    std::cout << "Scanning " << db_files.size() << " database files for symbols..." << std::endl;
    
    int file_count = 0;
    for (const auto& db_path : db_files) {
        file_count++;
        std::cout << "Scanning [" << file_count << "/" << db_files.size() << "]: " 
                  << db_path.substr(db_path.find_last_of("/\\") + 1) << std::endl;
        
        sqlite3* db;
        int rc = sqlite3_open_ptr(db_path.c_str(), &db);
        
        if (rc != SQLITE_OK) {
            std::cout << "  Skipped (cannot open)" << std::endl;
            continue;
        }
        
        const char* symbols_sql = "SELECT DISTINCT symbol FROM trade;";
        sqlite3_stmt* stmt;
        
        rc = sqlite3_prepare_v2_ptr(db, symbols_sql, -1, &stmt, nullptr);
        if (rc != SQLITE_OK) {
            sqlite3_close_ptr(db);
            std::cout << "  Skipped (SQL error)" << std::endl;
            continue;
        }
        
        int symbol_count = 0;
        while ((rc = sqlite3_step_ptr(stmt)) == SQLITE_ROW) {
            const char* symbol = reinterpret_cast<const char*>(sqlite3_column_text_ptr(stmt, 0));
            if (symbol) {
                all_symbols.insert(std::string(symbol));
                symbol_count++;
            }
        }
        
        sqlite3_finalize_ptr(stmt);
        sqlite3_close_ptr(db);
        
        std::cout << "  Found: " << symbol_count << " symbols" << std::endl;
    }
    
    return all_symbols;
}

// Save symbols list to file
void saveSymbolsList(const std::set<std::string>& symbols) {
    std::ofstream file("all_symbols.txt");
    if (file.is_open()) {
        for (const auto& symbol : symbols) {
            file << symbol << std::endl;
        }
        file.close();
        std::cout << "Symbols list saved to: all_symbols.txt" << std::endl;
    }
}

// Load symbols list from file
std::set<std::string> loadSymbolsList() {
    std::set<std::string> symbols;
    std::ifstream file("all_symbols.txt");
    if (file.is_open()) {
        std::string symbol;
        while (std::getline(file, symbol)) {
            if (!symbol.empty()) {
                symbols.insert(symbol);
            }
        }
        file.close();
        std::cout << "Loaded " << symbols.size() << " symbols from all_symbols.txt" << std::endl;
    }
    return symbols;
}

// Load Excel symbols list from file
std::set<std::string> loadExcelSymbolsList() {
    std::set<std::string> symbols;
    std::ifstream file("excel_stocks.txt");
    if (file.is_open()) {
        std::string symbol;
        while (std::getline(file, symbol)) {
            if (!symbol.empty()) {
                symbols.insert(symbol);
            }
        }
        file.close();
        std::cout << "Loaded " << symbols.size() << " symbols from excel_stocks.txt" << std::endl;
    }
    return symbols;
}

// Progress tracking
struct TestingProgress {
    std::set<std::string> completed_symbols;
    std::set<std::string> failed_symbols;
    size_t total_symbols;
    
    void save() {
        std::ofstream file("testing_progress.txt");
        if (file.is_open()) {
            file << total_symbols << std::endl;
            file << completed_symbols.size() << std::endl;
            for (const auto& symbol : completed_symbols) {
                file << "COMPLETED:" << symbol << std::endl;
            }
            file << failed_symbols.size() << std::endl;
            for (const auto& symbol : failed_symbols) {
                file << "FAILED:" << symbol << std::endl;
            }
            file.close();
        }
    }
    
    void load() {
        std::ifstream file("testing_progress.txt");
        if (file.is_open()) {
            file >> total_symbols;
            
            size_t completed_count;
            file >> completed_count;
            file.ignore();
            
            for (size_t i = 0; i < completed_count; i++) {
                std::string line;
                std::getline(file, line);
                if (line.substr(0, 10) == "COMPLETED:") {
                    completed_symbols.insert(line.substr(10));
                }
            }
            
            size_t failed_count;
            file >> failed_count;
            file.ignore();
            
            for (size_t i = 0; i < failed_count; i++) {
                std::string line;
                std::getline(file, line);
                if (line.substr(0, 7) == "FAILED:") {
                    failed_symbols.insert(line.substr(7));
                }
            }
            
            file.close();
            std::cout << "Progress loaded: " << completed_symbols.size() << " completed, " 
                      << failed_symbols.size() << " failed" << std::endl;
        }
    }
    
    void printStatus() {
        std::cout << "\n=== TESTING PROGRESS ===" << std::endl;
        std::cout << "Total symbols: " << total_symbols << std::endl;
        std::cout << "Completed: " << completed_symbols.size() << std::endl;
        std::cout << "Failed: " << failed_symbols.size() << std::endl;
        std::cout << "Remaining: " << (total_symbols - completed_symbols.size() - failed_symbols.size()) << std::endl;
        if (total_symbols > 0) {
            double progress = (double)(completed_symbols.size() + failed_symbols.size()) / total_symbols * 100.0;
            std::cout << "Progress: " << std::fixed << std::setprecision(1) << progress << "%" << std::endl;
        }
        std::cout << "=========================" << std::endl;
    }
};

// Run single symbol test
bool runSymbolTest(const std::string& symbol) {
    std::string command = "dc_database_test_with_reports.exe " + symbol;
    
    std::cout << "Testing symbol: " << symbol << std::endl;
    
    int result = system(command.c_str());
    
    if (result == 0) {
        std::cout << "SUCCESS: " << symbol << " completed" << std::endl;
        return true;
    } else {
        std::cout << "FAILED: " << symbol << " failed with code " << result << std::endl;
        return false;
    }
}

void printUsage(const char* program_name) {
    std::cout << "Usage:" << std::endl;
    std::cout << "  " << program_name << " --scan          # Scan databases for all symbols" << std::endl;
    std::cout << "  " << program_name << " --test-all      # Test all symbols (resume if stopped)" << std::endl;
    std::cout << "  " << program_name << " --test-excel    # Test Excel symbols (resume if stopped)" << std::endl;
    std::cout << "  " << program_name << " --status        # Show testing progress" << std::endl;
    std::cout << "  " << program_name << " --list          # Show all discovered symbols" << std::endl;
    std::cout << "  " << program_name << " --list-excel    # Show Excel symbols" << std::endl;
}

int main(int argc, char* argv[]) {
    std::cout << "=== DC Generator Mass Testing System ===" << std::endl;
    
    if (argc != 2) {
        printUsage(argv[0]);
        return 1;
    }
    
    std::string command = argv[1];
    
    if (!loadSQLite()) {
        std::cout << "Failed to load SQLite library" << std::endl;
        return 1;
    }
    
    if (command == "--scan") {
        std::cout << "Scanning all databases for symbols..." << std::endl;
        auto symbols = scanAllSymbols();
        
        std::cout << "\n=== SCAN RESULTS ===" << std::endl;
        std::cout << "Total unique symbols found: " << symbols.size() << std::endl;
        
        saveSymbolsList(symbols);
        
        std::cout << "\nFirst 20 symbols:" << std::endl;
        int count = 0;
        for (const auto& symbol : symbols) {
            std::cout << "  " << symbol << std::endl;
            if (++count >= 20) break;
        }
        
        if (symbols.size() > 20) {
            std::cout << "  ... and " << (symbols.size() - 20) << " more" << std::endl;
        }
        
    } else if (command == "--test-all") {
        auto symbols = loadSymbolsList();
        if (symbols.empty()) {
            std::cout << "No symbols found. Run --scan first." << std::endl;
            return 1;
        }
        
        TestingProgress progress;
        progress.total_symbols = symbols.size();
        progress.load();
        
        std::cout << "Starting mass testing of " << symbols.size() << " symbols..." << std::endl;
        progress.printStatus();
        
        for (const auto& symbol : symbols) {
            // Skip if already completed or failed
            if (progress.completed_symbols.count(symbol) || progress.failed_symbols.count(symbol)) {
                continue;
            }
            
            std::cout << "\n" << std::string(50, '=') << std::endl;
            
            bool success = runSymbolTest(symbol);
            
            if (success) {
                progress.completed_symbols.insert(symbol);
            } else {
                progress.failed_symbols.insert(symbol);
            }
            
            progress.save();
            progress.printStatus();
        }
        
        std::cout << "\n=== MASS TESTING COMPLETED ===" << std::endl;
        progress.printStatus();

    } else if (command == "--test-excel") {
        auto symbols = loadExcelSymbolsList();
        if (symbols.empty()) {
            std::cout << "No Excel symbols found. Run read_excel_stocks.py first." << std::endl;
            return 1;
        }

        TestingProgress progress;
        progress.total_symbols = symbols.size();
        progress.load();

        std::cout << "Starting Excel stock testing of " << symbols.size() << " symbols..." << std::endl;
        progress.printStatus();

        for (const auto& symbol : symbols) {
            // Skip if already completed or failed
            if (progress.completed_symbols.count(symbol) || progress.failed_symbols.count(symbol)) {
                continue;
            }

            std::cout << "\n" << std::string(50, '=') << std::endl;

            bool success = runSymbolTest(symbol);

            if (success) {
                progress.completed_symbols.insert(symbol);
            } else {
                progress.failed_symbols.insert(symbol);
            }

            progress.save();
            progress.printStatus();
        }

        std::cout << "\n=== EXCEL STOCK TESTING COMPLETED ===" << std::endl;
        progress.printStatus();

    } else if (command == "--status") {
        TestingProgress progress;
        progress.load();
        progress.printStatus();
        
    } else if (command == "--list") {
        auto symbols = loadSymbolsList();
        std::cout << "All discovered symbols (" << symbols.size() << "):" << std::endl;
        for (const auto& symbol : symbols) {
            std::cout << "  " << symbol << std::endl;
        }

    } else if (command == "--list-excel") {
        auto symbols = loadExcelSymbolsList();
        std::cout << "Excel symbols (" << symbols.size() << "):" << std::endl;

        int sh_count = 0, sz_count = 0;
        for (const auto& symbol : symbols) {
            std::cout << "  " << symbol << std::endl;
            if (symbol.substr(0, 2) == "sh") sh_count++;
            else if (symbol.substr(0, 2) == "sz") sz_count++;
        }

        std::cout << "\nSummary:" << std::endl;
        std::cout << "  Shanghai (sh): " << sh_count << std::endl;
        std::cout << "  Shenzhen (sz): " << sz_count << std::endl;
        std::cout << "  Total: " << symbols.size() << std::endl;

    } else {
        std::cout << "Unknown command: " << command << std::endl;
        printUsage(argv[0]);
        return 1;
    }
    
    return 0;
}
