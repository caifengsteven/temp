#include <iostream>
#include <vector>
#include <string>
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
    sqlite3_errmsg_ptr = (sqlite3_errmsg_func)GetProcAddress(hModule, "sqlite3_errmsg");
    
    return (sqlite3_open_ptr && sqlite3_close_ptr && sqlite3_prepare_v2_ptr && 
            sqlite3_step_ptr && sqlite3_finalize_ptr);
}

int main() {
    std::cout << "=== China Database Diagnostic Tool ===" << std::endl;
    
    if (!loadSQLite()) {
        std::cout << "Failed to load SQLite library" << std::endl;
        return 1;
    }
    
    std::cout << "SQLite library loaded successfully" << std::endl;
    
    // Test database path
    std::string db_path = "I:\\zhubi\\cpp_implementation\\sqlite_databases\\2018\\2018_01.db";
    std::cout << "Testing database: " << db_path << std::endl;
    
    sqlite3* db;
    int rc = sqlite3_open_ptr(db_path.c_str(), &db);
    
    if (rc != SQLITE_OK) {
        std::cout << "FAILED to open database: " << sqlite3_errmsg_ptr(db) << std::endl;
        std::cout << "Error code: " << rc << std::endl;
        
        // Check if file exists
        HANDLE hFile = CreateFileA(db_path.c_str(), GENERIC_READ, FILE_SHARE_READ, NULL, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, NULL);
        if (hFile == INVALID_HANDLE_VALUE) {
            std::cout << "Database file does not exist or cannot be accessed" << std::endl;
            std::cout << "Please check:" << std::endl;
            std::cout << "1. Path exists: " << db_path << std::endl;
            std::cout << "2. File permissions" << std::endl;
            std::cout << "3. Drive is accessible" << std::endl;
        } else {
            CloseHandle(hFile);
            std::cout << "Database file exists but SQLite cannot open it" << std::endl;
        }
        
        if (db) sqlite3_close_ptr(db);
        return 1;
    }
    
    std::cout << "Database opened successfully!" << std::endl;
    
    // Test getting symbols
    std::string sql = "SELECT DISTINCT symbol FROM trade ORDER BY symbol LIMIT 10;";
    sqlite3_stmt* stmt;
    
    rc = sqlite3_prepare_v2_ptr(db, sql.c_str(), -1, &stmt, nullptr);
    if (rc != SQLITE_OK) {
        std::cout << "FAILED to prepare SQL: " << sqlite3_errmsg_ptr(db) << std::endl;
        sqlite3_close_ptr(db);
        return 1;
    }
    
    std::cout << "SQL prepared successfully" << std::endl;
    std::cout << "Getting first 10 symbols..." << std::endl;
    
    int count = 0;
    while ((rc = sqlite3_step_ptr(stmt)) == SQLITE_ROW) {
        const char* symbol = reinterpret_cast<const char*>(sqlite3_column_text_ptr(stmt, 0));
        if (symbol) {
            std::cout << "  " << (count + 1) << ". " << symbol << std::endl;
            count++;
        }
    }
    
    sqlite3_finalize_ptr(stmt);
    sqlite3_close_ptr(db);
    
    if (count == 0) {
        std::cout << "NO SYMBOLS FOUND!" << std::endl;
        std::cout << "This explains why the mass testing is stuck." << std::endl;
        std::cout << "The 'trade' table might be empty or have different structure." << std::endl;
    } else {
        std::cout << "Found " << count << " symbols successfully!" << std::endl;
        std::cout << "The database connection is working." << std::endl;
        std::cout << "The mass testing should work now." << std::endl;
    }
    
    return 0;
}
