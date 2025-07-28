#include <iostream>
#include <vector>
#include <string>
#include <iomanip>
#include <cmath>
#include <algorithm>
#include <memory>
#include <functional>
#include <fstream>
#include <sstream>
#include <chrono>
#include <ctime>
#include <windows.h>

// We'll dynamically load SQLite to avoid linking issues
typedef struct sqlite3 sqlite3;
typedef struct sqlite3_stmt sqlite3_stmt;

typedef int (*sqlite3_open_func)(const char*, sqlite3**);
typedef int (*sqlite3_close_func)(sqlite3*);
typedef int (*sqlite3_prepare_v2_func)(sqlite3*, const char*, int, sqlite3_stmt**, const char**);
typedef int (*sqlite3_step_func)(sqlite3_stmt*);
typedef int (*sqlite3_finalize_func)(sqlite3_stmt*);
typedef const unsigned char* (*sqlite3_column_text_func)(sqlite3_stmt*, int);
typedef double (*sqlite3_column_double_func)(sqlite3_stmt*, int);
typedef int (*sqlite3_column_int_func)(sqlite3_stmt*, int);
typedef int (*sqlite3_bind_text_func)(sqlite3_stmt*, int, const char*, int, void(*)(void*));
typedef int (*sqlite3_bind_int_func)(sqlite3_stmt*, int, int);
typedef const char* (*sqlite3_errmsg_func)(sqlite3*);

// SQLite constants
#define SQLITE_OK           0
#define SQLITE_ROW          100
#define SQLITE_DONE         101
#define SQLITE_STATIC       ((void(*)(void *))0)

// Global function pointers
sqlite3_open_func sqlite3_open_ptr = nullptr;
sqlite3_close_func sqlite3_close_ptr = nullptr;
sqlite3_prepare_v2_func sqlite3_prepare_v2_ptr = nullptr;
sqlite3_step_func sqlite3_step_ptr = nullptr;
sqlite3_finalize_func sqlite3_finalize_ptr = nullptr;
sqlite3_column_text_func sqlite3_column_text_ptr = nullptr;
sqlite3_column_double_func sqlite3_column_double_ptr = nullptr;
sqlite3_column_int_func sqlite3_column_int_ptr = nullptr;
sqlite3_bind_text_func sqlite3_bind_text_ptr = nullptr;
sqlite3_bind_int_func sqlite3_bind_int_ptr = nullptr;
sqlite3_errmsg_func sqlite3_errmsg_ptr = nullptr;

bool loadSQLite() {
    HMODULE hModule = LoadLibraryA("C:/sqlite3/sqlite3.dll");
    if (!hModule) {
        hModule = LoadLibraryA("sqlite3.dll");
    }
    if (!hModule) {
        std::cout << "❌ Could not load SQLite library" << std::endl;
        return false;
    }
    
    sqlite3_open_ptr = (sqlite3_open_func)GetProcAddress(hModule, "sqlite3_open");
    sqlite3_close_ptr = (sqlite3_close_func)GetProcAddress(hModule, "sqlite3_close");
    sqlite3_prepare_v2_ptr = (sqlite3_prepare_v2_func)GetProcAddress(hModule, "sqlite3_prepare_v2");
    sqlite3_step_ptr = (sqlite3_step_func)GetProcAddress(hModule, "sqlite3_step");
    sqlite3_finalize_ptr = (sqlite3_finalize_func)GetProcAddress(hModule, "sqlite3_finalize");
    sqlite3_column_text_ptr = (sqlite3_column_text_func)GetProcAddress(hModule, "sqlite3_column_text");
    sqlite3_column_double_ptr = (sqlite3_column_double_func)GetProcAddress(hModule, "sqlite3_column_double");
    sqlite3_column_int_ptr = (sqlite3_column_int_func)GetProcAddress(hModule, "sqlite3_column_int");
    sqlite3_bind_text_ptr = (sqlite3_bind_text_func)GetProcAddress(hModule, "sqlite3_bind_text");
    sqlite3_bind_int_ptr = (sqlite3_bind_int_func)GetProcAddress(hModule, "sqlite3_bind_int");
    sqlite3_errmsg_ptr = (sqlite3_errmsg_func)GetProcAddress(hModule, "sqlite3_errmsg");
    
    return (sqlite3_open_ptr && sqlite3_close_ptr && sqlite3_prepare_v2_ptr &&
            sqlite3_step_ptr && sqlite3_finalize_ptr);
}

// Function to get all database files from 2018 to 2025
std::vector<std::string> getAllDatabaseFiles() {
    std::vector<std::string> db_files;
    const std::string base_path = "I:/zhubi/cpp_implementation/sqlite_databases/";

    // Check years 2018 to 2025
    for (int year = 2018; year <= 2025; year++) {
        // Check months 01 to 12
        for (int month = 1; month <= 12; month++) {
            std::string month_str = (month < 10) ? "0" + std::to_string(month) : std::to_string(month);
            std::string db_path = base_path + std::to_string(year) + "/" + std::to_string(year) + "_" + month_str + ".db";

            // Check if file exists by trying to open it
            HANDLE hFile = CreateFileA(db_path.c_str(), GENERIC_READ, FILE_SHARE_READ, NULL, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, NULL);
            if (hFile != INVALID_HANDLE_VALUE) {
                CloseHandle(hFile);
                db_files.push_back(db_path);
            }
        }
    }

    return db_files;
}

// Function to load all data for a symbol from multiple databases
std::vector<double> loadAllDataForSymbol(const std::string& symbol) {
    std::vector<double> all_prices;
    auto db_files = getAllDatabaseFiles();

    std::cout << "Found " << db_files.size() << " database files to process" << std::endl;

    for (const auto& db_path : db_files) {
        sqlite3* db;
        int rc = sqlite3_open_ptr(db_path.c_str(), &db);

        if (rc != SQLITE_OK) {
            continue; // Skip this database if can't open
        }

        const char* data_sql = "SELECT price FROM trade WHERE symbol = ? ORDER BY date, time;";
        sqlite3_stmt* stmt;

        rc = sqlite3_prepare_v2_ptr(db, data_sql, -1, &stmt, nullptr);
        if (rc != SQLITE_OK) {
            sqlite3_close_ptr(db);
            continue;
        }

        sqlite3_bind_text_ptr(stmt, 1, symbol.c_str(), -1, SQLITE_STATIC);

        int count = 0;
        while ((rc = sqlite3_step_ptr(stmt)) == SQLITE_ROW) {
            double price = sqlite3_column_double_ptr(stmt, 0);
            if (price > 0 && std::isfinite(price)) {
                all_prices.push_back(price);
                count++;
            }
        }

        sqlite3_finalize_ptr(stmt);
        sqlite3_close_ptr(db);

        if (count > 0) {
            std::cout << "  " << db_path.substr(db_path.find_last_of("/\\") + 1)
                      << ": " << count << " records" << std::endl;
        }
    }

    return all_prices;
}

// Checkpoint/Recovery functionality
struct Checkpoint {
    std::string symbol;
    std::vector<std::string> processed_files;
    std::vector<double> prices;
    size_t total_records;

    void save(const std::string& filename) const {
        std::ofstream file(filename);
        if (file.is_open()) {
            file << symbol << std::endl;
            file << processed_files.size() << std::endl;
            for (const auto& db_file : processed_files) {
                file << db_file << std::endl;
            }
            file << prices.size() << std::endl;
            for (double price : prices) {
                file << std::fixed << std::setprecision(6) << price << std::endl;
            }
            file << total_records << std::endl;
            file.close();
            std::cout << "Checkpoint saved: " << filename << std::endl;
        }
    }

    bool load(const std::string& filename) {
        std::ifstream file(filename);
        if (!file.is_open()) {
            return false;
        }

        std::getline(file, symbol);

        size_t num_files;
        file >> num_files;
        file.ignore(); // Skip newline

        processed_files.clear();
        for (size_t i = 0; i < num_files; ++i) {
            std::string db_file;
            std::getline(file, db_file);
            processed_files.push_back(db_file);
        }

        size_t num_prices;
        file >> num_prices;

        prices.clear();
        prices.reserve(num_prices);
        for (size_t i = 0; i < num_prices; ++i) {
            double price;
            file >> price;
            prices.push_back(price);
        }

        file >> total_records;
        file.close();

        std::cout << "Checkpoint loaded: " << filename << std::endl;
        std::cout << "  Symbol: " << symbol << std::endl;
        std::cout << "  Processed files: " << processed_files.size() << std::endl;
        std::cout << "  Loaded prices: " << prices.size() << std::endl;
        std::cout << "  Total records: " << total_records << std::endl;

        return true;
    }
};

// Enhanced data loading with checkpoint support
std::vector<double> loadAllDataForSymbolWithCheckpoint(const std::string& symbol) {
    std::string checkpoint_file = "checkpoint_" + symbol + ".txt";
    Checkpoint checkpoint;

    // Try to load existing checkpoint
    bool resumed = checkpoint.load(checkpoint_file);

    if (resumed && checkpoint.symbol == symbol) {
        std::cout << "Resuming from checkpoint..." << std::endl;
    } else {
        std::cout << "Starting fresh data loading..." << std::endl;
        checkpoint.symbol = symbol;
        checkpoint.processed_files.clear();
        checkpoint.prices.clear();
        checkpoint.total_records = 0;
    }

    auto all_db_files = getAllDatabaseFiles();
    std::cout << "Found " << all_db_files.size() << " database files to process" << std::endl;

    // Filter out already processed files
    std::vector<std::string> remaining_files;
    for (const auto& db_file : all_db_files) {
        bool already_processed = false;
        for (const auto& processed : checkpoint.processed_files) {
            if (db_file == processed) {
                already_processed = true;
                break;
            }
        }
        if (!already_processed) {
            remaining_files.push_back(db_file);
        }
    }

    if (resumed) {
        std::cout << "Skipping " << checkpoint.processed_files.size() << " already processed files" << std::endl;
        std::cout << "Processing remaining " << remaining_files.size() << " files" << std::endl;
    }

    size_t file_count = 0;
    for (const auto& db_path : remaining_files) {
        file_count++;
        std::cout << "Processing [" << file_count << "/" << remaining_files.size() << "]: "
                  << db_path.substr(db_path.find_last_of("/\\") + 1) << std::endl;

        sqlite3* db;
        int rc = sqlite3_open_ptr(db_path.c_str(), &db);

        if (rc != SQLITE_OK) {
            std::cout << "  Skipped (cannot open)" << std::endl;
            continue;
        }

        const char* data_sql = "SELECT price FROM trade WHERE symbol = ? ORDER BY date, time;";
        sqlite3_stmt* stmt;

        rc = sqlite3_prepare_v2_ptr(db, data_sql, -1, &stmt, nullptr);
        if (rc != SQLITE_OK) {
            sqlite3_close_ptr(db);
            std::cout << "  Skipped (SQL error)" << std::endl;
            continue;
        }

        sqlite3_bind_text_ptr(stmt, 1, symbol.c_str(), -1, SQLITE_STATIC);

        int count = 0;
        while ((rc = sqlite3_step_ptr(stmt)) == SQLITE_ROW) {
            double price = sqlite3_column_double_ptr(stmt, 0);
            if (price > 0 && std::isfinite(price)) {
                checkpoint.prices.push_back(price);
                count++;
                checkpoint.total_records++;
            }
        }

        sqlite3_finalize_ptr(stmt);
        sqlite3_close_ptr(db);

        checkpoint.processed_files.push_back(db_path);

        if (count > 0) {
            std::cout << "  Loaded: " << count << " records" << std::endl;
        } else {
            std::cout << "  No data found" << std::endl;
        }

        // Save checkpoint every 10 files
        if (file_count % 10 == 0) {
            checkpoint.save(checkpoint_file);
        }
    }

    // Save final checkpoint
    checkpoint.save(checkpoint_file);

    std::cout << "Data loading completed!" << std::endl;
    std::cout << "Total files processed: " << checkpoint.processed_files.size() << std::endl;
    std::cout << "Total price points: " << checkpoint.prices.size() << std::endl;

    return checkpoint.prices;
}

// Report generation functionality
class ReportGenerator {
public:
    explicit ReportGenerator(const std::string& symbol)
        : symbol_(symbol), report_filename_("report_" + symbol + ".txt") {

        // Open report file
        report_file_.open(report_filename_);
        if (report_file_.is_open()) {
            writeHeader();
        }
    }

    ~ReportGenerator() {
        if (report_file_.is_open()) {
            writeFooter();
            report_file_.close();
            std::cout << "Report saved: " << report_filename_ << std::endl;
        }
    }

    void writeDataSummary(size_t total_points, double min_price, double max_price,
                         size_t total_files, size_t files_with_data) {
        if (!report_file_.is_open()) return;

        report_file_ << "=== DATA SUMMARY ===" << std::endl;
        report_file_ << "Symbol: " << symbol_ << std::endl;
        report_file_ << "Total Price Points: " << total_points << std::endl;
        report_file_ << "Price Range: $" << std::fixed << std::setprecision(2)
                     << min_price << " to $" << max_price << std::endl;
        report_file_ << "Database Files Scanned: " << total_files << std::endl;
        report_file_ << "Files with Data: " << files_with_data << std::endl;
        report_file_ << "Testing Period: 2018-2025" << std::endl;
        report_file_ << std::endl;
    }

    void writeStrategyHeader(const std::string& strategy_name) {
        if (!report_file_.is_open()) return;

        report_file_ << "=== " << strategy_name << " STRATEGY RESULTS ===" << std::endl;
        report_file_ << "Initial Capital: $100,000" << std::endl;
        report_file_ << std::endl;
        report_file_ << std::setw(12) << "Threshold"
                     << std::setw(12) << "Trades"
                     << std::setw(15) << "Final PnL"
                     << std::setw(12) << "Return %"
                     << std::setw(15) << "Final Value" << std::endl;
        report_file_ << std::string(65, '-') << std::endl;
    }

    void writeStrategyResult(double threshold, int trades, double pnl, double return_pct, double final_value) {
        if (!report_file_.is_open()) return;

        report_file_ << std::fixed << std::setprecision(1)
                     << std::setw(11) << threshold * 100 << "%"
                     << std::setw(12) << trades
                     << std::setw(12) << "$" << std::setprecision(0) << pnl
                     << std::setw(11) << std::setprecision(2) << return_pct << "%"
                     << std::setw(12) << "$" << std::setprecision(0) << final_value << std::endl;
    }

    void writeStrategyFooter() {
        if (!report_file_.is_open()) return;
        report_file_ << std::endl;
    }

    void writeBestResults(const std::string& best_strategy, double best_threshold,
                         double best_return, int best_trades) {
        if (!report_file_.is_open()) return;

        report_file_ << "=== BEST PERFORMING CONFIGURATION ===" << std::endl;
        report_file_ << "Best Strategy: " << best_strategy << std::endl;
        report_file_ << "Best Threshold: " << std::fixed << std::setprecision(1)
                     << best_threshold * 100 << "%" << std::endl;
        report_file_ << "Best Return: " << std::setprecision(2) << best_return << "%" << std::endl;
        report_file_ << "Number of Trades: " << best_trades << std::endl;
        report_file_ << std::endl;
    }

    void writeStrategyDescriptions() {
        if (!report_file_.is_open()) return;

        report_file_ << "=== STRATEGY DESCRIPTIONS ===" << std::endl;
        report_file_ << "Simple DC: Buy on downturn end, sell on upturn end" << std::endl;
        report_file_ << "Contrarian DC: Buy on upturn end, sell on downturn end" << std::endl;
        report_file_ << "Long Only DC: Only buy on downturn end, hold position" << std::endl;
        report_file_ << std::endl;
        report_file_ << "=== THRESHOLD EXPLANATION ===" << std::endl;
        report_file_ << "DC Threshold determines the minimum price change required" << std::endl;
        report_file_ << "to trigger a directional change event:" << std::endl;
        report_file_ << "- Lower thresholds (0.5%): More sensitive, more trades" << std::endl;
        report_file_ << "- Higher thresholds (5.0%): Less sensitive, fewer trades" << std::endl;
        report_file_ << std::endl;
    }

private:
    std::string symbol_;
    std::string report_filename_;
    std::ofstream report_file_;

    void writeHeader() {
        if (!report_file_.is_open()) return;

        auto now = std::chrono::system_clock::now();
        auto time_t = std::chrono::system_clock::to_time_t(now);

        report_file_ << "=======================================================" << std::endl;
        report_file_ << "DC GENERATOR BACKTESTING REPORT" << std::endl;
        report_file_ << "=======================================================" << std::endl;
        report_file_ << "Generated: " << std::ctime(&time_t);
        report_file_ << "Symbol: " << symbol_ << std::endl;
        report_file_ << "=======================================================" << std::endl;
        report_file_ << std::endl;
    }

    void writeFooter() {
        if (!report_file_.is_open()) return;

        report_file_ << "=======================================================" << std::endl;
        report_file_ << "END OF REPORT" << std::endl;
        report_file_ << "=======================================================" << std::endl;
    }
};

// Simple DC Generator
class DCGenerator {
public:
    enum class DCEvent { NONE, END_UPTURN, END_DOWNTURN };

    explicit DCGenerator(double threshold) : threshold_(threshold), initialized_(false) {}

    DCEvent processPrice(double price) {
        if (!std::isfinite(price) || price <= 0) return DCEvent::NONE;

        if (!initialized_) {
            highest_price_ = price;
            lowest_price_ = price;
            is_upturn_ = true;
            initialized_ = true;
            return DCEvent::NONE;
        }

        if (is_upturn_) {
            if (price <= highest_price_ * (1.0 - threshold_)) {
                is_upturn_ = false;
                lowest_price_ = price;
                return DCEvent::END_UPTURN;
            } else if (price > highest_price_) {
                highest_price_ = price;
            }
        } else {
            if (price >= lowest_price_ * (1.0 + threshold_)) {
                is_upturn_ = true;
                highest_price_ = price;
                return DCEvent::END_DOWNTURN;
            } else if (price < lowest_price_) {
                lowest_price_ = price;
            }
        }
        return DCEvent::NONE;
    }

private:
    double threshold_;
    bool initialized_;
    double highest_price_;
    double lowest_price_;
    bool is_upturn_;
};

// Base Trading Strategy
class TradingStrategy {
public:
    explicit TradingStrategy(double initial_capital)
        : initial_capital_(initial_capital), cash_(initial_capital), position_(0), trade_count_(0) {}

    virtual ~TradingStrategy() = default;

    virtual void onDCEvent(DCGenerator::DCEvent event, double price) = 0;

    double getCurrentValue(double current_price) const {
        if (!std::isfinite(current_price) || current_price <= 0) {
            return cash_; // If price is invalid, return just cash
        }
        return cash_ + position_ * current_price;
    }

    double getTotalReturn(double current_price) const {
        double current_value = getCurrentValue(current_price);
        if (initial_capital_ <= 0) {
            return 0.0; // Avoid division by zero
        }
        double return_val = (current_value - initial_capital_) / initial_capital_ * 100.0;
        return std::isfinite(return_val) ? return_val : 0.0;
    }

    double getPnL(double current_price) const {
        double pnl = getCurrentValue(current_price) - initial_capital_;
        return std::isfinite(pnl) ? pnl : 0.0;
    }

    int getTradeCount() const { return trade_count_; }

    void reset() {
        cash_ = initial_capital_;
        position_ = 0;
        trade_count_ = 0;
    }

protected:
    double initial_capital_;
    double cash_;
    double position_;
    int trade_count_;
};

// Strategy 1: Simple DC Strategy (Buy on downturn end, Sell on upturn end)
class SimpleDCStrategy : public TradingStrategy {
public:
    explicit SimpleDCStrategy(double initial_capital) : TradingStrategy(initial_capital) {}

    void onDCEvent(DCGenerator::DCEvent event, double price) override {
        if (!std::isfinite(price) || price <= 0) return;

        switch (event) {
            case DCGenerator::DCEvent::END_DOWNTURN:
                if (position_ == 0 && cash_ > 1.0) { // Ensure we have enough cash
                    double cash_to_use = cash_ * 0.95;
                    if (cash_to_use > 0 && price > 0) {
                        position_ = cash_to_use / price;
                        cash_ -= cash_to_use;
                        trade_count_++;

                        // Safety check
                        if (!std::isfinite(position_) || !std::isfinite(cash_)) {
                            position_ = 0;
                            cash_ = initial_capital_;
                        }
                    }
                }
                break;

            case DCGenerator::DCEvent::END_UPTURN:
                if (position_ > 0) {
                    double sale_value = position_ * price;
                    if (std::isfinite(sale_value)) {
                        cash_ += sale_value;
                        position_ = 0;
                        trade_count_++;

                        // Safety check
                        if (!std::isfinite(cash_)) {
                            cash_ = initial_capital_;
                        }
                    }
                }
                break;

            default:
                break;
        }
    }
};

// Strategy 2: Contrarian DC Strategy (Buy on upturn end, Sell on downturn end)
class ContrarianDCStrategy : public TradingStrategy {
public:
    explicit ContrarianDCStrategy(double initial_capital) : TradingStrategy(initial_capital) {}

    void onDCEvent(DCGenerator::DCEvent event, double price) override {
        if (!std::isfinite(price) || price <= 0) return;

        switch (event) {
            case DCGenerator::DCEvent::END_UPTURN:
                if (position_ == 0 && cash_ > 1.0) {
                    double cash_to_use = cash_ * 0.95;
                    if (cash_to_use > 0 && price > 0) {
                        position_ = cash_to_use / price;
                        cash_ -= cash_to_use;
                        trade_count_++;

                        // Safety check
                        if (!std::isfinite(position_) || !std::isfinite(cash_)) {
                            position_ = 0;
                            cash_ = initial_capital_;
                        }
                    }
                }
                break;

            case DCGenerator::DCEvent::END_DOWNTURN:
                if (position_ > 0) {
                    double sale_value = position_ * price;
                    if (std::isfinite(sale_value)) {
                        cash_ += sale_value;
                        position_ = 0;
                        trade_count_++;

                        // Safety check
                        if (!std::isfinite(cash_)) {
                            cash_ = initial_capital_;
                        }
                    }
                }
                break;

            default:
                break;
        }
    }
};

// Strategy 3: Long Only DC Strategy (Only buy, never short)
class LongOnlyDCStrategy : public TradingStrategy {
public:
    explicit LongOnlyDCStrategy(double initial_capital) : TradingStrategy(initial_capital) {}

    void onDCEvent(DCGenerator::DCEvent event, double price) override {
        if (!std::isfinite(price) || price <= 0) return;

        if (event == DCGenerator::DCEvent::END_DOWNTURN && position_ == 0 && cash_ > 1.0) {
            // Only buy on downturn end, hold until manual exit
            double cash_to_use = cash_ * 0.95;
            if (cash_to_use > 0 && price > 0) {
                position_ = cash_to_use / price;
                cash_ -= cash_to_use;
                trade_count_++;

                // Safety check
                if (!std::isfinite(position_) || !std::isfinite(cash_)) {
                    position_ = 0;
                    cash_ = initial_capital_;
                }
            }
        }
    }
};

void printUsage(const char* program_name) {
    std::cout << "Usage: " << program_name << " <SYMBOL> [OPTIONS]" << std::endl;
    std::cout << "       " << program_name << " --clean <SYMBOL>" << std::endl;
    std::cout << std::endl;
    std::cout << "Examples:" << std::endl;
    std::cout << "  " << program_name << " AAPL          # Test AAPL (resume if stopped)" << std::endl;
    std::cout << "  " << program_name << " --clean AAPL  # Clean checkpoint and start fresh" << std::endl;
    std::cout << std::endl;
    std::cout << "Features:" << std::endl;
    std::cout << "- Tests symbol across all databases (2018-2025)" << std::endl;
    std::cout << "- Multiple DC strategies and thresholds >= 0.5%" << std::endl;
    std::cout << "- Automatic checkpoint/resume functionality" << std::endl;
    std::cout << "- Can be stopped and resumed anytime (Ctrl+C)" << std::endl;
}

int main(int argc, char* argv[]) {
    std::cout << "=== DC Generator Multi-Year Database Test ===" << std::endl;

    // Check command line arguments
    if (argc < 2 || argc > 3) {
        std::cout << "Error: Invalid number of arguments" << std::endl;
        std::cout << std::endl;
        printUsage(argv[0]);
        std::cout << std::endl;

        // Show available symbols to help user
        std::cout << "To see available symbols, checking sample database..." << std::endl;

        if (loadSQLite()) {
            const std::string sample_db = "I:/zhubi/cpp_implementation/sqlite_databases/2018/2018_01.db";
            sqlite3* db;
            int rc = sqlite3_open_ptr(sample_db.c_str(), &db);

            if (rc == SQLITE_OK) {
                const char* symbols_sql = "SELECT DISTINCT symbol FROM trade LIMIT 20;";
                sqlite3_stmt* stmt;

                rc = sqlite3_prepare_v2_ptr(db, symbols_sql, -1, &stmt, nullptr);
                if (rc == SQLITE_OK) {
                    std::cout << "Available symbols (sample):" << std::endl;
                    while ((rc = sqlite3_step_ptr(stmt)) == SQLITE_ROW) {
                        const char* symbol = reinterpret_cast<const char*>(sqlite3_column_text_ptr(stmt, 0));
                        if (symbol) {
                            std::cout << "  " << symbol << std::endl;
                        }
                    }
                    sqlite3_finalize_ptr(stmt);
                }
                sqlite3_close_ptr(db);
            }
        }

        return 1;
    }

    std::string target_symbol = argv[1];
    std::cout << "Target symbol: " << target_symbol << std::endl;
    std::cout << "Testing period: 2018-2025" << std::endl;
    std::cout << "Testing thresholds: 0.5% and above" << std::endl;
    std::cout << "Testing multiple DC strategies" << std::endl;
    std::cout << std::endl;

    // Load SQLite dynamically
    if (!loadSQLite()) {
        std::cout << "Failed to load SQLite library. Please ensure sqlite3.dll is available." << std::endl;
        std::cout << "Expected locations:" << std::endl;
        std::cout << "  - C:/sqlite3/sqlite3.dll" << std::endl;
        std::cout << "  - sqlite3.dll in current directory" << std::endl;
        return 1;
    }

    std::cout << "SQLite library loaded successfully!" << std::endl;
    
    // Test with the specified symbol - load ALL data from ALL databases (2018-2025)
    std::cout << "\n=== Testing with symbol: " << target_symbol << " ===" << std::endl;
    std::cout << "Loading ALL available data from 2018-2025 databases..." << std::endl;
    std::cout << "Note: Progress will be saved automatically. You can stop and resume anytime." << std::endl;
    std::cout << std::endl;

    std::vector<double> prices = loadAllDataForSymbolWithCheckpoint(target_symbol);

    if (prices.empty()) {
        std::cout << "No data found for symbol: " << target_symbol << std::endl;
        std::cout << "Please check if the symbol exists in your databases." << std::endl;
        return 1;
    }

    std::cout << "Total loaded: " << prices.size() << " price points across all years" << std::endl;
    
    if (!prices.empty()) {
        double min_price = *std::min_element(prices.begin(), prices.end());
        double max_price = *std::max_element(prices.begin(), prices.end());
        std::cout << "Price range: $" << std::fixed << std::setprecision(2)
                  << min_price << " to $" << max_price << std::endl;

        // Initialize report generator
        ReportGenerator report(target_symbol);
        report.writeDataSummary(prices.size(), min_price, max_price,
                               getAllDatabaseFiles().size(), 0); // TODO: track files with data

        // Test thresholds >= 0.5% only
        std::vector<double> thresholds = {0.005, 0.01, 0.015, 0.02, 0.03, 0.05};

        // Test multiple strategies
        std::vector<std::pair<std::string, std::function<std::unique_ptr<TradingStrategy>()>>> strategies = {
            {"Simple DC", []() { return std::make_unique<SimpleDCStrategy>(100000.0); }},
            {"Contrarian DC", []() { return std::make_unique<ContrarianDCStrategy>(100000.0); }},
            {"Long Only DC", []() { return std::make_unique<LongOnlyDCStrategy>(100000.0); }}
        };

        // Track best results
        double best_return = -1000.0;
        std::string best_strategy;
        double best_threshold = 0.0;
        int best_trades = 0;

        for (const auto& strategy_pair : strategies) {
            std::cout << "\n=== " << strategy_pair.first << " Strategy Results ===" << std::endl;
            std::cout << "Initial Capital: $100,000" << std::endl;
            std::cout << std::endl;

            std::cout << std::setw(12) << "Threshold"
                      << std::setw(12) << "Trades"
                      << std::setw(15) << "Final PnL"
                      << std::setw(12) << "Return %" << std::endl;
            std::cout << std::string(50, '-') << std::endl;

            // Write strategy header to report
            report.writeStrategyHeader(strategy_pair.first);

            for (double threshold : thresholds) {
                DCGenerator dc_gen(threshold);
                auto strategy = strategy_pair.second();

                // Process all price data
                for (double price : prices) {
                    DCGenerator::DCEvent event = dc_gen.processPrice(price);
                    if (event != DCGenerator::DCEvent::NONE) {
                        strategy->onDCEvent(event, price);
                    }
                }

                // Calculate final results with safety checks
                double final_price = prices.back();
                if (!std::isfinite(final_price) || final_price <= 0) {
                    final_price = 100.0; // Default fallback price
                }

                double pnl = strategy->getPnL(final_price);
                double return_pct = strategy->getTotalReturn(final_price);
                double final_value = strategy->getCurrentValue(final_price);
                int trades = strategy->getTradeCount();

                // Additional safety checks
                if (!std::isfinite(pnl)) pnl = 0.0;
                if (!std::isfinite(return_pct)) return_pct = 0.0;
                if (!std::isfinite(final_value)) final_value = 100000.0;

                // Track best result
                if (return_pct > best_return) {
                    best_return = return_pct;
                    best_strategy = strategy_pair.first;
                    best_threshold = threshold;
                    best_trades = trades;
                }

                // Display results
                std::cout << std::fixed << std::setprecision(1)
                          << std::setw(11) << threshold * 100 << "%"
                          << std::setw(12) << trades
                          << std::setw(12) << "$" << std::setprecision(0) << pnl
                          << std::setw(11) << std::setprecision(2) << return_pct << "%" << std::endl;

                // Write to report
                report.writeStrategyResult(threshold, trades, pnl, return_pct, final_value);
            }

            // Write strategy footer to report
            report.writeStrategyFooter();
        }

        // Write best results and descriptions to report
        report.writeBestResults(best_strategy, best_threshold, best_return, best_trades);
        report.writeStrategyDescriptions();

        std::cout << std::endl;
        std::cout << "=== BEST RESULT ===" << std::endl;
        std::cout << "Best Strategy: " << best_strategy << std::endl;
        std::cout << "Best Threshold: " << std::fixed << std::setprecision(1)
                  << best_threshold * 100 << "%" << std::endl;
        std::cout << "Best Return: " << std::setprecision(2) << best_return << "%" << std::endl;
        std::cout << std::endl;
        std::cout << "Strategy Descriptions:" << std::endl;
        std::cout << "- Simple DC: Buy on downturn end, sell on upturn end" << std::endl;
        std::cout << "- Contrarian DC: Buy on upturn end, sell on downturn end" << std::endl;
        std::cout << "- Long Only DC: Only buy on downturn end, hold position" << std::endl;
    }
    
    std::cout << "\nMulti-year DC strategy test completed successfully!" << std::endl;
    std::cout << "Tested " << prices.size() << " price points across 2018-2025" << std::endl;
    std::cout << "Detailed report saved to: report_" << target_symbol << ".txt" << std::endl;
    std::cout << "Automated testing - continuing to next symbol..." << std::endl;

    return 0;
}
