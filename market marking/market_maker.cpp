/**
 * Queue-Theory Based Market Making Backtester
 * 
 * Quote AAPL using limit orders, hedge with MSFT
 * Uses Poisson-based fill probability model
 */

#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <iomanip>
#include <deque>
#include <cstdint>
#include <chrono>

// Quote data structure
struct Quote {
    int64_t timestamp;      // nanoseconds since epoch
    double bid_price;
    int32_t bid_size;
    double ask_price;
    int32_t ask_size;
    int32_t bid_exchange;
};

// Order structure
struct Order {
    bool is_bid;            // true = buy, false = sell
    double price;
    int32_t size;
    int32_t queue_position; // lots ahead when placed
    int64_t place_time;
    bool filled;
};

// Trade record
struct Trade {
    int64_t timestamp;
    bool is_buy;
    double price;
    int32_t size;
    double pnl;
};

// Configuration parameters
struct Config {
    double tau = 0.5;                    // Fill probability horizon (seconds)
    double gamma = 1e-10;                // Inventory penalty coefficient
    double hedge_ratio = 0.8;            // Beta for hedging (AAPL vs MSFT)
    int max_inventory = 1000;            // Maximum shares position
    int order_size = 100;                // Shares per order
    double tick_size = 0.01;             // Price tick
    int rate_window_ns = 60000000000LL;  // 60 second window for rate estimation
    double adverse_cost_pct = 0.0001;    // 1 basis point adverse selection
};

// Rate estimation window
struct RateWindow {
    std::deque<int64_t> events;
    int64_t window_ns;
    
    RateWindow(int64_t window = 60000000000LL) : window_ns(window) {}
    
    void add_event(int64_t ts) {
        events.push_back(ts);
        // Prune old events
        while (!events.empty() && events.front() < ts - window_ns) {
            events.pop_front();
        }
    }
    
    double get_rate() const {
        if (events.size() < 2) return 0.1; // Default rate
        int64_t span = events.back() - events.front();
        if (span <= 0) return 0.1;
        return (double)(events.size() - 1) / (span / 1e9); // Events per second
    }
};

// Market maker state
struct MarketMaker {
    Config cfg;
    
    // Positions
    int aapl_position = 0;
    int msft_position = 0;
    double cash = 0.0;
    
    // Current orders
    Order bid_order;
    Order ask_order;
    bool has_bid_order = false;
    bool has_ask_order = false;
    
    // Rate estimation
    RateWindow bid_hit_rate;
    RateWindow ask_hit_rate;
    RateWindow bid_cancel_rate;
    RateWindow ask_cancel_rate;
    
    // Previous quotes for change detection
    Quote prev_aapl;
    Quote prev_msft;
    bool has_prev = false;
    
    // Statistics
    std::vector<Trade> trades;
    double total_pnl = 0.0;
    double hedge_cost = 0.0;
    int num_fills = 0;
    int num_bid_fills = 0;
    int num_ask_fills = 0;
    double max_position = 0;
    double min_position = 0;
    double gross_spread_captured = 0.0;  // Total spread earned before costs
    double total_traded_notional = 0.0;  // Total notional traded
    double hedge_residual = 0.0;         // Fractional hedge shares accumulator
    
    // Poisson CDF for fill probability
    double poisson_cdf(int k, double lambda) {
        if (lambda <= 0) return 0.0;
        double sum = 0.0;
        double term = exp(-lambda);
        for (int i = 0; i <= k; i++) {
            sum += term;
            term *= lambda / (i + 1);
        }
        return sum;
    }
    
    // Calculate fill probability
    double calc_fill_prob(int queue_pos, double delta, double tau) {
        if (queue_pos <= 0) return 1.0 - exp(-delta * tau);
        double lambda = delta * tau;
        return 1.0 - poisson_cdf(queue_pos - 1, lambda);
    }
    
    // Calculate expected value of quoting
    double calc_expected_value(double fill_prob, double edge, double adverse_cost,
                               double hedge_cost, double exposure) {
        double inv_penalty = cfg.gamma * exposure * exposure;
        return fill_prob * (edge - adverse_cost - hedge_cost) - inv_penalty;
    }

    // Process a new quote update
    void process_quote(const Quote& aapl, const Quote& msft) {
        if (!has_prev) {
            prev_aapl = aapl;
            prev_msft = msft;
            has_prev = true;
            return;
        }

        // Detect queue depletion events (bid lifted or ask hit)
        if (aapl.bid_price > prev_aapl.bid_price ||
            (aapl.bid_price == prev_aapl.bid_price && aapl.bid_size < prev_aapl.bid_size)) {
            bid_hit_rate.add_event(aapl.timestamp);
        }
        if (aapl.ask_price < prev_aapl.ask_price ||
            (aapl.ask_price == prev_aapl.ask_price && aapl.ask_size < prev_aapl.ask_size)) {
            ask_hit_rate.add_event(aapl.timestamp);
        }

        // Check if our orders got filled (simplified: if price moved through us)
        check_fills(aapl, msft);

        // Update quotes
        update_quotes(aapl, msft);

        prev_aapl = aapl;
        prev_msft = msft;
    }

    void check_fills(const Quote& aapl, const Quote& msft) {
        // Simulate fills using queue-based probability model
        // More conservative: require price to actually trade through or queue to fully deplete

        const double MIN_TIME_IN_QUEUE = 0.1;  // Minimum 100ms before fill possible

        if (has_bid_order && !bid_order.filled) {
            bool fill = false;
            double time_in_queue = (aapl.timestamp - bid_order.place_time) / 1e9;

            // Case 1: Price traded through (definite fill) - ask dropped below our bid
            if (aapl.ask_price < bid_order.price - 0.005) {
                fill = true;
            }
            // Case 2: Best bid moved significantly above our price (market moved up, we got filled)
            else if (aapl.bid_price > bid_order.price + 0.02 && time_in_queue > MIN_TIME_IN_QUEUE) {
                fill = true;
            }
            // Case 3: Queue-based fill - only if we've been in queue long enough
            else if (time_in_queue > MIN_TIME_IN_QUEUE &&
                     std::abs(aapl.bid_price - bid_order.price) < 0.005) {
                double delta = bid_hit_rate.get_rate() + 0.2;
                double fill_prob = calc_fill_prob(bid_order.queue_position / cfg.order_size, delta, time_in_queue);

                // Only fill if probability is very high (queue nearly depleted)
                if (fill_prob > 0.95) {
                    fill = true;
                }
            }

            if (fill) {
                execute_fill(true, bid_order.price, bid_order.size, aapl, msft);
                has_bid_order = false;
            }
        }

        if (has_ask_order && !ask_order.filled) {
            bool fill = false;
            double time_in_queue = (aapl.timestamp - ask_order.place_time) / 1e9;

            // Case 1: Price traded through - bid rose above our ask
            if (aapl.bid_price > ask_order.price + 0.005) {
                fill = true;
            }
            // Case 2: Best ask moved significantly below our price
            else if (aapl.ask_price < ask_order.price - 0.02 && time_in_queue > MIN_TIME_IN_QUEUE) {
                fill = true;
            }
            // Case 3: Queue-based fill
            else if (time_in_queue > MIN_TIME_IN_QUEUE &&
                     std::abs(aapl.ask_price - ask_order.price) < 0.005) {
                double delta = ask_hit_rate.get_rate() + 0.2;
                double fill_prob = calc_fill_prob(ask_order.queue_position / cfg.order_size, delta, time_in_queue);

                if (fill_prob > 0.95) {
                    fill = true;
                }
            }

            if (fill) {
                execute_fill(false, ask_order.price, ask_order.size, aapl, msft);
                has_ask_order = false;
            }
        }
    }

    void execute_fill(bool is_buy, double price, int size, const Quote& aapl, const Quote& msft) {
        double fill_value = price * size;

        if (is_buy) {
            aapl_position += size;
            cash -= fill_value;
            num_bid_fills++;
        } else {
            aapl_position -= size;
            cash += fill_value;
            num_ask_fills++;
        }

        // Delta-neutral hedge with MSFT
        // Target: MSFT position should offset AAPL notional exposure
        // target_msft_position = -hedge_ratio * aapl_position * aapl_mid / msft_mid
        double aapl_mid = (aapl.bid_price + aapl.ask_price) / 2.0;
        double msft_mid = (msft.bid_price + msft.ask_price) / 2.0;
        if (msft_mid < 1.0) {
            std::cerr << "Warning: Invalid MSFT mid price " << msft_mid << ", skipping hedge\n";
        } else {
            // Calculate target MSFT position based on current AAPL position
            int target_msft = (int)std::round(-cfg.hedge_ratio * aapl_position * aapl_mid / msft_mid);
            int hedge_shares = target_msft - msft_position;

            // Sanity check hedge size
            if (std::abs(hedge_shares) > 0 && std::abs(hedge_shares) < 10000) {
                if (hedge_shares < 0) {
                    // Sell MSFT
                    int shares_to_sell = -hedge_shares;
                    msft_position -= shares_to_sell;
                    double hedge_price = msft.bid_price;  // Sell at bid
                    cash += shares_to_sell * hedge_price;
                    hedge_cost += shares_to_sell * (msft_mid - hedge_price);
                } else {
                    // Buy MSFT
                    int shares_to_buy = hedge_shares;
                    msft_position += shares_to_buy;
                    double hedge_price = msft.ask_price;  // Buy at ask
                    cash -= shares_to_buy * hedge_price;
                    hedge_cost += shares_to_buy * (hedge_price - msft_mid);
                }
            }
        }

        num_fills++;
        max_position = std::max(max_position, (double)aapl_position);
        min_position = std::min(min_position, (double)aapl_position);
        total_traded_notional += fill_value;

        Trade t;
        t.timestamp = aapl.timestamp;
        t.is_buy = is_buy;
        t.price = price;
        t.size = size;
        trades.push_back(t);
    }

    void update_quotes(const Quote& aapl, const Quote& msft) {
        double mid = (aapl.bid_price + aapl.ask_price) / 2.0;
        double spread = aapl.ask_price - aapl.bid_price;

        // Calculate exposure (net notional)
        double msft_mid = (msft.bid_price + msft.ask_price) / 2.0;
        double exposure = aapl_position * mid - cfg.hedge_ratio * msft_position * msft_mid;

        // Get depletion rates
        double delta_bid = bid_hit_rate.get_rate() + 0.5; // add baseline
        double delta_ask = ask_hit_rate.get_rate() + 0.5;

        // Inventory skew - push quotes to reduce inventory
        double inv_skew = aapl_position * cfg.gamma * mid * 100;  // Skew in price terms

        // Queue position estimates
        int queue_join_bid = aapl.bid_size * 100;  // Approximate queue in shares
        int queue_join_ask = aapl.ask_size * 100;

        // Calculate fill probabilities
        double p_join_bid = calc_fill_prob(queue_join_bid / cfg.order_size, delta_bid, cfg.tau);
        double p_impr_bid = calc_fill_prob(0, delta_bid, cfg.tau);
        double p_join_ask = calc_fill_prob(queue_join_ask / cfg.order_size, delta_ask, cfg.tau);
        double p_impr_ask = calc_fill_prob(0, delta_ask, cfg.tau);

        // Edge calculation
        double edge = spread / 2.0;
        double msft_spread = msft.ask_price - msft.bid_price;
        double hedge_cost_est = msft_spread / 2.0 * cfg.hedge_ratio;
        double adverse_cost = mid * cfg.adverse_cost_pct;

        // Net edge after costs
        double net_edge = edge - hedge_cost_est - adverse_cost;

        // Decision: improve if queue is too deep, otherwise join
        bool improve_bid = (p_join_bid < 0.1 && p_impr_bid > 0.3);
        bool improve_ask = (p_join_ask < 0.1 && p_impr_ask > 0.3);

        // Always quote if within inventory limits and spread is reasonable
        bool should_quote = (spread >= 0.01 && spread <= 0.10);  // 1-10 cent spread

        // Place bid order
        if (should_quote && aapl_position < cfg.max_inventory && !has_bid_order) {
            bid_order.is_bid = true;
            bid_order.size = cfg.order_size;
            bid_order.place_time = aapl.timestamp;
            bid_order.filled = false;

            if (improve_bid) {
                bid_order.price = aapl.bid_price + cfg.tick_size;
                bid_order.queue_position = 0;
            } else {
                bid_order.price = aapl.bid_price;
                bid_order.queue_position = queue_join_bid;
            }
            // Apply inventory skew
            bid_order.price -= inv_skew;
            has_bid_order = true;
        }

        // Place ask order
        if (should_quote && aapl_position > -cfg.max_inventory && !has_ask_order) {
            ask_order.is_bid = false;
            ask_order.size = cfg.order_size;
            ask_order.place_time = aapl.timestamp;
            ask_order.filled = false;

            if (improve_ask) {
                ask_order.price = aapl.ask_price - cfg.tick_size;
                ask_order.queue_position = 0;
            } else {
                ask_order.price = aapl.ask_price;
                ask_order.queue_position = queue_join_ask;
            }
            // Apply inventory skew
            ask_order.price -= inv_skew;
            has_ask_order = true;
        }
    }

    void print_results(const Quote& final_aapl, const Quote& final_msft) {
        double aapl_mid = (final_aapl.bid_price + final_aapl.ask_price) / 2.0;
        double msft_mid = (final_msft.bid_price + final_msft.ask_price) / 2.0;

        // Mark to market
        double aapl_mtm = aapl_position * aapl_mid;
        double msft_mtm = msft_position * msft_mid;
        double total_mtm = cash + aapl_mtm + msft_mtm;

        // Calculate round-trip trades (min of buys and sells)
        int round_trips = std::min(num_bid_fills, num_ask_fills);

        std::cout << "\n========== BACKTEST RESULTS ==========\n";
        std::cout << std::fixed << std::setprecision(2);
        std::cout << "\nTRADING STATISTICS:\n";
        std::cout << "  Total fills: " << num_fills << "\n";
        std::cout << "  Bid fills (buys): " << num_bid_fills << "\n";
        std::cout << "  Ask fills (sells): " << num_ask_fills << "\n";
        std::cout << "  Round-trip trades: " << round_trips << "\n";
        std::cout << "  Total notional traded: $" << total_traded_notional << "\n";
        std::cout << "\nPOSITION:\n";
        std::cout << "  Final AAPL position: " << aapl_position << " shares\n";
        std::cout << "  Final MSFT position: " << msft_position << " shares\n";
        std::cout << "  Max AAPL position: " << max_position << "\n";
        std::cout << "  Min AAPL position: " << min_position << "\n";
        std::cout << "\nP&L BREAKDOWN:\n";
        std::cout << "  Cash: $" << cash << "\n";
        std::cout << "  AAPL MTM: $" << aapl_mtm << "\n";
        std::cout << "  MSFT MTM: $" << msft_mtm << "\n";
        std::cout << "  Hedge cost (slippage): $" << hedge_cost << "\n";
        std::cout << "  ----------------------------\n";
        std::cout << "  Total P&L (MTM): $" << total_mtm << "\n";
        std::cout << "\nPER-TRADE METRICS:\n";
        if (num_fills > 0) {
            std::cout << "  Avg P&L per fill: $" << total_mtm / num_fills << "\n";
            std::cout << "  Avg notional per fill: $" << total_traded_notional / num_fills << "\n";
            std::cout << "  P&L as % of notional: " << (total_mtm / total_traded_notional * 100) << "%\n";
        }
        if (round_trips > 0) {
            std::cout << "  Avg P&L per round-trip: $" << total_mtm / round_trips << "\n";
        }
        std::cout << "\nHEDGE ANALYSIS:\n";
        std::cout << "  Hedge cost as % of notional: " << (hedge_cost / total_traded_notional * 100) << "%\n";
        std::cout << "======================================\n";
    }
};

// Load binary quote data
std::vector<Quote> load_quotes(const char* filename) {
    std::vector<Quote> quotes;
    std::ifstream file(filename, std::ios::binary);

    if (!file) {
        std::cerr << "Error: Cannot open " << filename << std::endl;
        return quotes;
    }

    uint64_t num_records;
    file.read(reinterpret_cast<char*>(&num_records), sizeof(num_records));
    quotes.reserve(num_records);

    std::cout << "Loading " << num_records << " quotes from " << filename << "...\n";

    for (uint64_t i = 0; i < num_records; i++) {
        Quote q;
        file.read(reinterpret_cast<char*>(&q.timestamp), sizeof(q.timestamp));
        file.read(reinterpret_cast<char*>(&q.bid_price), sizeof(q.bid_price));
        file.read(reinterpret_cast<char*>(&q.bid_size), sizeof(q.bid_size));
        file.read(reinterpret_cast<char*>(&q.ask_price), sizeof(q.ask_price));
        file.read(reinterpret_cast<char*>(&q.ask_size), sizeof(q.ask_size));
        file.read(reinterpret_cast<char*>(&q.bid_exchange), sizeof(q.bid_exchange));
        quotes.push_back(q);
    }

    std::cout << "Loaded " << quotes.size() << " quotes.\n";

    // Debug: print first few quotes
    if (quotes.size() > 0) {
        std::cout << "  First quote: ts=" << quotes[0].timestamp
                  << " bid=" << quotes[0].bid_price
                  << " ask=" << quotes[0].ask_price << "\n";
    }
    if (quotes.size() > 1) {
        std::cout << "  Second quote: ts=" << quotes[1].timestamp
                  << " bid=" << quotes[1].bid_price
                  << " ask=" << quotes[1].ask_price << "\n";
    }

    return quotes;
}

// Merge two quote streams by timestamp
struct MergedQuote {
    Quote aapl;
    Quote msft;
    int64_t timestamp;
};

std::vector<MergedQuote> merge_quotes(const std::vector<Quote>& aapl,
                                       const std::vector<Quote>& msft) {
    std::vector<MergedQuote> merged;

    size_t i = 0, j = 0;
    Quote last_aapl = aapl[0];
    Quote last_msft = msft[0];

    // Sample every N quotes to speed up backtest
    const size_t sample_rate = 10;  // Process every 10th quote update for more fills
    size_t count = 0;

    while (i < aapl.size() && j < msft.size()) {
        MergedQuote mq;

        if (aapl[i].timestamp <= msft[j].timestamp) {
            last_aapl = aapl[i];
            i++;
        } else {
            last_msft = msft[j];
            j++;
        }

        count++;
        if (count % sample_rate == 0) {
            mq.aapl = last_aapl;
            mq.msft = last_msft;
            mq.timestamp = std::max(last_aapl.timestamp, last_msft.timestamp);
            merged.push_back(mq);
        }
    }

    std::cout << "Merged into " << merged.size() << " time points.\n";
    return merged;
}

int main() {
    std::cout << "=== Queue-Theory Market Making Backtester ===\n";
    std::cout << "Instrument: AAPL (quote) + MSFT (hedge)\n\n";

    // Load data
    auto aapl_quotes = load_quotes("aapl_quotes.bin");
    auto msft_quotes = load_quotes("msft_quotes.bin");

    if (aapl_quotes.empty() || msft_quotes.empty()) {
        std::cerr << "Failed to load data.\n";
        return 1;
    }

    // Merge streams
    auto merged = merge_quotes(aapl_quotes, msft_quotes);

    // ========== PARAMETER SENSITIVITY ANALYSIS ==========
    std::cout << "\n========== PARAMETER SENSITIVITY ANALYSIS ==========\n\n";

    // Parameter ranges to test
    std::vector<double> hedge_ratios = {0.0, 0.2, 0.3, 0.4, 0.48, 0.6, 0.8, 1.0};
    std::vector<double> taus = {1.0, 2.0, 4.0, 6.0, 8.0, 10.0, 15.0, 20.0};
    std::vector<double> gammas = {0.0, 1e-9, 2.5e-9, 5e-9, 1e-8, 2.5e-8, 5e-8, 1e-7};

    // Base parameters
    double base_hedge = 0.48;
    double base_tau = 8.0;
    double base_gamma = 2.5e-9;

    // Helper lambda to run backtest
    auto run_backtest = [&](double hedge, double tau, double gamma) -> std::tuple<double, int, double> {
        MarketMaker mm;
        mm.cfg.hedge_ratio = hedge;
        mm.cfg.tau = tau;
        mm.cfg.gamma = gamma;
        mm.cfg.max_inventory = 200;
        mm.cfg.order_size = 100;
        mm.cfg.adverse_cost_pct = 0.0001;

        for (size_t i = 0; i < merged.size(); i++) {
            mm.process_quote(merged[i].aapl, merged[i].msft);
        }

        double aapl_mid = (merged.back().aapl.bid_price + merged.back().aapl.ask_price) / 2.0;
        double msft_mid = (merged.back().msft.bid_price + merged.back().msft.ask_price) / 2.0;
        double pnl = mm.cash + mm.aapl_position * aapl_mid + mm.msft_position * msft_mid;
        return {pnl, mm.num_fills, mm.hedge_cost};
    };

    // 1. HEDGE RATIO SENSITIVITY
    std::cout << "1. HEDGE RATIO (beta) SENSITIVITY\n";
    std::cout << "   (tau=" << base_tau << "s, gamma=" << base_gamma << ")\n";
    std::cout << "   -----------------------------------------\n";
    std::cout << "   Beta    |    P&L ($)   | Fills     | Hedge Cost\n";
    std::cout << "   -----------------------------------------\n";

    for (double h : hedge_ratios) {
        auto [pnl, fills, hcost] = run_backtest(h, base_tau, base_gamma);
        std::cout << "   " << std::fixed << std::setprecision(2) << h
                  << "    | " << std::setw(12) << std::setprecision(0) << pnl
                  << " | " << std::setw(9) << fills
                  << " | $" << std::setprecision(0) << hcost << "\n";
    }

    // 2. TAU SENSITIVITY
    std::cout << "\n2. FILL HORIZON (tau) SENSITIVITY\n";
    std::cout << "   (beta=" << base_hedge << ", gamma=" << base_gamma << ")\n";
    std::cout << "   -----------------------------------------\n";
    std::cout << "   Tau (s) |    P&L ($)   | Fills     | Hedge Cost\n";
    std::cout << "   -----------------------------------------\n";

    for (double t : taus) {
        auto [pnl, fills, hcost] = run_backtest(base_hedge, t, base_gamma);
        std::cout << "   " << std::fixed << std::setprecision(1) << std::setw(5) << t
                  << "  | " << std::setw(12) << std::setprecision(0) << pnl
                  << " | " << std::setw(9) << fills
                  << " | $" << std::setprecision(0) << hcost << "\n";
    }

    // 3. GAMMA SENSITIVITY
    std::cout << "\n3. INVENTORY PENALTY (gamma) SENSITIVITY\n";
    std::cout << "   (beta=" << base_hedge << ", tau=" << base_tau << "s)\n";
    std::cout << "   -----------------------------------------\n";
    std::cout << "   Gamma     |    P&L ($)   | Fills     | Hedge Cost\n";
    std::cout << "   -----------------------------------------\n";

    for (double g : gammas) {
        auto [pnl, fills, hcost] = run_backtest(base_hedge, base_tau, g);
        std::cout << "   " << std::scientific << std::setprecision(1) << g
                  << " | " << std::fixed << std::setw(12) << std::setprecision(0) << pnl
                  << " | " << std::setw(9) << fills
                  << " | $" << std::setprecision(0) << hcost << "\n";
    }

    // 4. GRID SEARCH - Find optimal combination
    std::cout << "\n4. TOP 10 PARAMETER COMBINATIONS (Grid Search)\n";
    std::cout << "   --------------------------------------------------------\n";
    std::cout << "   Beta  | Tau  | Gamma    |    P&L ($)   | Fills\n";
    std::cout << "   --------------------------------------------------------\n";

    std::vector<std::tuple<double, double, double, double, int>> results;

    // Coarse grid for speed
    std::vector<double> h_grid = {0.0, 0.3, 0.48, 0.6, 0.8};
    std::vector<double> t_grid = {2.0, 4.0, 8.0, 12.0};
    std::vector<double> g_grid = {0.0, 1e-9, 5e-9, 1e-8};

    int total_combos = h_grid.size() * t_grid.size() * g_grid.size();
    int combo = 0;

    for (double h : h_grid) {
        for (double t : t_grid) {
            for (double g : g_grid) {
                combo++;
                std::cout << "\r   Searching: " << combo << "/" << total_combos << "...   " << std::flush;
                auto [pnl, fills, hcost] = run_backtest(h, t, g);
                results.push_back({h, t, g, pnl, fills});
            }
        }
    }

    // Sort by P&L descending
    std::sort(results.begin(), results.end(),
              [](const auto& a, const auto& b) { return std::get<3>(a) > std::get<3>(b); });

    std::cout << "\r                                      \r";

    for (int i = 0; i < 10 && i < results.size(); i++) {
        auto [h, t, g, pnl, fills] = results[i];
        std::cout << "   " << std::fixed << std::setprecision(2) << h
                  << "  | " << std::setprecision(1) << std::setw(4) << t
                  << " | " << std::scientific << std::setprecision(0) << g
                  << " | " << std::fixed << std::setw(12) << std::setprecision(0) << pnl
                  << " | " << std::setw(9) << fills << "\n";
    }

    std::cout << "\n=======================================================\n";

    return 0;
}

