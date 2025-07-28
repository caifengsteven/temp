#include "backtesting_engine.h"
#include "market_data.h"
#include <iostream>
#include <iomanip>

namespace dcbacktest {

// SimpleDCStrategy implementation

SimpleDCStrategy::SimpleDCStrategy(double position_size_pct) 
    : position_size_pct_(position_size_pct)
    , initial_capital_(0.0)
    , has_position_(false)
    , current_order_id_(0) {}

void SimpleDCStrategy::onDCEvent(DCEvent event, Price price, Timestamp timestamp, 
                                OrderManager& order_manager) {
    switch (event) {
        case DCEvent::END_DOWNTURN:
            // Buy signal - market has turned up
            if (!has_position_) {
                double cash_to_use = initial_capital_ * position_size_pct_;
                Quantity quantity = cash_to_use / price;
                
                current_order_id_ = order_manager.submitOrder(
                    Side::BUY, OrderType::MARKET, price, quantity, timestamp);
                has_position_ = true;
                
                std::cout << "BUY signal at " << price << " (quantity: " << quantity << ")" << std::endl;
            }
            break;
            
        case DCEvent::END_UPTURN:
            // Sell signal - market has turned down
            if (has_position_) {
                // For simplicity, assume we know our position size
                // In reality, you'd track this more carefully
                const Order* order = order_manager.getOrder(current_order_id_);
                if (order && order->status == OrderStatus::FILLED) {
                    Quantity quantity = order->filled_quantity;
                    
                    order_manager.submitOrder(
                        Side::SELL, OrderType::MARKET, price, quantity, timestamp);
                    has_position_ = false;
                    
                    std::cout << "SELL signal at " << price << " (quantity: " << quantity << ")" << std::endl;
                }
            }
            break;
            
        default:
            // No action for other events
            break;
    }
}

void SimpleDCStrategy::onStart(const BacktestConfig& config) {
    initial_capital_ = config.initial_capital;
    has_position_ = false;
    current_order_id_ = 0;
}

// BacktestingEngine implementation

BacktestingEngine::BacktestingEngine(const std::string& orderbook_db_path, 
                                   const std::string& trades_db_path)
    : verbose_(false) {
    
    orderbook_reader_ = std::make_unique<OrderBookReader>(orderbook_db_path);
    trades_reader_ = std::make_unique<TradesReader>(trades_db_path);
    dc_generator_ = std::make_unique<DCGenerator>(0.001); // Default threshold
    dc_history_ = std::make_unique<DCEventHistory>();
    order_manager_ = std::make_unique<OrderManager>();
    perf_calculator_ = std::make_unique<PerformanceCalculator>(100000.0); // Default capital
}

PerformanceMetrics BacktestingEngine::runBacktest(const BacktestConfig& config, 
                                                 std::unique_ptr<TradingStrategy> strategy) {
    
    std::cout << "Starting backtest for symbol: " << config.symbol << std::endl;
    std::cout << "Period: " << performance_utils::timestampToString(config.start_time) 
              << " to " << performance_utils::timestampToString(config.end_time) << std::endl;
    
    // Initialize components
    dc_generator_->setThreshold(config.dc_threshold);
    dc_history_->clear();
    order_manager_->clear();
    order_manager_->setTradingFees(config.maker_fee_bps, config.taker_fee_bps);
    perf_calculator_ = std::make_unique<PerformanceCalculator>(config.initial_capital);
    
    // Initialize strategy
    strategy->onStart(config);
    
    // Load market data
    std::vector<OrderBookSnapshot> orderbook_data;
    std::vector<Trade> trades_data;
    
    if (config.use_orderbook_data) {
        std::cout << "Loading orderbook data..." << std::endl;
        orderbook_data = orderbook_reader_->readOrderBookForSymbol(
            config.symbol, config.start_time, config.end_time);
        std::cout << "Loaded " << orderbook_data.size() << " orderbook snapshots" << std::endl;
    }
    
    if (config.use_trades_data) {
        std::cout << "Loading trades data..." << std::endl;
        trades_data = trades_reader_->readTradesForSymbol(
            config.symbol, config.start_time, config.end_time);
        std::cout << "Loaded " << trades_data.size() << " trades" << std::endl;
    }
    
    // Check data quality
    DataQualityChecker quality_checker;
    auto quality_report = quality_checker.checkDataQuality(orderbook_data, trades_data);
    if (verbose_) {
        quality_checker.printQualityReport(quality_report);
    }
    
    // Merge and process market data
    MarketDataProcessor processor;
    auto market_events = processor.mergeMarketData(orderbook_data, trades_data);
    auto price_ticks = processor.extractPriceSeries(market_events);
    
    std::cout << "Processing " << market_events.size() << " market events..." << std::endl;
    
    // Main backtesting loop
    double current_cash = config.initial_capital;
    double current_position_value = 0.0;
    Price last_price = 0.0;
    
    size_t progress_counter = 0;
    size_t progress_interval = market_events.size() / 100; // 1% intervals
    
    for (const auto& event : market_events) {
        // Update progress
        if (verbose_ && progress_interval > 0 && ++progress_counter % progress_interval == 0) {
            logProgress(event.timestamp, config.start_time, config.end_time);
        }
        
        if (event.type == MarketDataProcessor::MarketEvent::ORDERBOOK_UPDATE && event.orderbook) {
            // Process orderbook update
            order_manager_->processMarketUpdate(*event.orderbook, event.timestamp);
            strategy->onMarketUpdate(*event.orderbook, event.timestamp, *order_manager_);
            
            // Update last price from mid-price
            Price mid_price = event.orderbook->getMidPrice();
            if (mid_price > 0) {
                last_price = mid_price;
                
                // Process DC events
                DCEvent dc_event = dc_generator_->processPrice(mid_price, event.timestamp);
                if (dc_event != DCEvent::NONE) {
                    dc_history_->addEvent(event.timestamp, mid_price, dc_event);
                    strategy->onDCEvent(dc_event, mid_price, event.timestamp, *order_manager_);
                }
            }
            
        } else if (event.type == MarketDataProcessor::MarketEvent::TRADE && event.trade) {
            // Process trade
            order_manager_->processTrade(*event.trade, event.timestamp);
            strategy->onTrade(*event.trade, event.timestamp, *order_manager_);
            
            last_price = event.trade->price;
            
            // Process DC events
            DCEvent dc_event = dc_generator_->processPrice(event.trade->price, event.timestamp);
            if (dc_event != DCEvent::NONE) {
                dc_history_->addEvent(event.timestamp, event.trade->price, dc_event);
                strategy->onDCEvent(dc_event, event.trade->price, event.timestamp, *order_manager_);
            }
        }
        
        // Update portfolio tracking
        // This is simplified - in reality you'd track positions more carefully
        auto filled_orders = order_manager_->getFilledOrders();
        // Calculate current position value based on filled orders and current price
        // For simplicity, we'll update periodically
        
        if (last_price > 0) {
            perf_calculator_->updatePortfolio(event.timestamp, current_cash, 
                                            current_position_value, last_price);
        }
    }
    
    // Calculate final performance metrics
    auto metrics = perf_calculator_->calculateMetrics();
    
    // Finalize strategy
    strategy->onEnd(metrics);
    
    std::cout << "Backtest completed!" << std::endl;
    
    // Print results
    if (verbose_) {
        performance_utils::printPerformanceReport(metrics, perf_calculator_->getTradeHistory());
        
        // Print DC events summary
        auto dc_events = dc_history_->getEvents();
        std::cout << "\nDC Events Summary:" << std::endl;
        std::cout << "Total DC events: " << dc_events.size() << std::endl;
        
        // Count event types
        int upturn_ends = 0, downturn_ends = 0;
        for (const auto& event : dc_events) {
            if (event.event == DCEvent::END_UPTURN) upturn_ends++;
            if (event.event == DCEvent::END_DOWNTURN) downturn_ends++;
        }
        std::cout << "End upturn events: " << upturn_ends << std::endl;
        std::cout << "End downturn events: " << downturn_ends << std::endl;
    }
    
    return metrics;
}

void BacktestingEngine::logProgress(Timestamp current_time, Timestamp start_time, Timestamp end_time) {
    auto total_duration = end_time - start_time;
    auto elapsed_duration = current_time - start_time;
    
    if (total_duration.count() > 0) {
        double progress = static_cast<double>(elapsed_duration.count()) / total_duration.count() * 100.0;
        std::cout << "Progress: " << std::fixed << std::setprecision(1) << progress << "%\r" << std::flush;
    }
}

} // namespace dcbacktest
