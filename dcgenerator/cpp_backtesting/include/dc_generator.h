#pragma once

#include "types.h"
#include <vector>
#include <memory>

namespace dcbacktest {

class DCGenerator {
public:
    explicit DCGenerator(double threshold);
    ~DCGenerator() = default;

    // Process a new price and return the DC event (if any)
    DCEvent processPrice(Price price, Timestamp timestamp);
    
    // Get current DC state
    const DCState& getCurrentState() const { return state_; }
    
    // Reset the generator state
    void reset();
    
    // Get threshold
    double getThreshold() const { return threshold_; }
    
    // Set new threshold
    void setThreshold(double threshold) { threshold_ = threshold; }

private:
    double threshold_;
    DCState state_;
    bool initialized_;
    
    // Helper methods
    DCEvent checkForEvents(Price current_price);
    void updateState(Price current_price, DCEvent event);
};

class DCEventHistory {
public:
    struct DCEventRecord {
        Timestamp timestamp;
        Price price;
        DCEvent event;
        
        DCEventRecord(Timestamp ts, Price p, DCEvent e) 
            : timestamp(ts), price(p), event(e) {}
    };
    
    void addEvent(Timestamp timestamp, Price price, DCEvent event);
    const std::vector<DCEventRecord>& getEvents() const { return events_; }
    void clear() { events_.clear(); }
    
    // Get events within a time range
    std::vector<DCEventRecord> getEventsInRange(Timestamp start, Timestamp end) const;
    
    // Get the last N events
    std::vector<DCEventRecord> getLastEvents(size_t count) const;

private:
    std::vector<DCEventRecord> events_;
};

} // namespace dcbacktest
