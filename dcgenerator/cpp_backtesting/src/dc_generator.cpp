#include "dc_generator.h"
#include <algorithm>
#include <stdexcept>

namespace dcbacktest {

DCGenerator::DCGenerator(double threshold) 
    : threshold_(threshold), initialized_(false) {
    if (threshold <= 0.0 || threshold >= 1.0) {
        throw std::invalid_argument("DC threshold must be between 0 and 1");
    }
    reset();
}

DCEvent DCGenerator::processPrice(Price price, Timestamp timestamp) {
    if (price <= 0.0) {
        return DCEvent::NONE;
    }
    
    if (!initialized_) {
        state_.highest_price = price;
        state_.lowest_price = price;
        state_.is_upturn = true;
        state_.current_event = DCEvent::NONE;
        initialized_ = true;
        return DCEvent::NONE;
    }
    
    DCEvent event = checkForEvents(price);
    updateState(price, event);
    
    return event;
}

void DCGenerator::reset() {
    state_ = DCState();
    initialized_ = false;
}

DCEvent DCGenerator::checkForEvents(Price current_price) {
    if (state_.is_upturn) {
        // Currently in an upturn
        if (current_price <= state_.highest_price * (1.0 - threshold_)) {
            // Price dropped by threshold from peak - end of upturn
            return DCEvent::END_UPTURN;
        } else if (current_price > state_.highest_price) {
            // New high - continue upturn
            return DCEvent::NONE;
        }
    } else {
        // Currently in a downturn
        if (current_price >= state_.lowest_price * (1.0 + threshold_)) {
            // Price rose by threshold from trough - end of downturn
            return DCEvent::END_DOWNTURN;
        } else if (current_price < state_.lowest_price) {
            // New low - continue downturn
            return DCEvent::NONE;
        }
    }
    
    return DCEvent::NONE;
}

void DCGenerator::updateState(Price current_price, DCEvent event) {
    switch (event) {
        case DCEvent::END_UPTURN:
            state_.is_upturn = false;
            state_.lowest_price = current_price;
            state_.current_event = DCEvent::END_UPTURN;
            break;
            
        case DCEvent::END_DOWNTURN:
            state_.is_upturn = true;
            state_.highest_price = current_price;
            state_.current_event = DCEvent::END_DOWNTURN;
            break;
            
        default:
            // Update extremes during trend continuation
            if (state_.is_upturn && current_price > state_.highest_price) {
                state_.highest_price = current_price;
            } else if (!state_.is_upturn && current_price < state_.lowest_price) {
                state_.lowest_price = current_price;
            }
            state_.current_event = event;
            break;
    }
}

// DCEventHistory implementation

void DCEventHistory::addEvent(Timestamp timestamp, Price price, DCEvent event) {
    if (event != DCEvent::NONE) {
        events_.emplace_back(timestamp, price, event);
    }
}

std::vector<DCEventHistory::DCEventRecord> 
DCEventHistory::getEventsInRange(Timestamp start, Timestamp end) const {
    std::vector<DCEventRecord> result;
    
    for (const auto& event : events_) {
        if (event.timestamp >= start && event.timestamp <= end) {
            result.push_back(event);
        }
    }
    
    return result;
}

std::vector<DCEventHistory::DCEventRecord> 
DCEventHistory::getLastEvents(size_t count) const {
    if (count >= events_.size()) {
        return events_;
    }
    
    return std::vector<DCEventRecord>(events_.end() - count, events_.end());
}

} // namespace dcbacktest
