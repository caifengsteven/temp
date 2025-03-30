/**
 * @file VPINMonitorApp.cpp
 * @brief TBricks application for real-time VPIN calculation and alerting
 */



/*
Below is a C++ implementation of a VPIN (Volume-Synchronized Probability of Informed Trading) monitoring application within the TBricks framework. 
The application calculates real-time VPIN for specified stocks and generates alerts when VPIN exceeds a configurable threshold.




    * The application is designed to be run within the TBricks trading platform and uses TBricks API for market data and order management.
    * It includes features such as parameter validation, trade stream handling, VPIN calculation, and alert generation.
    * The code is structured to handle multiple instruments and provides a user-friendly interface for configuration and monitoring.
    * 
    * @note This code is intended for educational purposes and may require modifications to work in a specific environment.
    * 
    * 
    * 
    * 
    * Implementation Details
Key Features

Real-Time VPIN Calculation:
Processes trade streams for each instrument
Classifies trades as buy or sell-initiated using the tick rule
Aggregates trades into volume buckets
Calculates VPIN using the formula: sum of absolute differences between consecutive bucket buy fractions divided by (number of buckets - 1)
Alert Generation:
Sends alerts via TBricks AlertService when VPIN exceeds a configurable threshold
Includes instrument information, VPIN value, and threshold in the alert
Implements a cooldown period to prevent alert flooding
Configurability:
Instruments to monitor
Bucket size (in shares)
Number of buckets for VPIN calculation
VPIN threshold for alerts
Alert cooldown period
Classes and Functions
VPINMonitorApp: Main application class that handles trade streams and VPIN calculation
ProcessTrade(): Processes individual trades and updates VPIN buckets
CalculateVPIN(): Calculates the VPIN metric based on bucket data
CheckAndAlertOnVPIN(): Checks if VPIN exceeds threshold and sends alerts
Data Structures
InstrumentData: Stores per-instrument data including:
Bucket history
Current bucket state
Last trade information (for tick rule)
Current VPIN value
Last alert timestamp (for cooldown)
Using the Application
Setup:
Deploy the application to your TBricks environment
Configure the instruments, bucket size, number of buckets, VPIN threshold, and alert cooldown
Inter-App Communication:
Other applications can listen for alerts with type "VPIN_ALERT"
The alert contains extra data parameters:
instrument_id: Identifier of the instrument
vpin_value: Current VPIN value
threshold: VPIN threshold that was exceeded
Monitoring:
VPIN values are logged for each instrument
Alerts are generated when VPIN exceeds the threshold
This implementation provides a solid foundation for VPIN monitoring and can be extended with additional features like historical VPIN analysis, visualization, or more sophisticated trade classification methods.

*/





 #include <iostream>
 #include <vector>
 #include <deque>
 #include <unordered_map>
 #include <mutex>
 #include <algorithm>
 #include <cmath>
 #include <numeric>
 
 #include "strategy/API.h"
 #include "strategy/Stream.h"
 #include "strategy/parameter/ParameterDefinition.h"
 #include "strategy/parameter/VolumeParameterDefinition.h"
 #include "strategy/parameter/PriceParameterDefinition.h"
 #include "strategy/parameter/IntegerParameterDefinition.h"
 #include "strategy/parameter/DoubleParameterDefinition.h"
 #include "strategy/parameter/StringParameterDefinition.h"
 #include "strategy/parameter/InstrumentParameterDefinition.h"
 #include "strategy/parameter/BooleanParameterDefinition.h"
 #include "strategy/parameter/ParameterDefinition.h"
 #include "strategy/type/Price.h"
 #include "strategy/type/Volume.h"
 #include "strategy/type/Integer.h"
 #include "strategy/type/Double.h"
 #include "strategy/type/String.h"
 #include "strategy/type/MIC.h"
 #include "strategy/type/Boolean.h"
 #include "strategy/InstrumentVenueIdentification.h"
 #include "strategy/service/AlertService.h"
 #include "strategy/stream/TradeStream.h"
 #include "strategy/type/TransactionState.h"
 #include "strategy/visualization/ViewerContext.h"
 #include "strategy/Instrument.h"
 #include "strategy/Types.h"
 
 namespace tbricks {
 
 /**
  * @class VPINMonitorApp
  * @brief Application for monitoring VPIN metrics in real-time
  * 
  * Calculates VPIN (Volume-Synchronized Probability of Informed Trading) for specified
  * instruments and generates alerts when VPIN exceeds a specified threshold.
  */
 class VPINMonitorApp : public tbricks::Application {
 public:
     VPINMonitorApp(const tbricks::InitializationReason & reason) 
       : tbricks::Application(reason),
         m_alertService(AlertService::GetInstance()),
         m_viewer(nullptr)
     {
         // Register parameters
         m_parameters.AddParameter(m_paramInstruments);
         m_parameters.AddParameter(m_paramBucketSize);
         m_parameters.AddParameter(m_paramNumBuckets);
         m_parameters.AddParameter(m_paramVPINThreshold);
         m_parameters.AddParameter(m_paramAlertCooldownSeconds);
 
         // Set default values
         m_bucketSize = Volume(10000); // Default bucket size: 10,000 shares
         m_numBuckets = 50;            // Default number of buckets: 50
         m_vpinThreshold = 0.8;        // Default VPIN threshold: 0.8 (80%)
         m_alertCooldownSeconds = 300; // Default cooldown between alerts: 5 minutes
 
         // Initialize state
         m_isActive = false;
     }
 
     virtual void HandleInit() override {
         m_viewer = GetViewer();
         if (m_viewer) {
             m_parameters.RegisterWithViewer(*m_viewer);
         }
     }
 
     void HandleStopWarmUp() override {
         // Check if we have parameters set from a previous run
         const StrategyParameters & params = GetParameters();
         
         tbricks::Variant instruments;
         if (params.GetParameter("Instruments", instruments)) {
             SetInstruments(InstrumentParameterDefinition::Get().GetInstruments(instruments));
         }
         
         tbricks::Variant bucketSize;
         if (params.GetParameter("BucketSize", bucketSize)) {
             m_bucketSize = VolumeParameterDefinition::Get().GetVolume(bucketSize);
         }
         
         tbricks::Variant numBuckets;
         if (params.GetParameter("NumBuckets", numBuckets)) {
             m_numBuckets = IntegerParameterDefinition::Get().GetInteger(numBuckets);
         }
         
         tbricks::Variant vpinThreshold;
         if (params.GetParameter("VPINThreshold", vpinThreshold)) {
             m_vpinThreshold = DoubleParameterDefinition::Get().GetDouble(vpinThreshold);
         }
         
         tbricks::Variant alertCooldown;
         if (params.GetParameter("AlertCooldownSeconds", alertCooldown)) {
             m_alertCooldownSeconds = IntegerParameterDefinition::Get().GetInteger(alertCooldown);
         }
     }
     
     virtual void HandleOrderBookStream(const StreamIdentifier & /* stream */,
                                     const OrderBook & /* orderBook */) override {}
 
     virtual void HandleTradeStream(const StreamIdentifier & stream,
                                  const Trade & trade) override {
         // Only process when active
         if (!m_isActive) return;
         
         // Get the instrument for this stream
         auto it = m_streamToInstrument.find(stream);
         if (it == m_streamToInstrument.end()) {
             return; // Unknown stream
         }
         
         InstrumentVenueIdentification ivi = it->second;
         std::string instrumentId = ivi.GetInstrument().GetIdentifier();
         
         // Process trade for VPIN calculation
         ProcessTrade(instrumentId, trade);
     }
 
     virtual void HandleValidateParameters(tbricks::ParameterValidationContext & context) override {
         std::vector<Instrument> instruments;
         Volume bucketSize;
         Integer numBuckets;
         Double vpinThreshold;
         Integer alertCooldown;
 
         if (!context.GetParameter(m_paramInstruments, instruments)) {
             context.RejectByDefinition(m_paramInstruments, "At least one instrument must be specified");
             return;
         }
 
         if (!context.GetParameter(m_paramBucketSize, bucketSize) || bucketSize <= Volume(0)) {
             context.RejectByDefinition(m_paramBucketSize, "Bucket size must be positive");
             return;
         }
 
         if (!context.GetParameter(m_paramNumBuckets, numBuckets) || numBuckets <= 0) {
             context.RejectByDefinition(m_paramNumBuckets, "Number of buckets must be positive");
             return;
         }
 
         if (!context.GetParameter(m_paramVPINThreshold, vpinThreshold) || 
             vpinThreshold <= 0.0 || vpinThreshold > 1.0) {
             context.RejectByDefinition(m_paramVPINThreshold, "VPIN threshold must be between 0 and 1");
             return;
         }
 
         if (!context.GetParameter(m_paramAlertCooldownSeconds, alertCooldown) || alertCooldown < 0) {
             context.RejectByDefinition(m_paramAlertCooldownSeconds, "Alert cooldown must be >= 0");
             return;
         }
     }
 
     virtual void HandleModifyParameters(const StrategyParameters & parameters) override {
         std::vector<Instrument> instruments;
         if (parameters.GetParameter(m_paramInstruments, instruments)) {
             SetInstruments(instruments);
         }
 
         Volume bucketSize;
         if (parameters.GetParameter(m_paramBucketSize, bucketSize)) {
             m_bucketSize = bucketSize;
             // Reset calculations as bucket size has changed
             ResetVPINCalculations();
         }
 
         Integer numBuckets;
         if (parameters.GetParameter(m_paramNumBuckets, numBuckets)) {
             m_numBuckets = numBuckets;
             // Reset calculations as number of buckets has changed
             ResetVPINCalculations();
         }
 
         Double vpinThreshold;
         if (parameters.GetParameter(m_paramVPINThreshold, vpinThreshold)) {
             m_vpinThreshold = vpinThreshold;
         }
 
         Integer alertCooldown;
         if (parameters.GetParameter(m_paramAlertCooldownSeconds, alertCooldown)) {
             m_alertCooldownSeconds = alertCooldown;
         }
     }
 
     virtual void HandleStart() override {
         m_isActive = true;
     }
 
     virtual void HandleStop() override {
         m_isActive = false;
     }
 
     virtual void HandleDelete() override {
         // Clean up resources
         for (auto & stream : m_tradeStreams) {
             stream.second.Close();
         }
         m_tradeStreams.clear();
         m_streamToInstrument.clear();
         m_instrumentData.clear();
     }
 
 private:
     // Set the instruments to monitor
     void SetInstruments(const std::vector<Instrument> & instruments) {
         std::lock_guard<std::mutex> lock(m_mutex);
         
         // Close existing streams
         for (auto & stream : m_tradeStreams) {
             stream.second.Close();
         }
         m_tradeStreams.clear();
         m_streamToInstrument.clear();
         
         // Open new streams
         for (const auto & instrument : instruments) {
             InstrumentVenueIdentification ivi(instrument);
             
             // Create trade stream for this instrument
             TradeStream::Options options;
             options.SetInstrument(ivi);
             
             TradeStream stream(options);
             if (stream.Open(*this)) {
                 std::string instrumentId = instrument.GetIdentifier();
                 
                 StreamIdentifier streamId = stream.GetIdentifier();
                 m_tradeStreams[instrumentId] = std::move(stream);
                 m_streamToInstrument[streamId] = ivi;
                 
                 // Initialize data structure for this instrument if it doesn't exist
                 if (m_instrumentData.find(instrumentId) == m_instrumentData.end()) {
                     m_instrumentData[instrumentId] = InstrumentData();
                     m_instrumentData[instrumentId].lastPrice = Price();
                 }
             }
         }
     }
 
     // Process a trade for VPIN calculation
     void ProcessTrade(const std::string & instrumentId, const Trade & trade) {
         std::lock_guard<std::mutex> lock(m_mutex);
         
         auto it = m_instrumentData.find(instrumentId);
         if (it == m_instrumentData.end()) {
             return; // Unknown instrument
         }
         
         InstrumentData & data = it->second;
         
         // Get trade details
         Price price = trade.GetPrice();
         Volume volume = trade.GetVolume();
         
         // Skip trades with zero volume
         if (volume == Volume(0)) {
             return;
         }
         
         // Determine if this is a buy or sell-initiated trade using tick rule
         bool isBuyInitiated = false;
         if (!data.lastPrice.Empty() && price > data.lastPrice) {
             isBuyInitiated = true;
         } else if (!data.lastPrice.Empty() && price < data.lastPrice) {
             isBuyInitiated = false;
         } else {
             // Price unchanged - use last direction or assume buy
             isBuyInitiated = data.lastTradeWasBuy;
         }
         
         // Update last price and direction
         data.lastPrice = price;
         data.lastTradeWasBuy = isBuyInitiated;
         
         // Add to current bucket
         if (isBuyInitiated) {
             data.currentBucketBuyVolume += volume;
         }
         data.currentBucketTotalVolume += volume;
         
         // Check if the current bucket is full
         if (data.currentBucketTotalVolume >= m_bucketSize) {
             // Calculate buy volume fraction for this bucket
             double buyFraction = (data.currentBucketTotalVolume > Volume(0)) ? 
                 data.currentBucketBuyVolume.GetDouble() / data.currentBucketTotalVolume.GetDouble() : 0.5;
             
             // Add to bucket list
             data.buckets.push_back(buyFraction);
             
             // Keep only the required number of buckets
             while (data.buckets.size() > m_numBuckets) {
                 data.buckets.pop_front();
             }
             
             // Reset current bucket
             data.currentBucketBuyVolume = Volume(0);
             data.currentBucketTotalVolume = Volume(0);
             
             // Calculate VPIN if we have enough buckets
             if (data.buckets.size() == m_numBuckets) {
                 double vpin = CalculateVPIN(data.buckets);
                 data.currentVPIN = vpin;
                 
                 // Check if we need to alert
                 CheckAndAlertOnVPIN(instrumentId, vpin);
                 
                 // Log VPIN value
                 tbricks::LogInfo("VPIN for %s: %.4f", instrumentId.c_str(), vpin);
             }
         }
     }
 
     // Calculate VPIN based on bucket data
     double CalculateVPIN(const std::deque<double> & buckets) {
         if (buckets.empty()) return 0.0;
         
         double sum = 0.0;
         for (size_t i = 0; i < buckets.size() - 1; ++i) {
             sum += std::abs(buckets[i] - buckets[i+1]);
         }
         
         return sum / (buckets.size() - 1);
     }
 
     // Check VPIN threshold and send alert if needed
     void CheckAndAlertOnVPIN(const std::string & instrumentId, double vpin) {
         auto & data = m_instrumentData[instrumentId];
         
         // Get current timestamp
         tbricks::DateTime now = tbricks::DateTime::Now();
         
         // Check if VPIN exceeds threshold and cooldown period has passed
         if (vpin >= m_vpinThreshold && 
             (data.lastAlertTime.Empty() || 
              (now - data.lastAlertTime).GetSeconds() >= m_alertCooldownSeconds)) {
             
             // Send alert
             std::string alertMessage = "VPIN Alert: " + instrumentId + " VPIN = " + 
                                        std::to_string(vpin) + " exceeds threshold " + 
                                        std::to_string(m_vpinThreshold.GetDouble());
             
             tbricks::AlertProperties props;
             props.SetAlertType("VPIN_ALERT");
             props.SetSeverity(tbricks::AlertSeverity::MEDIUM);
             props.SetText(alertMessage);
             props.SetSourceIdentifier("VPINMonitor");
             
             // Add instrument identifications
             auto instIt = m_tradeStreams.find(instrumentId);
             if (instIt != m_tradeStreams.end()) {
                 auto streamId = instIt->second.GetIdentifier();
                 auto it = m_streamToInstrument.find(streamId);
                 if (it != m_streamToInstrument.end()) {
                     props.SetInstrumentVenueIdentification(it->second);
                 }
             }
             
             // Add extra fields for other apps to process
             tbricks::StrategyParameters extraParams;
             extraParams.SetParameter("instrument_id", instrumentId);
             extraParams.SetParameter("vpin_value", vpin);
             extraParams.SetParameter("threshold", m_vpinThreshold);
             props.SetExtraData(extraParams);
             
             // Send the alert
             m_alertService.Send(props);
             
             // Update last alert time
             data.lastAlertTime = now;
             
             tbricks::LogInfo("VPIN Alert sent for %s: %.4f", instrumentId.c_str(), vpin);
         }
     }
 
     // Reset all VPIN calculations
     void ResetVPINCalculations() {
         std::lock_guard<std::mutex> lock(m_mutex);
         
         for (auto & pair : m_instrumentData) {
             InstrumentData & data = pair.second;
             data.buckets.clear();
             data.currentBucketBuyVolume = Volume(0);
             data.currentBucketTotalVolume = Volume(0);
             data.currentVPIN = 0.0;
             // Don't reset lastAlertTime to preserve cooldown periods
         }
     }
 
 private:
     // Structure to hold per-instrument data
     struct InstrumentData {
         std::deque<double> buckets;           // Bucket values (buy volume fractions)
         Volume currentBucketBuyVolume;        // Buy volume in current bucket
         Volume currentBucketTotalVolume;      // Total volume in current bucket
         Price lastPrice;                      // Last trade price (for tick rule)
         bool lastTradeWasBuy;                 // Direction of last trade
         double currentVPIN;                   // Current VPIN value
         tbricks::DateTime lastAlertTime;      // Time of last alert
     };
 
     // Parameter definitions
     InstrumentParameterDefinition m_paramInstruments = 
         InstrumentParameterDefinition(ParameterDefinition::CreationContext(),
                                      "Instruments",
                                      "Instruments to monitor for VPIN");
     
     VolumeParameterDefinition m_paramBucketSize = 
         VolumeParameterDefinition(ParameterDefinition::CreationContext(),
                                 "BucketSize",
                                 "Volume bucket size");
     
     IntegerParameterDefinition m_paramNumBuckets = 
         IntegerParameterDefinition(ParameterDefinition::CreationContext(),
                                  "NumBuckets",
                                  "Number of buckets for VPIN calculation");
     
     DoubleParameterDefinition m_paramVPINThreshold = 
         DoubleParameterDefinition(ParameterDefinition::CreationContext(),
                                 "VPINThreshold",
                                 "Threshold for VPIN alerts (0-1)");
     
     IntegerParameterDefinition m_paramAlertCooldownSeconds = 
         IntegerParameterDefinition(ParameterDefinition::CreationContext(),
                                  "AlertCooldownSeconds",
                                  "Minimum seconds between alerts for the same instrument");
 
     // Application state
     tbricks::ParameterSet m_parameters;
     std::mutex m_mutex;
     std::unordered_map<std::string, InstrumentData> m_instrumentData;
     std::unordered_map<std::string, TradeStream> m_tradeStreams;
     std::unordered_map<StreamIdentifier, InstrumentVenueIdentification> m_streamToInstrument;
     
     // Configuration values
     Volume m_bucketSize;
     Integer m_numBuckets;
     Double m_vpinThreshold;
     Integer m_alertCooldownSeconds;
     
     // Services and state
     AlertService & m_alertService;
     ViewerContext * m_viewer;
     bool m_isActive;
 };
 
 } // namespace tbricks
 
 // Required app factory function
 tbricks::Application * GetApplication(const tbricks::InitializationReason & reason) {
     return new tbricks::VPINMonitorApp(reason);
 }