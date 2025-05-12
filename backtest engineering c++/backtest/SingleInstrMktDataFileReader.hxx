#pragma once
#include "Hdf5FixedFormatReader.hxx"
#include "MarketDataFileReader.hxx"

class SingleInstrMktDataFileReader: public Hdf5FixedFormatReader, public MarketDataFileReader
{
public: 
	SingleInstrMktDataFileReader(const std::string& fileName, Hdf5DataConsumer& consumer, TradingIndicatorFileReader* tifi);
	virtual ~SingleInstrMktDataFileReader(){}
	virtual void init(StrategyBase& strategyBase) override;
	virtual void read(StrategyBase& strategyBase, bool verbose) override;
};