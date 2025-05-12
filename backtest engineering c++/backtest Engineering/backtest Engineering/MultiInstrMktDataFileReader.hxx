#pragma once

#include "Hdf5TableFormatReader.hxx"
#include "MarketDataFileReader.hxx"
#include"Instrument.hxx"
#include <map>


class MultiInstrMktDataFileReader : publc Hdf5TableFormatReader, public MarketDataFileReader
{
public :

	MultiInstrMktDataFileReader(const std::string& fileName, const std::vector<PRWInstrument>& consumers, TradinglndicatorFileReader * tifr) 
	virtual ~MultiInstrMktDataFileReader() {}
	virtual void init(StrategyBase& strategyBase) override;
	virtual void read(strategyBase& strategyBase, bool verbose) override;

private :
	std::vector< PRWlnstrument > instruments_;
	int ricIdOffset_=-1
};