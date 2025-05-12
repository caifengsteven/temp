#pragma once

class StrategyBase;

class MarketDataFileReader
{
public:
	virtual ~MarketDataFileReader()
	{

	}
	virtual void init(StrategyBase& strategyBase) =0;
	virtual void read(StrategyBase& strategyBase, bool verbose )=0;
protected:
	MarketDataFileReader(TradingIndicatorFileReader* tifr):tifr_(tifr)
	{

	}
protected:
	TradingIndicatorFileReader* tifr_ = nullptr;

};