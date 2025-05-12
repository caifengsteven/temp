#pragma once

#include "HistoricalDataHolder.hxx"

#include "Price.hxx"
#include "Quantity.hxx"

#include <vector>
#include <string>
#include <memory>

class Hdf5DataConsumer:virtual public HistoricalDataHolder
{
public:
	Hdf5DataConsumer();
	virtual ~Hdf5DataConsumer(){}
	virtual void configurePrices(const std::vector<std::string>& columnNames)=0;
	virtual void cofnigureQuantities(const std::vector<std::string>& columns) = 0;
	void readPrices(const char * buffer);
	Quantity readQuantities(const char * buffer);

protected:
	static Price unusedPrice_;
	static Quantity unusedQuantity_;
	std::unique_ptr<std::vector<Price*>[]>prices_;
	std::unique_ptr<std::vector<Quantity*>[]>quantities_;
};
