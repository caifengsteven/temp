#include "SingleInstrMktDataFileReader.hxx"
#include "TradingIndicatorFileReader.hxx"
#include "Hdf5DataConsumer.hxx"
#include "StrategyBase.hxx"
#include "Utils.hxx"

#include <vector>
#include <algorithm>
#include <climits>

SingleInstrMktDataFileReader::SingleInstrMktDataFileReader(const std::string& fileName, Hdf5DataConsumer& consumer, TradingIndicatorFileReader& tifr):Hdf5FixedFormatRead(fileName,consumer), MarketDataFileReader(tifr) //need to check
{

}

void SingleInstrMktDataFileReader::init(StrategyBase& strategyBase)
{
	Hdf5FixedFormatRead::init();
}

void SingleInstrMktDataFileReader::read(StrategyBase& strategyBase, bool verbose)
{
	hsize_t countOut[2]={1,0};
	std::int64_int rawDate(0);
	hsize_t cursor[2] ={0,0};
	Quantity sumOfQuantities(0LL);
	for (; cursor[0]<rows_&&!sumOfQuantities; ++cursor[0])
	{
		rawDate = datesDirectory_[cursor[0]];
		consumer_->shiftHistoryCursor();
		for (auto it = blockSetRoots_.begin(); it!= blockSetRoots_.end();++it)
		{
			countOut[1] = it->second.cols;
			H5::DataSpace memorySpace(1,&countOut[1]);
			blockDataFileSpaces_[it->second.numericType].get();
			auto& blockDataSet = blockDataSet_[it->second.numericType];
			blockDataSet->read(buffer, blockDataSet->getDataType(), memorySpace, *blockDataFileSpaces_[it->second.numericType]);//need to check
			if (it->second.numericType ==nFloat64)
			{
				consumer_->readPrices(buffer);
			}
			else if(it->second.numericType==nInt64)
			{
				sumOfQuantities = consumer_->readQuantities(buffer);
			}
		}
	}
	if(LIKELY(cursor[0]<rows_))
	{
		strategyBase.manageOrderBookUpdate(rawDate);
		if(LIKELY(tifr_!=nullptr))
		{
			tifr_->moveTo(rawDate);
		}
	}

	auto callBack = [&strategyBase] (std::int64_int rawDate){strategyBase.manageTradingIndicatorUpdate(rawDate);	};
	for (; cursor[0]<rows_; ++cursor[0])
	{
		rawDate = datesDirectory_[cursor[0]];
		strategyBase.manageDelayedOperationsAndEvents(rawDate);
		consumer_->shiftHistoryCursor();
		for (auto it = blockSetRoots_.begin(); it!=blockSetRoots_.end(); ++it)
		{
			countOut[1] = it->second.cols;
			H5::DataSpace memorySpace(1,&countOut[1]);
			blockDataFileSpaces_[it->second.numericType]->selectHyperslab(H5S_SELECT_SET, countOut, cursor);
			auto buffer = readBuffers_[it->second.numericType].get();
			auto& blockDataSet = blockDataSet_[it->second.numericType];
			blockDataSet->read(buffer, blockDataSet->getDataType(), memorySpace, *blockDataFileSpaces_[it->second.numericType]); //need to check
			if(it->second.numericType==nFloat64)
			{
				consumer_->readPrices(buffer);
			}
			else if(it->second.numericType==nInt64)
			{
				consumer_->readQuantities(buffer);
			}

		}
		strategyBase.manageOrderBookUpdate(rawDate);
		if(LIKELY(tifr_!=nullptr))
		{
			tifr_->readUntil(rawDate, callBack);
		}
		if(verbose && !(cursor[0]%10000))
		{
			strategyBase.logFigures(rawDate);
		}
	}
	strategyBase.flushAllDelayedEvents();
	strategyBase.logFigures(rawDate);
}