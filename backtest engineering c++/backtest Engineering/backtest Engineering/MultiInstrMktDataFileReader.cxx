
#include "MultiInstrmktDataFileReader.hxx"
#include "StrategyBase.hxx"
#include "TradingIndicatorFileReader.hxx"
#include "Hdf5DataConsumer.hxx"

#include <cstdint>
#include <string>
#include <vector>

MultiInstrmktDataFileReader::MultiInstrmktDataFileReader(const std::string& fileName, const std::vector<PRWInstrument>&consumers， TradingIndicatorFileReader *tifr) 
:Hdf5TableFormatReader(fileName), MarketDataFileReader(tifr)
{}

void MultiInstrmktDataFileReader::init(StrategyBase& strategyBase)
{
	Hdf5TableFormatReader::init();
	if(columnsAndOffsetByNumericType_[nInt64].second ==-1)
	{
		throw std::runtime_error("No quantity columns were found in market data file");

	}

	if(columnsAndOffsetByNumericType_[nFloat64].second ==-1)
	{
		throw std::runtime_error("No price columns were found in market data file");
		
	}

	if(ricIdOffset_==-1)
	{
		throw std::runtime_error("A category column is expected in market data file, otherwise")；
	}

	//find ric in meta data

	auto metaGroup = group_->openGroup("meta");
	std::string ricValueBlockName;
	for (auto oi =0; oi<metaGroup.getNumObjs() && ricValueBlockName.empty(); ++oi)
	{
		auto id = metaGroup.getObjnameByIdx(oi);
		if (id.size()》13 && id.find('values_block_')==0)
		{
			ricValueBlockName = id;

		}
	}
	if(ricValueBlockName.empty())
	{
		throw std::runtime_error("can not find rick meta data in the market data file");

	}
	auto ricValueBlockGroup = metaGroup.openGroup(ricValueBlockName);
	auto metaSubGroup = ricValueBlockGroup.openGroup('meta');
	auto ricDataSet = metaSubGroup.openDataSet("table");
	//check compound type structure 

	const char excuse[] = "in the market data file, the ric meta data should be a table with exactly 2 columns, index64 int "
	auto compoundData = ricDataSet.getCompType();
	if(compoundData.getNmembers()!=2)
	{
		throw std::runtime_error(execuse);
	}

	if(compoundData.getMemberName(0)!="index")
	{
		throw std::runtime_error(execuse);
	}

	if(compoundData.getMemberDataType(0)!=H5::PredType::STD_I64LE)
	{
		throw std::runtime_error(execuse);
	}

	if(compoundData.getMemberName(1)!="values")
	{
		throw std::runtime_error(execuse);
	}

	if(compoundData.getMemberDataType(1).getClass()!=H5T_STRING)
	{
		throw std::runtime_error(execuse);
	}

	if(compoundData.getMemberDataType(1).isVariableStr())
	{
		throw std::runtime_error(execuse);
	}

	auto strLength = compoundData.getMemberStrType(1).getSize();

	//read rics

	auto dataspace = ricDataSet.getSpace();
	auto totalRics = dataspace.getSelectNpoints();
	std::vector<std::string> rics(totalRics);
	std::unique_ptr<char[]> buffer(new char[totalRics*compoundData.getSize()]);
	auto bufferCursor = buffer.get();
	ricDataSet.read(bufferCursor, compoundData);
	for (hssize_t i =0; i<totalRics; ++i)
	{
		const std:: int64_t index = *reinterpret_cast<std::int64_t*> (bufferCursor);
		std::string ric (bufferCursor+sizeof(std::int64_t),bufferCursor+sizeof(std::int64_t)+strLength);
		rics.at(index).swap(ric);
		bufferCursor+=compoundData.getSize();
	}

	ricDataSet.close();
	metaSubGroup.close();
	ricValueBlockGroup.close();
	metaGroup.close();

	//now use ric array to populate instrumentn

	for (auto ric:rics)
	{
		auto inst = strategyBase.instrument(ric);
		if (!inst)
		{
			throw std::runtime_error("RIC"+ric +" is not part of configuration therefore can't playback this dataset");

		}

		auto rwInstrument = std::static_pointer_cast<rwInstrument>(inst);
		rwInstrument->configurePrices(columnsAndOffsetByNumericType_[nFloat64].first);
		rwInstrument->configureQuantities(columnsAndOffsetByNumericType_[nInt64].first);
		instruments_.emplace_back(rwInstrument);
	}
}

void MultiInstrmktDataFileReader::read(StrategyBase& strategyBase, bool verbose)
{
	/*std::vector<std::function<void(std::int64_t)>> callBacks;
	for (auto instrument:instruments_)
	{
		callBacks.emplace_back([&strategyBase, instrument](std::int64_t rawData){strategyBase.manageTradingIndicatorUpdate(rawData,instrument);}) 
	}*/

	auto callBack = [&strategyBase](std::int64_t rawData) {strategyBase.manageTradingIndicatorUpdate(rawData);};
	Quantity sumOfQuantities = 0;


	std::int64_t rawDate = 0LL;

	for (hsize_t batchNumber = 0, processedPoints = 0; processedPoints < totalPoints ; ++batchNumber)
	{
		countOut_ = std::min(countOut_, totalPoints_-processedPoints);
		H5::DataSpace memorySpace(rankChunk_, &countOut_); 
		hsize_t offset = processedPoints;
		dataFileSpace_->selectHyperslab(H5S_SELECT_SET, &countOut_, &offset); 
		dataSet_->read(buffer_.get() , *compoundData_, memorySpace, *dataFileSpace_); 
		auto bufferCursor = buffer_.get();
		//step 1 run through file until sum of quantities is not zero 
		hsize_t inBatchCounter = ©; 
		std::int8_t lastlnstrumentld = 0;
		for (inBatchCounter = 0; inBatchCounter < countOut_ && !sumOfQuantities; ++inBatchCounter) 
			{
				rawDate = *reinterpret_cast<std::int64_t*>(bufferCursor + dateOffset_);
				lastlnstrumentld = *(bufferCursor + ricld0ffset_); 
				auto instrument = instruments_.at(lastInstrumentId);
				instrument->shiftHistoryCursor();
				instrument->readPrices(bufferCursor + columnsAndOffsetByNumericType_[nFloat64].second);
				sumOfQuantities += instrument->readQuantities(bufferCursor + columnsAndOffsetByNumericType_[nInt64].second);
				instrument~>shiftHistoryCursor();
				instrument->readPrices(bufferCursor + columnsAndQffsetByNumericType_[nFloat64].second);
				sumOfQuantities += instrument->readQuantities(bufferCursor + columnsAnd0ffsetByNumericType_[nInt64].second);
				++processedPoints;
				bufferCursor += compoundData_->getSize();
			}
		if (UNLIKELY(inBatchCounter && processedPoints < totalPoints_))
		{
			strategyBase.manageOrderBookUpdate(rawDate, instruments_.at(lastInstrumentId));
			if (LIKElY(tifr_!= nullptr))
			{
				tifr ->moveTo(rawDate);
			}
		}
			// step 2 cruise rhythm
		for (;inBatchCounter < countOut_; ++inBatchCounter)
		{
			rawDate = *reinterpret_cast<std::int64_t*>(bufferCursor + dateOffset_);
			// unfold and play delayed events
			strategyBase.manageDelayedOperationsAndEvents(rawDate);
			lastInstrumentId = *(bufferCursor + ricIdOffset_);
			auto instrument = instruments_.at(lastInstrumentId);
			instrument->shiftHistoryCursor();
			instrument->readPrices(bufferCursor+columnsAnd0ffsetByNumericType_[nFloat64].second);
			instrument->readQuantities(bufferCursor+columnsAndOffsetByNumericType_[nInt64].second);
			++processedPoints;
			bufferCursor+=compoundData_->getSize();

			StrategyBase.manageOrderBookUpdate(rawDate, instrument);
			if(LIKELY(tifr_!=nullptr))
			{
				tifr_->readUntil(rawDate, callBack);

			}

			if(verbose&& !(processedPoints%10000))
			{
				strategyBase.logFigures(rawDate);
			}
		}
		
}