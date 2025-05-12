#include "TradingIndicatorFileReader.hxx"
#include "Hdf5DataConsumer.hxx"

TradingIndicatorFileReader::TradingIndicatorFileReader(const std::string& fileName, Hdf5DataConsumer& consumer):Hdf5FixedFormatReader(fileName,consumer)
{

}

void TradingIndicatorFileReader::moveTo(const std::int64_t to)
{
	for (std::int64_t rawDate = 0LL; cursor_[0]<rows_ && (rawDate= datesDirectory_[cursor_[0]])<to; ++cursor_[0])
	{
		;
	}

}

void TradingIndicatorFileReader::readUntil(const std::int64_t until, const std::function<void(std::int64_t)>& callBack)
{
	hsize_t countOut[2] ={1,0};
	for (std::int64_t rawDate = 0LL; cursor_[0]<rows_ && (rawDate= datesDirectory_[cursor_[0]])<= until; ++cursor_[0])
	{
		consumer_->shiftHistoryCursor();
		for (auto it= blockSetRoots_.begin(); it!= blockSetRoots_.end(); ++it)
		{
			countOut[1] = it->second.cols;
			H5::DataSpace memorySpace(1,&countOut[1]);
			blockDataFileSpace_[it->second.numericType]->selectHyperslab(H5S_SELECT_SET, countOut, cursor_);
			auto buffer = readBuffers_[it->second.numericType];
			auto& blockDataSet = blockDataSets_[it->second.numericType];
			blockDataSet->read(buffer, blockDataSet->getDataType(), memorySpace, *blockDataFileSpace_[it->second.numericType])//need to check
			if(it->second.numericType== nFloat64)
			{
				consumer_->readPrices(buffer);
			}
			else if (it->second.numericType==nInt64)
			{
				consumer_->readQuantities(buffer);
			}
		}
		callBack(rawDate);
	}
}