#pragma once
#include "Hdf5Reader.hxx"

class Hdf5FixedFormatReader:public Hdf5Reader
{
protected:
	Hdf5FixedFormatReader(const std::string& fileName, hdf5DataConsumer& consumer);
public:
	void init();
protected:
	hdf5DataConsumer* consumer_ = nullptr;
	std::unique_ptr<std::int64_t[]> datesDirectory_;
	std::shared_ptr<H5::DataSet> blockDataSets_[__n_size__];
	std::shared_ptr<H5::DataSpace> blockDataFileSpaces_[__n_size__];
	std::shared_ptr<char[]> readBuffers_[__n_size__];
}