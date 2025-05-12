#pragma once

#include "Hdf5Reader.hxx"

#include <vector>

class Hdf5TableFormatReader: public Hdf5Reader
{
protected:
	Hdf5TableFormatReader(const std::string&fileName);
public:
	void init();
protected:
	hsize_t totalPoints_ =0;
	hsize_t countOut_=0;
	int rankChunk_=1;
	int dateOffset_ =-1;

	std::unique_ptr<char[]> buffer_;
	std::pair<std::vector<std::string>, int> columnsAndOffsetByNumericType_[__n_size__];
	std::shared_ptr<H5::DataSet> dataSet_;
	std::shared_ptr<H5::DataSpace> dataFileSpace_;
	std::shared_ptr<H5::CompType> compoundData_;
};