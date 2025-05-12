#pragma once

#include <H5Cpp.h>
#include <string>
#include <memory>
#include <map>
#include <cstdint>

class Hdf5DataConsumer;
class Hdf5Reader
{
protected:
	Hdf5Reader(const std::string& fileName);
	void init();
protected:
	const std::string fileName_;
	std::shared_ptr<H5::H5File> file_;
	std::shared_ptr<H5::Group> group_;
	typedef enum {nUndefined=0, nInt8 =2, nFloat32=3, nFloat64=4, __n_size__} NumericType;
	struct NumericTypeAndCols
	{
		NumericType numericType = (NumericType)0;
		hsize_t cols =0;
		hsize_t byteBufferSize() const
		{
			hsize_t unitSize =1;
			switch (numericType)
			{
				case nInt8: 
				unitSize =1;
				break;
				case nFloat32:
				unitSize=4;
				break;
				case nInit64:
				case nFloat64:
				unitSize=8;
				break;
			}
			return unitSize*cols;
		}
	};
	std::map<std::string, NumericTypeAndCols> blockSetRoots_;
	std::size_t rows_=0ULL;
};