#include "Hdf5FixedFormatReader.hxx"
#include "Hdf5DataConsumer.hxx"

#include <vector>
#include <algorithm>

Hdf5FixedFormatReader::Hdf5FixedFormatReader(const std::string& fileName, Hdf5DataConsumer& consumer) : Hdf5Reader(fileName), consumer_(&consumer)
{}

void Hdf5FixedFormatReader::init()
{
	Hdf5Reader::init();
	std::string dateSetRoot;
	bool foundMainAxis = false;
	for (auto oi =0; oi <group_->getNumObjs(); ++oi)
	{
		auto id = group_->getObjnameByIdx(oi);
		if (id.size() >7 && id.find("block")==0 && id.find("_values") != std::string::npos)
		{
			auto valueSet = group_->openDataSet(id);
			if (valueSet.getDataType()== H5::PredType::IEEE_F64LE ||valueSet.getDataType() == H5::PredType::NATIVE_DOUBLE)
			{
				blockSetRoots_[id.substr(0, id.find('_'))].numericType = nFloat64;

			}
			else if (valueSet.getDataType()== H5::PredType::IEEE_F32LE ||valueSet.getDataType() == H5::PredType::NATIVE_FLOAT)
			{
				blockSetRoots_[id.substr(0, id.find('_'))].numericType = nFloat32;

			}

			else if (valueSet.getDataType()== H5::PredType::STD_I64LE ||valueSet.getDataType() == H5::PredType::NATIVE_LLONG)
			{
				blockSetRoots_[id.substr(0, id.find('_'))].numericType = nInt64;

			}


		}
		else if (id.size()>4 && id.find("axis")==0)
		{
			auto dateSet = group_->openDataSet(id);
			if (dataSet.getDataType() == H5::PredType::STD_I64LE)
			{
				for (int a =0; a<dateSet.getNumattrs(); ++a)
				{
					auto att = dateSet.openAttribute(a);
					if(att.getName()=="name")
					{
						std::string name ;
						att.read(att.getDataType(), name);
						if(name == "msgstamp")
						{
							int id0(0);
							if(1== std::sscanf(id.c_str(), "axis%d", &id0))
							{
								dateSetRoot = "axis"+std::to_string(id0);
							}
						}
					}
				}
			}
		}
	}

	if (blockSetRoots_.empty())
	{
		throw std::runtime_error("Cant find valid block data!");
	}

	for (auto& bsrEntry :blockSetRoots_)
	{
		auto dataSet = group_->openDataSet(bsrEntry.first+"_items");
		auto space = dataSet.getSpace();
		hsize_t dim(0);
		space.getSimpleExtentDims(&dim);
		auto stringSize = dataSet.getStrType().getSize();
		std::unique_ptr<char[]>buffer(new char[stringSize *dim]);
		dataSet.read(buffer.get(), dataSet.getStrType());
		std::vector<std::string> columnNames;
		for (hsize_t i=0; i<dim; i++)
		{
			auto cstr = buffer.get()+i*stringSize;
			auto last = std::find(cstr, cstr+stringSize,'\0');
			columnNames.emplace_back(cstr, last);
		}
		if(bsrEntry.second.numericType == nFloat64)
		{
			consumer_->configurePrices(columnNames);

		}
		else if (bsrEntry.second.numericType== nInt64)
		{
			consumer_->configureQuantities(columnNames);
		}
	}
	{
		for (auto& bsrEntry:blockSetRoots_)
		{
			auto blockDataSet = group_->openDataSet(bsrEntry.first+"_values");
			auto blockDataFileSpace = blockDataSet.getSpace();
			{
				hsize_t dims[2];
				blockDataFileSpace.getSimpleExtentDims(dims);
				if (rows_&& rows_!=dims[0])
				{
					throw std::runtime_error("Block data size are inconsistent");
				}
				rows_= dims[0];
				bsrEntry.second.cols = dims[1];
			}
		}
		hsize_t datesIndexSize(0);
		hsize_t allDateSize(0);
		{
			auto dateDataSet = group->openDataSet(dateSetRoot);
			auto dateDataFileSpace = dateDataSet.getSpace();
			dateDataFileSpace.getSimpleExtentDims(&allDatesSize);
			datesDirectory_.reset(new std::int64_t[allDatesSize]);
			int rankChunk =1;
			hsize_t countOut = allDatesSize;
			auto cparams = dateDataSet.getCreatePlist();
			if(cparams.getLayout()== H5D_CHUNKED)
			{
				rankChunk = cparams.getChunk(1, &countOut);

			}
			H5::DataSpace fileSpace(dateDataSet.getSpace());
			for (hsize_t cursor =0; cursor<allDatesSize;)
			{
				H5::DataSpace memorySpace(1, &countOut);
				fileSpace.selectHyperslab(H5S_SELECT_SET, &countOut, &cursor);
				dateDataSet.read(datesDirectory_.get()+cursor, dateDataSet.getDataType(), memorySpace, fileSpace);
				cursor+=countOut;
				countOut = std::min(countOut, allDatesSize-cursor);
			}
		}
	}
	for (auto& bsrEntry:blockSetRoots_)
	{
		blockDataSets_[bsrEntry.second.numericType].reset(new H5::dataSet(group_->openDataSet(bsrEntry.first +"_value")));
		blockDataFileSpace_[bsrEntry.second.numericType].reset(new H5::DataSpace(blockDataSets_[bsrEntry.second.numericType]->getSpace()));
		readBuffers_[bsrEntry.second.numericType].reset(new char[bsrEntry.second.byteBufferSize()]);
	}
}
