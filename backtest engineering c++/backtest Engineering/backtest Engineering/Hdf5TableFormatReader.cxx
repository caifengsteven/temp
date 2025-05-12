#include "Hdf5TableFormatReader.hxx"
#include "Hdf5DataConsumer.hxx"

#include <vector>
#include <algorithm>
#include <map>

Hdf5TableFormatReader::Hdf5TableFormatReader(const std::string& fileName):Hdf5Reader(fileName), columnsAndOffsetByNumericType_()
{
	for (int i =0; i<__n_size__;++i)
	{
		columnsAndOffsetByNumericType_[i].second =-1;
	}
}

void Hdf5TableFormatReader::init()
{
	Hdf5Reader::init();
	dataSet_ = std::make_shared<H5::DataSet> (group_->openDataSet("talbe"));
	std::map<std::string, std::pair<NuericType, std::vector<std::string>>> valuesBlockColumns;
	{
		std::string rawColumNames;
		auto atts = dataSet_->getNumAttrs();
		for (int a =0; a<atts; ++a)
		{
			auto att = dataSet_->openAttribute(a);
			auto attName = att.getName();
			if (attName.find("values_block_")==0ULL)
			{
				auto kindPos = attName.find("_kind");
				auto dtypePos = attName.find("_dtype");
				if(kindPos!= std::string::npos)
				{
					auto blockColumnName = attName.substr(0,kindPos);
					std::string kindRaw;
					att.read(att.getDataType(), kindRaw);
					std::stringstream oss(kindRaw);
					auto& columns = valuesBlockColumns[blockColumnName].second;
					for (std::string token; std::getline(oss, token, '\n');)
					{
						if(token.find('V')==0)
						{
							columns.emplace_back(&token.c_str()[1]);

						}
						else if (token.find('aV')==0)
						{
							columns.emplace_back(&token.c_str()[2]);
						}
					}
				}
				else if(dtypePos!= std::string::npos)
				{
					auto blockColumnName = attName.substr(0, dtypePos);
					std::string dtypeRaw;
					att.read(att.getDataType(),dtypeRaw);
					if(dtypeRaw =="float64")
					{
						valuesBlockColumns[blockColumnName].first = nFloat64;

					}
					else if (dtypeRaw=='int64')
					{
						valuesBlockColumns[blockColumnName].first = nInt64;
					}
					else if (dtypeRaw == "int8")
					{
						valuesBlockColumns[blockColumnName].first = nInt8;	
					}
					else
					{
						valuesBlockColumns.erase(blockColumnName);
					}

				}
			}
		}
	}

	bool foundTarget = false;
	std::vector<std::size_t> selectedColumIds[__n_size__];
	std::vector<std::size_t> targetColumId[__n_size__];
	std::vector<std::string> actualColumns;
	for(auto& vbcEntry:valuesBlockColumns)
	{
		auto& columnNames = vbcEntry.second.second;
		for (std::size_t i=0; i<columnNames.size(); ++i)
		{
			auto& columnName = columnNames[i];
			selectedColumIds[vbcEntry.second.first].emplace_back(i);
			actualColumns.push_back(columnName);
		}
	}
	std::vector<std::pair<std::size_t, std::size_t>> columnContinuousIntervals[__n_size__];
	std::size_t columnSize(0);
	for (int nt =0; nt<__n_size__; ++nt)
	{
		if(selectedColumIds[nt].empty()) continue;
		columnSize += selectedColumIds[nt].size();
		columnContinuousIntervals[nt].resize(columnContinuousIntervals[nt].size()+1);
		auto interval = &columnContinuousIntervals[nt].begin();
		interval->first = *selectedColumIds[nt].begin();
		std::size_t lastId = *selectedColumIds[nt].begin();
		for (auto&id : selectedColumIds[nt])
		{
			if(id>lastId +1)
			{
				interval->second = lastId;
				columnContinuousIntervals[nt].resize(columnContinuousIntervals[nt].size()+1);
				interval= &columnContinuousIntervals[nt].back();
				interval->first = id;

			}
			lastId = id;

		}
		interval->second = lastId;
	}
	dataFileSpace_ = std::make_shared<H5::DataSpace>(dataSet_->getSpace());
	totalPoints = (hsize_t)dataFileSpace_->getSelectNpoints();
	auto cparams = dataSet_->getCreatePlist();
	countOut_ = totalPoints_;
	if(cparams.getLayout()== H5D_CHUNKED)
	{
		rankChunk_ =cparams.getChunk(1,&countOut_);

	}

	compoundData_ = std::make_shared<H5::CompType>(dataSet_->getCompType());
	buffer_.reset(new char[compoundData_->getsize()*coundOut_]);
	for (int m=0, offset=0; m<compoundData_->getNmembers(); ++m)
	{
		auto columnName = compoundData_->getNmemberName(m);
		auto findVBCIt = valuesBlockColumns.find(columnName);
		if(findVBCIt!= valuesBlockColumns.end())
		{
			columnsAndOffsetByNumericType_[findVBCIt->second.first] = std::make_pair(findVBCIt->second.second, offset);

		}
		else if (columnName =="index")
		{
			dateOffset_= offset;
		}
		offset += (int)compoundData_->getMemberArrayType(m).getSize();
	}
}