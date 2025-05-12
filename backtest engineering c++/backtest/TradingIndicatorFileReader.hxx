#pragma once
#include "Hdf5FixedFormatReader.hxx"
#include <cstdint>
#include <string>
#include <functional>

class TradingIndicatorFileReader: public Hdf5FixedFormatReader
{
public:
	TradingIndicatorFileReader(const std::string& fileName, Hdf5FixedFormatReader& consumer);
	void moveTo(const std::int64_t to);
	void readUntil(const std::int64_t until, const std::function<void(std::int64_t)>& callBack);
private:
	long long unsigned int cursor_[2] = {0ULL, 0ULL};

};