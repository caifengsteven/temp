#pragma once

#include <cstdint>

template <typename T>

class Averager
{
	std::uint64_t count_ = 0u;
	T sum_;

public:
	Averager():sum_(){}
	void record(const T value)
	{
		++count_;
		sum_+=value;
	}

	T mean() const
	{
		return sum_/count_;
	}
};