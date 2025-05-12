#include "Window.hxx"

PriceWindowIndicator::PriceWindowIndicator(const std::int64_t windowLengthNs):windowLengthNs_(windowLengthNs)
{

}

PriceSumIndicator::PriceSumIndicator(const std::int64_t windowLengthNs):PriceSumIndicator(windowLengthNs)
{}

const Price& PriceSumIndicator::record(const std::int64_t when, const Price i)
{
	window_.emplace_back(when, i);
	value_+=i;
	while(!window_.empty()&& window_.front().first+windowLengthNs<when)
	{
		value_ -=window_.front().second;
		window_.pop_front();
	}
	return value_;
}
PriceAverageIndicator::PriceAverageIndicator(const std::int64_t windowLengthNs) :PriceWindowIndicator(windowLengthNs)
{
}
const Price& PriceAverageIndicator::record(const std::int64_t when, const Price i)
{
	window_.emplace_back(when, i);
	windowSum_+=i;
	while(!window_.empty() && window_.front().first+windowLengthNs_<when)
	{
		windowSum_ -=window_.front().second;
		window_.pop_front();
	}
	value_ = windowSum_ /window_.size();
	return value_;
}


PriceStdevIndicator::PriceStdevIndicator(const std::int64_t windowLengthNs):PriceWindowIndicator(windowLengthNs)
{
}
const Price& PriceStdevIndicator::record(const std::int64_t when, const Price i)
{
	window_.emplace_back(when, i);
	windowSum_+=i;
	while(!window_.empty() && window_.front().first+windowLengthNs_<when)
	{
		windowSum_ -=window_.front().second;
		window_.pop_front();
	}
	const auto mean = windowSum_/window_.size();
	if(UNLIKELY(window_.size()==1u))
	{
		value_ = 0.0;
		return value_
	}
	double cumul = 0.0;
	for (const auto& v :window_)
	{
		auto tmp = (v.second - mean).dbl();
		cumul +=tmp * tmp;
	}
	value_ = std::sqrt(cumul/double(window_.size()-1u));
	return value_;
}

PriceTshiftIndicator::PriceTshiftIndicator(const std::int64_t windowLengthNs):PriceWindowIndicator(windowLengthNs)
{}

const Price& PriceTshiftIndicator::record(const std::int64_t when, const Price i)
{
	window_.emplace_back(when, i);
	while(!window_.empty() && window_.front().first+windowLengthNs_<when)
	{
		value_ = window_.front().second;
		window_.pop_front();
	}
	return value_;


}

