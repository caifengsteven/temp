#pragma once

#include "Quantity.hxx"
#include "Price.hxx"

#include <rapidjson/document.h>

#include <string>
#include <vector>
#include <utility>
#include <cstdint>
#include <map>
#include <set>

class IndicatorLogger
{
public:
	template<class INDICATOR_TYPE>
	class Indicator
	{
		std::vector<std::pair<std::int64_t, INDICATOR_TYPE>> data_;
	public:
		Indicator()
		{
			data_.reserve(10000);
		}

		virtual ~Indicator()
		{

		}

		void logIndicator(const std::int64_t when, const INDICATOR_TYPE i)
		{
			data_.emplace_back(std::piecewise_construct, std::forward_as_tuple(when), std::forward_as_tuple(i));
		}

	private:
		friend class IndicatorLogger;
		const auto& data() const
		{
			return data_;
		}
	};
	typedef std::shared_ptr<Indicator<Quantity>> PQuantityIndicator;
	typedef std::shared_ptr<Indicator<Price>> PPriceIndicator;
	typedef std::shared_ptr<Indicator<std::string>> PStringIndicator;

	PPriceIndicator declarePriceIndicator(const std::string& name);
	PQuantityIndicator declareQuantityIndicator(const std::string& name);
	PStringIndicator declareStringIndicator(const std::string& name)

	template<class INDICATOR_TYPE>
	class LogIfChangeIndicator
	{
		INDICATOR_TYPE lastLoggedValue_;
		bool logged_ = false;
		const std::shared_ptr<Indicator<INDICATOR_TYPE>> wrapped_;
	public:
		LogIfChangeIndicator(const std::shared_ptr<Indicator<INDICATOR_TYPE>> wrapped): lastLoggedValue_(), wrapped_(wrapped){}

		void logIndicator(const std::int64_t when, const INDICATOR_TYPE i)
		{
			if (LIKELY(!logIndicator||lastLoggedValue_!=i))
			{
				wrapped_->logIndicator(when, i);
				lastLoggedValue_=i;
				logged_= true;

			}
		}
	};
	typedef std::shared_ptr<LogIfChangeIndicator<Quantity>> PQuantityLogIfChangeIndicator;
	typedef std::shared_ptr<LogIfChangeIndicator<Price>> PPriceLogIfChangeIndicator;
	typedef std::shared_ptr<LogIfChangeIndicator<std::string>> PStringLogIfChangeIndicator;
	template<class INDICATOR_TYPE>
	static std::shared_ptr<LogIfChangeIndicator<INDICATOR_TYPE>> logIfChange(std::shared_ptr<Indicator<INDICATOR_TYPE>> wrapped)
	{
		return std::make_shared<LogIfChangeIndicator<INDICATOR_TYPE>>(wrapped);
	}

	void writeIndicatorsTo(rapidjson::Document& report) const;
	std::set<std::string> availableTimeSeries() const;
	std::set<std::string>& indicatorFilter(){return indicatorFilter_;}
private:
	std::map<std::string, PQuantityIndicator> quantityIndicators_;
	std::map<std::string, PPriceIndicator> priceIndicators_;
	std::map<std::string, PStringIndicator> stringIndicators_;
	std::set<std::string> indicatorFilter_;

};