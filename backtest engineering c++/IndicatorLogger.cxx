#include "IndicatorLogger.hxx"

#include "Utils.hxx"

#include <tuple>

IndicatorLogger::PPriceIndicator IndicatorLogger::declarePriceIndicator(const std::string& name)
{
	auto& pi = priceIndcator_[name];
	if(pi|| quantityIndicators_.find(name)!= quantityIndicators_.end()||stringIndicators_.find(name)!=stringIndicators_.end())
	{
		throw std::runtime_error("Logged price indicator "+name+" already exist");

	}
	pi.reset(new Indicator<Price>);
	return pi;
}

IndicatorLogger::PQuantityIndicator IndicatorLogger::declareQuantityIndicator(const std::string& name)
{
	auto& pi = quantityIndicators_[name];
	if(pi||priceIndicators_.find(name)!= priceIndicators_.end()||stringIndicators_.find(name)!= stringIndicators_.end())
	{
		throw std::runtime_error("LOgged quantity indicator "+name+" already exist");
	}

	pi.reset(new Indicator<Quantity>);
	return pi;
}

IndicatorLogger::PStringIndicator IndicatorLogger::declareStringIndicator(const std::string&name)
{
	auto& pi = stringIndicators_[name];
	if(pi || quantityIndicators_.find(name)!=quantityIndicators_.end()||priceIndicators_.find(name)!=priceIndicators_.end())
	{
		throw std::runtime_error("Logged string indicator "+name +" already exists");
	}

	pi.reset(new Indicator<std::string>);
	return pi;
}

void IndicatorLogger::writeIndicatorsTo(rapidjson::Document& report) const
{
	auto &allocator = report.GetAllocator();
	for (const auto& qi:quantityIndicators_)
	{
		if(!indicatorFilter_.empty()&& indicatorFilter_.find(qi.first)==indicatorFilter_.end())
		{
			continue;
		}
		rapidjson::Value whens(rapidjson::kArrayType);
		rapidjson::Value quantities(rapidjson::kArrayType);
		for (const auto&qip:qi.second->data())
		{
			whens.PushBack(rapidjson::value(qip.first).Move(), allocator);
			quantities.PushBack(rapidjson::Value(qip.second).Move(), allocator);
		}
		rapidjson::Value indicator(rapidjson::KObjectType);
		indicator.AddMember("when", whens.Move(), allocator);
		indicator.AddMember(rapidjson::StringRef(qi.first.c_str()),quantities.Move(), allocator);
		rapidjson::Value tsname;
		tsname.SetString(std::string(qi.first+"_timeseries").c_str(), allocator);
		report.AddMember(tsname.Move(), indicator.Move(), allocator);
	}

	for (const auto& pi:priceIndicators_)
	{
		if(!indicatorFilter_.empty()&& indicatorFilter_.find(qi.first)==indicatorFilter_.end())
		{
			continue
		}
		rapidjson::Value whens(rapidjson::kArrayType);
		rapidjson::Value quantities(rapidjson::kArrayType);
		for (const auto&pip : pi.second->data())

	}
}