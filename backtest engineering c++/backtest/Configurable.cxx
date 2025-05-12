#include "Configurable.hxx"

#include <sstream>
#include <exception>
#include <set>

template<>
void Configurable::QuantityParameterHandler::writeValue(rapidjson::Value& doc, rapidjson::Document::AllocatorType& allocator) const
{
	rapidjson::Value key(name_.c_str(), allocator);
	rapidjson::Value value;
	value.Set<std::int64_t>(parameter_);
	doc.AddMember(key, value, allocator);
}

template<>
void Configurable::PriceParameterHandler::writeValue(rapidjson::Value& doc, rapidjson::Document::AllocatorType& allocator)const
{
	rapidjson::Value key(name_.c_str(), allocator);
	rapidjson::Value value;
	value.Set<double>(parameter_.dbl());
	doc.AddMember(key, value, allocator);

}

template<>
void Configurable::StringParameterHandler::writeValue(rapidjson::Value& doc, rapidjson::Document::AllocatorType& allocator)const
{
	rapidjson::Value key(name_.c_str(), allocator);
	rapidjson::Value value;
	value.SetString(parameter_.c_str(), static_cast<rapidjson::SizeType>(parameter_.size()));
	doc.AddMember(key, value.Move(), allocator);

}

void Configurable::readParameterValues(const rapidjson::Value& object)
{
	std::set<std::string> incompatibleVariables;
	for (auto& spph:allParameterHandlers())
	{
		for (auto& pph:spph.second)
		{
			if(!pph.second->readValue(object))
			{
				incompatibleVariables.emplace(pph.first);
			}
		}

	}
	if(!incompatibleVariables.empty())
	{
		std::stringstream oss; oss<<"Those variables could not be read properly, wrong type"<<incompatibleVariables;
		throw std::runtime_error(oss.str());

	}
	onInit();
}