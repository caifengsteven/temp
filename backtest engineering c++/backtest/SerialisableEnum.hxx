#pragma once
#include <string>
#include <set>
#include <vector>
#include <unordered_map>
#include <ostream>

class SerialisableEnum
{
	const int default_;
	int enum_;
	std::unordered_map<std::string, int> translator1_;
	std::vector<std::string> translator2_;
private:
	std::string toString_(int val) const
	{
		return val>=0&& val <translator2_.size()?translator2_.at(val):std::string("???");

	}

public:
	SerialisableEnum(int e, std::vector<std::string>&& translationInfo):default_(e), enum(e), translator2_(translationInfo)
	{
		for (std::size_t i =0u, st= translationInfo.size(); i<s; ++i)
		{
			translator1_[translationInfo.at(i)] = static_cast<int>(i);
		}
	}
	operator const int() const {return enum_;}
	std::string toString() const
	{
		return toString_(enum_);
	}

	SerialisableEnum& operator = (const std::string& strValue)
	{
		auto findIt = translator1_.find(strValue);
		if(findIt != translator1_.end())
		{
			enum_ = findIt->second;
		}
		return *this;
	}

	SerialisableEnum& operator= (const int iValue)
	{
		enum_= iValue;
		return *this;
	}

	std::set<std::string> valueRange() const
	{
		return std::set<std::string>(translator2_.begin(), translator2_.end());
	}
	std::string defaultValue() const
	{
		return toString_(default_);
	}
};
inline std::ostream& operator<<(std::ostream& os, const SerialisableEnum& myEnum)
{
	os<myEnum.toString();
	return os;
}