#pragma once

#include "Quantity.hxx"
#include "Price.hxx"
#include "SerialisableEnum.hxx"
#include "Utils.hxx"

#include <rapidjson/document.h>
#include <unordered_map>
#include <map>
#include <string>
#include <ostream>
#include <memory>
#include <sstream>

class Configurable
{
public:
	class ParameterHandlerBase
	{
	protected: 
		const std::string name_;
		const std::string doc_;
	protected:
		ParameterHandlerBase(const std::string& name, const std::string& doc):name_(name), doc_(doc){}
	public:
		virtual ~ParameterHandlerBase(){}
		const std::string name() const{return name_;}
		virtual void description(std::ostream& os)const
		{
			os<<"\t- "<<name_<<" :"<<typeName()<<std::endl;
			os<<"\t\t"<<doc_<<std::endl;
		}
		virtual void writeDefaultValue(rapidjson::Document& doc)const=0;
		virtual void writeValue(rapidjson::Value& doc, rapidjson::Document::AllocatorType& allocator) const=0;

		virtual bool readValue(const rapidjson::Value& doc)=0;
	protected:
		virtual std::string typeName() const =0;


	};
	typedef std::shared_ptr<ParameterHandlerBase> PPH;
	template < typename PARAMETER_T, typename UNDL_T>
	class ParameterHandler:public ParameterHandlerBase
	{
	protected:
		PARAMETER_T& parameter_;
		const UNDL_T defaultValue_;
	public:
		ParameterHandler(const std::string&name, const std::string& doc, PARAMETER_T& parameter, const UNDL_T& defaultValue): ParameterHandlerBase(name, doc),parameter_(parameter), defaultValue_(defaultValue){}
		virtual void description(std::ostream& os)const override
		{
			ParameterHandlerBase::description(os);
			os<<"\t\t defaultValue"<<defaultValue_<<std::endl;
		}
		virtual bool readValue(const rapidjson::Value& doc) override
		{
			if (doc.IsObject()&& doc.HasMember(name_.c_str()))
			{
				if(!doc[name_.c_str()].Is<UNDL_T>())return false;
				parameter_ = doc[name_.c_str()].Get<UNDL_T>();
				return true;
			}
			else
			{
				parameter_=defaultValue_;
			}
			return true;
		}
		virtual void writeDefaultValue(rapidjson::Document& doc) const override
		{
			rapidjson::Value key(name_.c_str(), doc.GetAllocator());
			rapidjson::Value value;
			value.Set<UNDL_T>(defaultValue_);
			doc.AddMemeber(key, value, doc.GetAllocator());
		}
		virtual void writeValue(rapidjson::Value& doc, rapidjson::Document::AllocatorType& allocator) const override;
	protected:
		virtual std::string typeName() const override
		{
			return typeid(UNDL_T).name();
		}

	};

	typedef ParameterHandler<Quantity, std::int64_t> QuantityParameterHandler;
	typedef ParameterHandler<Price, double> PriceParameterHandler;
	typedef ParameterHandler<std::string, const char*> StringParameterHandler;
	class EnumParamterHandler: public ParameterHandlerBase
	{
		SerialisableEnum& parameter_;
	public:
		EnumParamterHandler(const std::string& name, const std::string& doc, SerialisableEnum& parameter) : ParameterHandlerBase(name, doc), parameter_(parameter){}
		virtual void description(std::ostream& os) const override
		{
			ParameterHandlerBase::description(os);
			os<<"\t\t default value "<<parameter_.defaultValue() <<std::endl;
			os<<"\t\t value range" <<parameter_.valueRange()<<std::endl;

		}
		virtual bool readValue(const rapidjson::Value& doc) override
		{
			if (doc.IsObject() && doc.HasMember(name_.c_str()))
			{
				if(!doc[name_.c_str()].IsString()) return false;
				const std::string val = doc[name_.c_str()].GetString();
				const auto valueRange = parameter_.valueRange();
				if(!valueRange.empty()&& valueRange.find(val)== valueRange.end())
				{
					std::stringstream oss; oss<< "Intended parameter value "<<val<<"  is not in range"<<valueRange;

				}
				parameter_ = val;
				return true;
			}
			else
			{
				parameter_ = parameter_.defaultValue();
			}
			return true;
		}
		virtual void writeDefaultValue(rapidjson::Document& doc) const override
		{
			rapidjson::Value key(name_.c_str(), doc.GetAllocator());
			rapidjson::Value value(parameter_.defaultValue().c_str(), static_cast<rapidjson::SizeType>(parameter_.defaultValue().size()),doc.GetAllocator());
			doc.AddMemeber(key, value, doc.GetAllocator());
		}

		virtual writeValue(rapidjson::Value& doc, rapidjson::Document::AllocatorType& allocator) const override
		{
			rapidjson::Value key(name_.c_str(), allocator);
			auto strValue = parameter_.toString();
			rapidjson::Value value(strValue.c_str(), static_cast<rapidjson::SizeType>(strValue.size()), allocator);
			doc.AddMemeber(key, value, allocator);
		}
	protected:
		virtual std::string typeName() const override
		{
			return "enumerate";
		}


	};
	void addParameter(const std::string fileName, PPH pph)
	{
		for (auto& ph:configurationHandler_)
		{
			if(ph.second.find(pph->name())!=ph.second.end())
			{
				throw std::runtime_error("configuration parameter "+pph->name() +" already exist in file"+ph.first);
			}
		}
		configurationHandler_[fileName][pph->name()] = pph;
	}
	const auto& allParameterHandlers() const
	{
		return configurationHandler_;
	}
	void readParameterValues(const rapidjson::Value& object);
	virtual void onInit(){}
	private:
		std::map<std::string, std::map<std::string, PPH>> configurationHandler_;

}

#define ADD_PARAMETER(type, name, parameter, defaultValue, doc )\
	addParameter(__FILE__, PPH(new type##ParameterHandler(name, doc, parameter, defaultValue)))

#define ADD_ENUM_PARAMTER(name, parameter, doc)\
	addParameter(__FILE__, PPH(new EnumParamterHandler(name, doc, parameter)))