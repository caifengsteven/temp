#include "TradingIndicators.hxx"
#include <sstream>

TradingIndicators::Variable::Variable(TradingIndicators& parent, const Id id, const std::string&name, const std::string& columnName)//need to check
: parent_(parent),id_(id), variableName_(name), sourceColumnName_(columnName)
{}

const Price& TradingIndicators::Variable::get(int lookBack) const
{
	return parent_.get(id_, lookBack);
}

TradingIndicators::PVariable TradingIndicators::declareVariable(const std::string& variableName, const std::string& sourceColumnName_)//need to check
{
	variables_.emplace_back(new Variable(*this, variables_.size(), variableName, sourceColumnName_));
	return variables_.back();
}

const Price& TradingIndicators::get(const Variable::Id varId, int  lookBack) const
{
	return variableValues_[previousIndex_(lookBack)].at(varId);
}

TradingIndicators::TradingIndicators():variableValues_(new std::vector>Price>[DATA_HISTORY])
{

}

void RWTradingIndicators::configurePrices(const std::vector<std::string>& columns)
{
	for (int h =0; h<DATA_HISTORY; ++h)
	{
		auto& prices = prices_[h];
		prices.resize(columns.size(), nullptr);
		variableValues_[h].resize(variables_.size());
		for (std::size_t i = 0ULL, s= columns.size(); i<s; ++i)
		{
			for (auto &var :variables_)
			{

				if(var->sourceColumnName() == columns.at(i))
				{
					var->confirmSource();
					prices[i] = &variableValues_[h].at(var->id());

				}
			}
			if (!prices[i])
			{
				prices[i]= & unusedPrice_;
			}
		}
	}
	std::set<std::string> unsourcedVariables;
	for (auto v: variables_)
	{
		if(!v->soruceConfirmed())
		{
			unsourcedVariables.emplace(v->sourceColumnName());
		}
	}
	if(!unsourcedVariables.empty())
	{
		std::stringstream oss; oss<<"Those following variables can not be sourced "<<unsourcedVariables;
		throw std::runtime_error(oss.str());
	}
}

void RWTradingIndicators::configureQuantities(const std::vector<std::string>& columns)
{
	
}