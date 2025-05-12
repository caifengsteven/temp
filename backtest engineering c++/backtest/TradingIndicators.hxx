#pragma once
#include "HistoricalDataHolder.hxx"
#include "Hdf5DataConsumer.hxx"
#include "Price.hxx"
#include <memory>
#include <string>

class RWTradingIndicators;
class TradingIndicators: virtual public HistoricalDataHolder
{
public:
	virtual ~ TradingIndicators(){}

	class Variable
	{
		TradingIndicators& parent_;
		const std::size_t id_;
		const std::string variableName_;
		std::string sourceColumnName_;
		bool sourceConfirmed_ = false;
	public:
		typedef std::size_t Id;
		const Id id() const {return id_;}
		const std::string& variableName() const {return variableName_;}
		std::string& sourceColumnName() {return sourceColumnName_;}
		const Price& get(int lookBack=0) const;
		bool valueJustChanged() const {return get()!=get(1);}
	private:
		bool sourceConfirmed() const {return sourceConfirmed_;}
		void confirmSource() {sourceConfirmed_= true;}
	private:
		friend class TradingIndicators;
		friend class RWTradingIndicators;
		Variable(TradingIndicators& parent, const Id id, const std::string& name, const std::string& columnName);
		Variable(const Variable&) = delete;
		Variable(Variable&&) delete;
	};
	typedef std::shared_ptr<Variable> PVariable;
	PVariable declareVariable(const std::string& variableName, const std::string& sourceColumnName);
private:
	const Price& get(const Variable::Id varId, int lookBack=0) const;
protected:
	TradingIndicators();
protected:
	std::unique_ptr<std::vector<Price>[]> variableValues_;
	std::vector<PVariable> variables_;
};
typedef std::shared_ptr<TradingIndicators>PTradingIndicators;

class RWTradingIndicators: public TradingIndicators, public Hdf5DataConsumer
{
	virtual void configurePrices(const std::vector<std::string>& columnNames) override;
	virtual void configureQuantities (const std::vector<std::string>& columnNames) override;
public:
	virtual ~RWTradingIndicators(){}

};
typedef std::shared_ptr<RWTradingIndicators>PRWTradingIndicators;
