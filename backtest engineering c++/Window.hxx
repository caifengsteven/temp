#pragma once
#include "Price.hxx"
#include "Quantity.hxx"

#include <cstdint>
#include <deque>
#include <utility>
#include <memory>

class PriceWindowIndicator
{
protected:
	const std::int64_t windowLengthNs_;
	typedef std::deque<std::pair<std::int64_t, Price>> Window;
	Window window_;
	Price value_;

protected:
	PriceWindowIndicator(const std::int64_t windowLengthNs);
public:
	virtual ~PriceWindowIndicator(){}
	const Price& get() const {return value_;}
};
class PriceSumIndicator:public PriceWindowIndicator
{
	public: PriceSumIndicator(const std::int64_t windowLengthNs);
	const Price& record(const std::int64_t when, const Price i);

};

typedef std::shared_ptr<PriceSumIndicator> PPriceSumIndicator;

class PriceAverageIndicator:public PriceWindowIndicator
{
	Price windowSum_;
public:
	PriceAverageIndicator(const std::int64_t windowLengthNs);
	const Price& record(const std::int64_t when, const Price i );
};

typedef std::shared_ptr<PriceAverageIndicator> PPriceAverageIndicator;

class PriceStdevIndicator:public PriceWindowIndicator
{
	Price windowSum_;
public:
	PriceStdevIndicator(const std::int64_t windowLengthNs);
	const Price& record(const std::int64_t when, const Price i);
};

typedef std::shared_ptr<PriceStdevIndicator> PPriceStdevIndicator;

class PriceTshiftIndicator: public PriceWindowIndicator
{
	public:PriceTshiftIndicator(const std::int64_t windowLengthNs);
	const Price& record (const std::int64_t when, const Price i);
};
typedef std::shared_ptr<PriceTshiftIndicator> PPriceTshiftIndicator;