#pragma once
#include "Price.hxx"
#include "Quantity.hxx"
#include "Side.hxx"

class Trade
{
public:
	Trade(Price p, Quantity q, Side buySell, std::int64_t when);
	typedef std::int64_t Id;
	const Id id() const {return id_;}
	const Price price() const {return price_;}
	Quantity Quantity() const {return quantity_;}
	Side buySell() const {return buySell_;}
	std::int64_t when() const {return when_;}
private:
	static Id idCache_;
	const Id id_;
	Price price_;
	Quantity quantity_;
	Side buySell_;
	const std::int64_t when_;
};
typedef std::shared_ptr<Trade> PTrade;