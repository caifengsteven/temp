#pragma once

#include "Price.hxx"
#include "Quantity.hxx"
#include "Side.hxx"

#include <cstdint>
#include <memory>

class RWInstrument;
class Order
{
public:
	Order(Price p, Quantity q);
	Order(const order& toCopy);
	Order(Order&& toMove);
	typedef std::int64_t Id;
	const Id id() const { return id_;}
	const Id completedId() const {return quantity_? 0LL: id_;}
	const Price price() const{return price_;}
	Quantity quantity() const {return quantity_;}
	const int isOurs() const {return isOurs_;}

private:
	friend class RWInstrument;
	friend class OrderContainer;
	Order(Price p, Quantity q, bool isMarketOrder);
	void reduceQuantity(const Quantity q);
	void addQuantity(const Quantity q);
private:
	static Id idCache_;
	const Id id_;
	Price price_;
	Quantity quantity_;
	const int isOurs_;
};

typedef std::shared_ptr<Order> POrder;
