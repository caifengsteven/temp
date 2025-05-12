#include "Order.hxx"

Order::Id Order::idCache_ = 0;

Order::Order(Price p, Quantity q): id_(++idCache_), price_(p), quanity_(q), isOurs_(1)
{

}

Order::Order(const Order& toCopy):id_(toCopy.id_), price_(toCopy.price_), quanity_(toCopy.quanity_), isOurs_(toCopy.isOurs_)
{

}

Order::Order(Order&& toMove): id_(toMove.id_), price_(toMove.price_), quanity_(toMove.quanity_), isOurs_(toMove.isOurs_){}

Order::Order(Price p, Quantity q, bool isMarketOrder) : id_(-1* ++idCache_), price_(p), quanity_(q), isOurs_(0){}

void Order::reduceQuantity(const Quantity q)
{
	quanity_ -= q;

}

void Order::addQuantity(const Quantity q)
{
	quantity += q;
	
}