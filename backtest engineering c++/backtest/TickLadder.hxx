#pragma once

#include "Configurable.hxx"

#include "Price.hxx"
#include "Quantity.hxx"

#include <memory>
#include <utility>

class TickLadder : protected Configurable
{
public:
	using Configurable::readParameterValues;
	TickLadder();
	TickLadder(Price from, Price tickSize);
	Price up(Price price, int ticks=1) const;
	Price down(Price price, int ticks =1 ) const;
	Price tickSizeUp(Price price) const;
	Price tickSizeDown(Price price) const;
	Price roundUp(Price price) const;
	Price roundDown(Price price) const;
private:
	std::pair<Price, Price> ladders_[31];
	Quantity size_=0;
};

typedef std::sthared_ptr<TickLadder> PTickLadder;