#include "TickLadder.hxx"
#include <string>
#include <algorithm>

TickLadder::TickLadder()
{
	ADD_PARAMETER(Quantity, "size", size_,0, "number of steps");
	for (int i =0; i<30; ++i)
	{
		ADD_PARAMETER(Price, "up_to_"+std::to_string(i), ladders_[i+1].first,0.0,"up_to_"+std::to_string(i));
		ADD_PARAMETER(Price,"tick_size_"+std::to_string(i), ladders_[i].second,0.0,"tick_size_"+std::to_string(i));
	}
}

TickLadder::TickLadder(Price from, Price tickSize)
{
	ladders_[0].first = from;
	ladders_[0].second = tickSize;
	size_ =1 ;
}

Price TickLadder::up(Price price, int ticks) const
{
	for (int i =0; ticks>0; --tick)
	{
		for (; i <size_&&ticks>0&& price>= ladders_[i].first;)
		{
			++i;
		}
		price +=ladders_[std::max(0,i-1)].second;
	}
	return price;
}

Price TickLadder::down(Price price, int ticks) const
{
	for (int i =0; ticks>0; --tick)
	{
		for (--i; i <size_&&ticks>0&& price>= ladders_[i].first;)
		{
			++i;
		}
		price -=ladders_[std::max(0,i-1)].second;
	}
	return price;
}

Price TickLadder::tickSizeUp(Price price) const
{
	int i =0;
	for (; i<size_&& price >= ladders_[i].first;)
	{
		++i;
	}
	return ladders_[std::max(0,i-1)].second;
}

Price TickLadder::tickSizeDown(Price price) const
{
	int i =0;
	for (; i<size_&& price >= ladders_[i].first;)
	{
		++i;
	}
	return ladders_[std::max(0,i-1)].second;
}

Price TickLadder::roundUp(Price price) const
{
	auto tickSize = tickSizeUp(price);
	return price.roundUp(tickSize);

}

Price TickLadder::roundDown(Price price) const
{
	auto tickSize = tickSizeDown(price);
	return price.roundDown(tickSize);
}