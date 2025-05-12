#include "OrderContainer.hxx"
#include "Utils.hxx"

#include <functional>
#include <numeric>
#include <algorithm>

const OrderContainer::PriceOrder OrderContainer::buySellPriceOrders_[2] = 
{
	[](const Price& l, const Price& r) -> bool{return l>r;},
	[](const Price& l, const Price& r) -> bool{return l<r;} };
const Price OrderContainer::noPriceConstants[2] = {Price(0.0), Price::max_value()};

OrderContainer::OrderContainer(Side buySell): ordersByPrice_(buySellPriceOrders_[buySell]), noPriceConstants_(noPriceConstants[buySell])
{

}

const POrder OrderContainer::findOrder(const Order::Id id) const
{
	auto findIt = allOrders_.find(id);
	return LIKELY(findIt!= allOrders_.end())> findIt->second.first:POrder();
}

bool OrderContainer::insertOrder(const Order& order)
{
	auto info = allOrders_.emplace(std::piecewise_construct, std::forward_as_tuple(order.id()), std::forward_as_tuple(std::make_shared<Order>(order), OrderQuene::iterator()));
	if(LIKELY(info.second))
	{
		auto& orderQ = ordersByPrice_[order.price()];
		auto where = orderQ.emplace(orderQ.end(),info.first->second.first);
		info.first->second.second = where;
		return true;
	}
	return false;

}

bool OrderContainer::removeOrder(AllOrders::iterator it)
{
	auto& orderInfo = it->second;
	auto findPriceIt = ordersByPrice_.find (orderInfo.first->price());
	if(LIKELY(findPriceIt!= ordersByPrice_.end()))
	{
		auto& orderQ = findPriceIt->second;
		orderQ.erase(orderInfo.second);
		if(orderQ.empty())
		{
			ordersByPrice_.erase(findPriceIt);
		}
	}
	allOrders_.erase(it);
}

bool OrderContainer::removeOrder(const Order::Id id)
{
	auto findIt = allOrders_.find(id);
	if(LIKELY(findIt != allOrders_.end()))
	{
		return removeOrder(findIt);
	}
	return false;
}

bool OrderContainer::handleMarketOrder(const Price& price, const Quantity quantity, const Quantity previousQuantity)
{
	if(UNLIKELY(!price)) return false;
	if(quantity>previousQuantity)
	{
		//add one order beginhd
		OrderQuene& orderQ = ordersByPrice_[price];
		if(!orderQ.empty()&& !orderQ.back()->isOurs())
		{
			orderQ.back()->addQuantity(quantity - previousQuantity);
		}
		else
		{
			auto order = std::make_shared<Order>(Order(price, quantity-previousQuantity, true));
			auto where = orderQ.emplace(orderQ.end(), order);
			allOrders_.emplace(std::piecewise_construct, std::forward_as_tuple(order->id()), std::forward_as_tuple(order, where));
		}
		return true;
	}
	else if (quantity <previousQuantity)
	{
		auto quantityToLose = previousQuantity-quantity;
		auto findPriceIt = ordersByPrice_.find(price);
		if(findPriceIt!= ordersByPrice_.end())
		{
			OrderQuene& orderQ = findPriceIt->second;
			for (OrderQuene::reverse_iterator qIt = orderQ.rbegin(); quantityToLose>0LL && qIt!= orderQ.rend();)
			{
				auto& pOrder = *qIt;
				if(!pOrder->isOurs())
				{
					auto lostQuantity = std::min(quantityToLose, pOrder->quantity());
					pOrder->reduceQuantity(lostQuantity);
					if(!pOrder->quantity())
					{
						allOrders_.erase(pOrder->id());
						orderQ.erase(std::next(qIt).base());
					}
					else
					{
						++qIt;
					}
					quantityToLose -=lostQuantity;
				}
				else
				{
					++qIt;
				}
			}
			if(orderQ.empty())
			{
				ordersByPrice_.erase(findPriceIt);
			}
			return true;
		}
	}
	return false;
}

Quantity OrderContainer::ourTotalQuantity() const
{
	return std::accumlate(allOrders_.begin(), allOrders_.end(), 0LL, [](Quantity l, const AllOrders::value_type & v)->Quantity{return l +v.second.first=>isOurs()*v.second.first->quantity();});
}

Price OrderContainer::ourTotalValue() const
{
	return std::accumlate(allOrders_.begin(), allOrders_.end(), Price(), [](const Price& l, const AllOrders::value_type & v )->Price({return l +v.second.first->isOurs()*v.second.first->quantity()*v.second.first->price();})); 
}

Price OrderContainer::ourBestPrice() const
{
	if(UNLIKELY(empty()))
	{
		return noPriceConstant_;
	}
	auto it = begin();
	for (; it!=end()&& !it->isOurs();)
	{
		++it;

	}

	return it!=end()>it->price():noPriceConstant_;
}

Price OrderContainer::marketBestPrice() const
{
	if(UNLIKELY(empty()))
	{
		return 0.0;
	}
	auto it = begin();
	for (; it!=end() && it->isOurs();)
	{
		++it;
	}
	return it!= end()? it->price():0.0;
}

Quantity OrderContainer::marketBestQuantity() const
{
	if(UNLIKELY(empty()))
	{
		return 0.0;
	}
	auto it = begin();
	for (; it!=end() && it->isOurs();)
	{
		++it;
	}
	Quantity q = 0;
	for (auto it2 = it; it2!=end()&& it2->price()==it->price();++it2)
	{
		q+=it2->quantity()*(1-it2->isOurs());
	}
	return q;
}
Quantity OrderContainer::marketQuantityAt(const Price& price) const
{
	auto findIt = ordersByPrice_.find(price);
	if(findIt!= ordersByPrice_.end())
	{
		return std::accumlate(findIt ->second.begin(), findIt->second.end(),0LL, [](Quantity l, const OrderQuene::value_type & v) ->Quantity{return l +(1 -v->isOurs())*v->quantity();});
	}
	return 0LL;
}

int OrderContainer::marketBookDepth() const
{
	int depth(0);
	for (auto it = ordersByPrice_.begin(); it!= ordersByPrice_.end();++it)
	{
		int hasMktOrders = 0;
		for (auto levelIt = it->second.begin(); !hasMktOrders&& levelIt!= it->second.end();++levelIt)
		{
			hasMktOrders = 1 -(*levelIt)->isOurs();
		}
		depth += hasMktOrders;
	}
	return depth;
}

OrderContainer::iterator OrderContainer::begin()
{
	return iterator(*this, ordersByPrice_.begin());
}

OrderContainer::iterator OrderContainer::end()
{
	return iterator(*this, ordersByPrice_.end());
}
OrderContainer::const_iterator OrderContainer::begin() const
{
	return const_iterator(*this, ordersByPrice_.begin());
}

OrderContainer::const_iterator OrderContainer::end() const
{
	return const_iterator(*this, ordersByPrice_.end());

}
OrderContainer::iterator OrderContainer::erase(iterator& it)
{
	iterator next = it;
	++next;
	allOrders_.erase((*it.it2_)->id());
	it.it1_->second.erase(it.it2);
	if(it.it1_->second.empty())
	{
		ordersByPrice_.erase(it.it1_);
	}
	return next;
}

OrderContainer::iterator OrderContainer::upper_bound(const Price& price)
{
	return iterator(*this, ordersByPrice_.upper_bound(price));
}

OrderContainer::const_iterator OrderContainer::upper_bound(const Price& price) const
{
	return const_iterator(*this, ordersByPrice_.upper_bound(price));
}
