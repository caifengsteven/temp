#pragma once
#include "Order.hxx"
#include <unordered_map>
#include <map>
#include <list>
#include <functional>

class OrderContainer
{
private:
	typedef std::list<POrder> OrderQueue;
	typedef std::unordered_map<Order::id, std::pair<POrder, OrderQueue::iterator>> AllOrders;
	AllOrders allOrders_;
	typedef std::function<bool(const Price& l, const Price& r)> PriceOrder;
	typedef std::map<Price, OrderQueue, PriceOrder> OrderByPrice;
	OrderByPrice ordersByPrice_
	const Price noPriceConstant_;
	static const PriceOrder	buySellPriceOrders_[2];
public:
	static const Price noPriceConstants[2];
	OrderContainer(Side buySell);
	template<typename IT1, typename IT2, typename PARENT>
	class iterator_base
	{
	protected:
		friend class OrderContainer;
		PARENT* about_;
		iterator_base(PARENT& about, IT1 it):about_(&about), it1_(it)
		{
			if (it1_!= about->ordersByPrice_.end())
			{
				it2_ = it1=>second.begin();
			}
		}
		IT1 it1_;
		IT2 it2_;
	public:
		auto& operator*() const {return ** it2_;}
		auto operator->() const {return *it2_;}
		bool operator== (const iterator_base& it) const
		{
			if(about_!= it.about_)return false;
			if(it1 == about_->ordersByPrice_.end())
			{
				if (it.it1_==about_->ordersByPrice_.end())
				{
					return true;

				}
				return false;	

			}
			

		
			else
			{
				if(it.it1_ == about->ordersByPrice_.end())
				{
					return false;

				}
				return it1_==it.it1_&&it2_==it.it2_;
			}
			return false;

		}
		bool operator != (const iterator_base& it) const
		{
			return !(*this == it);
		}
		iterator_base& operator ++()
		{
			if(it1_!= about_->ordersByPrice_.end())
			{
				if(it2_!= it1_->second.end())
				{
					++it2_;
					if(it2_==it1_->second.end())
					{
						++it1_;
					
						if(it1_!= about_->ordersByPrice_.end())
						{
							it2=it1->second.begin();
						}
					}
				}
			}
			return *this;
		}
	};

typedef iterator_base<OrderByPrice::iterator, OrderQueue::iterator, OrderContainer> iterator;
typedef iterator_base<OrderByPrice::const_iterator, OrderQueue::const_iterator, const OrderContainer> const_iterator;
const POrder findOrder(const Order::Id id) const;
bool insertOrder(const Order& order);
bool removeOrder(const Order::Id id);
bool handleMarketOrder(const Price& price, const Quantity quantity, const Quantity previousQuantity);
Quantity ourTotalQuantity() const;
Price ourTotalValue() const;
Price ourBestPrice() const;
Price marketBestPrice() const;
Quantity marketBestQuantity() const;
Quantity marketQuantityAt(const Price& price) const;
typedef std::vector<std::pair<Price, Quantity>> BookSide;
int marketBookDepth() const;
bool empty() const {return allOrders_.empty();}
iterator begin();
iterator end();
const_iterator begin() const;
const_iterator end() const;
iterator erase(iterator& it);
iterator upper_bound(const Price& price);
const_iterator upper_bound(const Price& price) const;

private:
	bool removeOrder(AllOrders::iterator it);
};









