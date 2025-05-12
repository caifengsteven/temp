#pragma once























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

	}










