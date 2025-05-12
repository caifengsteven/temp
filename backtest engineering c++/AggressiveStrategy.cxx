#include "AggressiveStrategy.hxx"

DECLARE_STRATEGY(AggressiveStrategy);

AggressiveStrategy::AggressiveStrategy(const std::string& name, const Quantity roundLotSize, const int verbosity)
		: StrategyBase(name, roundLotSize, verbosity)
{
	ADD_PARAMETER(Quantity, "order_quantity", order_quantity_, 100LL, "Max quantity of each order");
	ADD_PARAMETER(Price, "significant_return", significant_return_, 0.3,"Significant return to consider before considering an order sending");
	shortTermIndicator_ = tradingIndicator()->declareVariable ("return_term","10minreturn");
	ADD_PARAMETER(String, shortTermIndicator_->variableName(), shortTermIndicator_->sourceColumnName(),"10minreturn", "What is the term of the return to consider");
	fop_ = logIfChange(declareVariable("fishing_order_price"));

}
AggressiveStrategy::~AggressiveStrategy()
{

}
void AggressiveStrategy::onTrade(const PInstrument& i, const Side buySell, const Quantity quantity, const Price price, const std::int64_t when)
{

}
void AggressiveStrategy::onOrderAck(const PInstrument& i, const Order::Id id,const Side buySell, const Quantity remainingQuantity, const Price currentPrice, const std::int64_t when)
{
	if(currentFishingOrder_==id && !remainingQuantity)
	{
		currentFishingOrder_ = 0LL;

	}
}
void AggressiveStrategy::compute_(const PInstrument instrument, const std::int64_t when)
{
	if (!currentFishingOrder_&&when >blindTimeOut_ && instrument->isMarketOpen())
	{
		if (phase_== pWaitingOpportunity)
		{
			auto str = shortTermIndicator_->get();
			if(str>significant_return_)
			{
				if(currentFishingOrder_ = sendOrder(instrument, Buy,orderQuantity_, instrument->lastPrice(), when))
				{
					phase_ = pBull;
					blindTimeOut_ = when +shortTermReturnMins_*60LL*1000000000LL;
					LOG("\tOpening, buy "<<orderQuantity_<<'@'<<instrument->lastPrice());
					fop_->logIndicator(when, instrument->lastPrice());
				}
			}
			else if(str<-significant_return_)
			{
				if(currentFishingOrder_=sendOrder(instrument, Sell, orderQuantity_, instrument->lastPrice(), when))
				{
					phase_=pBear;
					blindTimeOut_ = when +shortTermReturnMins_*60LL *1000000000LL;
					LOG("\tOpening, sell "<<orderQuantity_<<'@'<<instrument->lastPrice());
					fop_->logIndicator(when, instrument->lastPrice());	
				}
			}

		}
		else
		{
			const auto side = phase_== pBull ? Sell:Buy;
			if (currentFishingOrder_=sendOrder(instrument, side, orderQuantity_, instrument->lastPrice(), when))
			{
					phase_=pWaitingOpportunity;
					LOG("\tClosing "<<(side==Sell) ?"sell ":"buy "<<orderQuantity_<<'@'<<instrument->lastPrice());
					fop_->logIndicator(when, instrument->lastPrice());	

			}
		}
	}
}
void AggressiveStrategy::onOrderBookUpdate(const PInstrument& instrument, const std::int64_t when)
{
	if(instrument->tradeJustOccurred())
	{
		compute_(instrument, when);
	}
}

void AggressiveStrategy::onTradingIndicatorUpdate(const PInstrument& instrument, const PTradingIndicator& tradingIndicator, const std::int64_t when)
{
	if(LIKELY(shortTermIndicator_->valueJustChanged()))
	{
		compute_(instrument, when);
	}
}

void AggressiveStrategy::onStrategyInit()
{
	if(1!= std::sscanf(shortTermIndicator_->sourceColumnName().c_str(),"%dminreturn", &shortTermReturnMins_))
	{
		shortTermReturnMins_=0;

	}
}
void AggressiveStrategy::onMarketPhaseUpdate(const PInstrument& instrument, const std::int64_t when)
{
	auto strPhase = Instrument::strMarketPhase(instrument->marketPhase());
	LOG("\tMarket phase changed to "<<strPhase<<" @ "<<readableTimeStamp(when));
	mps_->logIndicator(when, strPhase);
	if(instrument->isMarketOpen())
	{
		compute_(instrument, when);
	}

}