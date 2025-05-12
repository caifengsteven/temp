#pragma once
#include "StrategyBase.hxx"
#include "SerialisableEnum.hxx"

class AggressiveStrategy: public StrategyBase
{
	public:
		AggressiveStrategy(const std::string& name,const Quantity roundLotSize, const int verbosity);
		virtual ~AggressiveStrategy();
		virtual void onTrade(const PInstrument& i, const Side buySell, const Quantity quantity, const Price price, const std::int64_t when) override;
		virtual void onOrderBookUpdate(const PInstrument& i, const std::int64_t when) override;
		virtual void onTradingIndicatorUpdate(const PInstrument& instrument, const PTradingIndicator& tradingIndicator, const std::int64_t when) override;
		virtual void onOrderAck(const PInstrument& i, const Order::Id id, const Side buySell, const Quantity remainingQuantity, const Price currentPrice, const std::int64_t when) override;
		virtual void onMarketPhaseUpdate(const PInstrument& instrument, const std::int64_t when) override;
	private:
		void compute_(const PInstrument i, const std::int64_t when);
		virtual void onStrategyInit() override;
	private: 
		Order::Id 		currentFishingOrder_ = 0;
		std::int64_t	blindTimeOut_ = 0;
		typedef enum 	{ pWaitingOpportunity,pBull,pBear} Phase;
		Phase 			phase_ = pWaitingOpportunity;
		Quantity 		orderQuantity_ = 100LL;
		Price 			significantReturn_ = 0.3;
		TradingIndicators:: PVariable shortTermIndicator_;
		int 			shortTermReturnmins_=0;
		PPriceLogIfChangeIndicator 	fop_;
		PStringIndicator 	mps_;



};