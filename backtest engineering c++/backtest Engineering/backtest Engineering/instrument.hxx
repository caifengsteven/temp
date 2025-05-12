#pragma once

#include "Configurable.hxx"
#include "HistoricalDataHolder.hxx"
#include "Hdf5DataConsumer.hxx"
#include "TickLadder.hxx"
#include "OrderContainer.hxx"
#include "Validity.hxx"

#include <cstdint>
#include <array>
#include <utility>
#include <string>
#include <vector>
#include <memory>
#include <numeric>
#include <map>
#include <deque>
#include <climits>
#include <unordered_map>
#include <ostream>

class StrategyBase;
class Instrument:virtual public HistoricalDataHolder
{
public:
	typedef std::int64_t Id;
	virtual ~Instrument(){}

	const Id id() const {return id_;}
	std::string ric() const {return ric_;}

	const Price& bestBid() const { return buySide().front().first;}
	const Price& bestAsk() const { return sellSide().front().first;}
	const Quantity bestBidQuantity() const {return buySide().front().second;}
	const Quantity bestAskQuantity() const {return sellSide().front().second;}
	const Price& lastPrice() const{return lastPrices_[dataHistoryCursor()];}
	const Quantity lastQuantity() const {return lastQuantities_[dataHistoryCursor()];}
	const Quantity lastQualifier() const {return lastQualifiers_[dataHistoryCursor()];}
	const bool isContinuousTrading() const {return !lastQualifier();}
	const Quantity accVol() const {return accVols_[dataHistoryCursor()];}
	static const int MARKET_DATA_DEPTH = 10;
	typedef std::array<std::pair<Price, Quantity>, MARKET_DATA_DEPTH> BookSide;
	const BookSide& buySide() const;
	const BookSide& sellSide() const;
	const BookSide& previousBuyBook(int lookBack =1 ) const;
	const BookSide& previousSellBook(int lookBack=1) const;
	const Price& previousLastPrice(int lookBack=1 ) const;
	const Quantity previousLastQuantity(int lookBack=1) const;
	const Quantity previousLastqualifier(int lookBack=1) const;
	const Price& previousBestBid(int lookBack=1) const {return previousBuyBook(lookBack).front().first;}
	const Price& previousBestAsk(int lookBack=1) const {return previousSellBook(lookBack).front().first;}
	const Quantity& previousBestBidSize(int lookBack=1) const {return previousBuyBook(lookBack).front().second;}
	const Quantity& previousBestAskSize(int lookBack=1) const {return previousSellBook(lookBack).front().second;}
	bool tradeJustOccurred() const {return previousAccVol()<accVol();}
	int checkConsistency(bool verbose, std::int64_t when) const;

	const Price ourBestBid() const;
	const Price ourBestAsk() const;
	const Quantity ourPendingBuy() const;
	const Quantity ourPendingSell() const;
	const Quantity ourPendingPosition() const;
	const Quantity ourPendingQuantity() const;

	const Quantity ourLiveBuy() const;
	const Quantity ourLiveSell() const;
	const Quantity ourLiveBuyOrSell() const;
	const Quantity ourLivePosition() const;
	const Quantity ourLiveQuantity() const;

	const Quantity ourPotentialBuyOrSell(const Side buySell) const;
	const Quantity ourPotentialPosition() const;
	const Quantity ourPotentialQuantity() const;

	const Price ourConservativeBestBuy() const;
	const Price ourConservativeBestSell() const;

	bool onFirstBuyLimit() const;
	bool onFirstSellLimit() const;
	bool aloneOnFirstBuyLimit() const;
	bool aloneOnFirstSellLimit() const;

	Price tickUp(Price price, int ticks =1 ) const;
	Price tickDown(Price price, int ticks =1) const;
	Price roundUp (Price price) const;
	Price roundDown(Price price) const;

	Quantity dayVolume() const;
	Quantity dayPosition() const;
	Price averageTradePrice() const;
	Price averageBuyPrice() const;
	Price averageSellPrice() const;
	Price tradePnL() const;
	Price realisedPnL() const;
	Price buyValue() const;
	Price sellValue() const;
	Quantity buyQuantity() const;
	Quantity sellQuantity() const;

	Quantity canTrade(const Side buySell, Quantity q) const;
	typedef enum{mpBeforeOpen, mpMorning, mpLunchBreak, mpAfternoon, mpClose, __mp_size__} MarketPhase;
	MarketPhase marketPhase() const
	{
		return marketPhase_;
	}

	static std::string strMarketPhase(MarketPhase mp);
	bool isMarketOpen() const;
protected:
	Instrument(StrategyBase& parent, const Quantity roundLotSize);
	static bool buySideMatching(const Price&, const Price&);
	static bool sellSizeMatching(const Price&, const Price&);

	BookSide& buySide();
	BookSide& sellSide();
	typedef std::array<BookSide,2> BuySellBook;
	BuySellBook& currentBuySellBook();
	Quantity canTrade(const Side buySell, Quantity q, const Quantity maxBuySellTradedQuantity, const Quantity maxAbsolutePosition, const Quantity maxVolume) const;

protected:
	StrategyBase& parent_;
	std::string ric_;
	Quantity roundLotSize_;
	Quantity contractSize_ =1;

	std::unique_ptr<BuySellBook[]> marketDataHistory_;
	std::unique_ptr<Price[]> lastPrices_;
	std::unique_ptr<Quantity[]> lastQuantities_;
	std::unique_ptr<Quantity[]> accVols_;
	std::unique_ptr<Quantity[]> lastQualifiers_;

	OrderContainer liveOrders_[2];
	OrderContainer ourPendingOrders_[2];

	typedef bool(*MatchingFP) (const Price&, const Price&);
	MatchingFP matchingFPs_[2];
	Price tradedValues_[2];
	Quantity tradedQuantities_[2];
	PTickLadder tickLadder_;
	MarketPhase marketPhase_=mpMorning;
	std::int64_t mpts_[__mp_size__ +1] = {0LL, std::numeric_limits<std::int64_t>::max(), std::numeric_limits<std::int64_t>::max(), std::numeric_limits<std::int64_t>::max(),std::numeric_limits<std::int64_t>::max(), std::numeric_limits<std::int64_t>::max()}

private:
	static Id idCache_;
	const Id id_;


};

typedef std::shared_ptr<Instrument> PInstrument;

inline std::ostream& operator<<(std::ostream& os, const Instrument::MarketPhase mp)
{
	os<<Instrument::strMarketPhase(mp);
	return os;

}


class RWInstrument:public Instrument, public Hdf5DataConsumer, public std::enable_shared_from_this<RWInstrument>, protected Configurable 
{
public:
	using Configurable::readParameterValues;
	RWInstrument(StrategyBase& parent, const Quantity roundLotSize);
	virtual ~RWInstrument(){}
	bool sendOrder(const Side buySell, const Order& order, const std::int64_t when);
	bool acceptOrder(const Side buySell, const Order& order, const std::int64_t when, const Validity validity);
	bool cancelOrder( const Order::Id id, const std::int64_t when);
	void manageOrderBookUpdate(const std::int64_t when);
	void accountTrade(const Side buySell, const Price& price, const Quantity quantity);
	const OrderContainer& liveOrders(Side side) const{return liveOrders_[side];}
	void setTickLadder(PTickLadder TickLadder);
	virtual void configurePrices(const std::vecor<std::string>& columnNames) override;
	virtual void configureQuantities(const std::vector<std::string>& columnNames) override;
	virtual void onInit() override;

};
typedef std::shared_ptr<RWInstrument> PRWInstrument;