#include "Instrument.hxx"
#include "StrategyBase.hxx"
#include "Utils.hxx"

#include <algorithm>

bool Instrument::buySideMatching(const Price& candidatePrice, const Price& bookPrice)
{
	return candidatePrice <= bookPrice;
}

bool Instrument::sellSideMatching(const Price& candidatePrice, const Price& bookPrice)
{
	return candidatePrice>= bookPrice;
}

Instrument::Id Instrument::idCache_(0);
Instrument::Instrument(StrategyBase& parent, const Quantity roundLotSize): parent_(parent), roundLotSize_(roundLotSize), marketDataHistory_(new BuySellBook[DATA_HISTORY]), lastPrices_(new Price[DATA_HISTORY]), accVols_(new Quantity[DATA_HISTORY]), lastQualifiers_(new Quantity[DATA_HISTORY]), liveOrders_{OrderContainer(Buy), OrderContainer(Sell)}, ourPendingOrders_{OrderContainer(Buy), OrderContainer(Sell)}, matchingFPs_{buySideMatching, sellSideMatching}, tradedQuantities_{0,0}, id_(idCache_++)
{
	std::memset(lastQuantities_.get(),0,DATA_HISTORY*sizeof(Quantity));
	std::memset(accVols_.get(),0,DATA_HISTORY*sizeof(Quantity));
	std::memset(lastQualifiers_.get(),0,DATA_HISTORY*sizeof(Quantity));
}

const Price Instrument::ourBestBid() const
{
	return liveOrders_[Buy].ourBestPrice();
}

const Price Instrument::ourBestAsk() const
{
	return liveOrders_[Sell].ourBestPrice();
}

const Quantity Instrument::ourPendingBuy() const
{
	return ourPendingOrders_[Buy].ourTotalQuantity();
}

const Quantity Instrument::ourPendingSell() const
{
	return ourPendingOrders_[Sell].ourTotalQuantity();
}

const Quantity Instrument::ourPendingPosition() const
{
	return ourPendingBuy() - ourPendingSell();
}

const Quantity Instrument::ourPendingQuantity() const
{
	return ourPendingBuy() + ourPendingSell();

}

const Quantity Instrument::ourLiveBuy() const
{
	return ourLiveBuyOrSell(Buy);
}
const Quantity Instrument::ourLiveSell() const
{
	return ourLiveBuyOrSell(Sell);
}
const Quantity Instrument::ourLiveBuyOrSell( const Side buySell) const
{
	return liveOrders_[buySell].ourTotalQuantity();
}

const Quantity Instrument::ourLivePosition() const
{
	return ourLiveBuy() - ourLiveSell();
}

const Quantity Instrument::ourLiveQuantity() const
{
	return ourLiveBuy() + ourLiveSell();
}

const Quantity Instrument::OurPotentialBuyOrSell(const Side buySell) const
{
	return liveOrders_[buySell].ourTotalQuantity+ourPendingOrders_[buySell].ourTotalQuantity();
}

const Quantity Instrument::ourPotentialPosition() const
{
	return ourPendingPosition() + ourLivePosition();
}

const Quantity Instrument::ourPotentialQuantity() const
{
	return ourPendingQuantity() + ourLiveQuantity();

}

const Price Instrument::ourConservativeBestBuy() const
{
	Price result = liveOrders_[Buy].ourBestPrice();
	result = std::max(result, ourPendingOrders_[Buy].ourBestPrice());
	return result;
}

const Price Instrument::ourConservativeBestSell() const
{
	Price result = std::min(OrderContainer::noPriceConstants[Sell], liveOrders_[Sell].ourBestPrice());
	result = std::min(result, ourPendingOrders_[Sell].ourBestPrice());
	return result == OrderContainer::noPriceConstants[Sell]?0.0:result;

}

bool Instrument::onFirstBuyLimit() const
{
	auto ourBid = ourBestBid();
	return ourBid !=0.0 && ourBid >= bestBid();
}

bool Instrument::onFirstSellLimit() const
{
	auto ourAsk = ourBestAsk();
	return ourAsk !=0.0 && ourAsk <= bestAsk();
}

bool Instrument::aloneOnFirstBuyLimit() const
{
	auto ourBid = ourBestBid();
	return ourBid !=0.0&& ourBid > bestBid();

}

bool Instrument::aloneOnFirstSellLimit() const
{
	auto ourAsk = ourBestAsk();
	return (ourAsk!=0.0 && (ourAsk<bestAsk()||bestAsk()==0.0));
}

Quantity Instrument::dayVolume() const
{
	return buyQuantity() +sellQuantity();
}

Quantity Instrument::dayPosition() const
{
	return buyQuantity() - sellQuantity();
}

Price Instrument::averageTradePrice() const
{
	return dayVolume() ? (buyValue()+sellValue())/dayVolume():0.0;
}

Price Instrument::averageBuyPrice() const
{
	return buyQuantity()> buyValue()/buyQuantity():0.0;
}

Price Instrument::averageSellPrice() const

{
	return sellQuantity()? sellValue()/sellQuantity():0.0;
}

Price Instrument::tradePnL() const
{
	return(lastPrice()-averageTradePrice())*dayPosition();
}

Price Instrument::realisedPnL() const
{
	//need to check
}

Price Instrument::buyValue() const
{
	return tradedValues_[Buy];

}

Price Instrument::sellValue() const
{
	return tradedValues_[Sell];
}

Quantity Instrument::buyQuantity() const
{
	return tradedQuantities_[Buy];
}
Quantity Instrument::sellQuantity() const
{
	return tradedQuantities_[Sell];

}

Quantity Instrument::canTrade(const side buySell, Quantity q, const Quantity maxBuySellTradedQuantity, const Quantity max)//need to check
{
	//need to check
}

Quantity Instrument::canTrade(const Side buySell, Quantity q) const
{
	q = canTrade(buySell, q, parent_.maxBuySellTradedQuantities(buySell), parent_.maxAbsolutePosition(), parent_.maxVolum());
	auto last = lastPrice();

	if (LIKELY(!!last))
	{
		auto factor = last * contractSize_;
		//need to check
	}
	return q;
}

bool Instrument::isMarketOpen() const
{
	return (marketPhase() & 1)!=0;
}

Instrument::BookSide& Instrument::buySide()
{
	return marketDataHistory_[dataHistoryCursor()][Buy];
}

Instrument::BookSide& Instrument::sellSide()
{
	return marketDataHistory_[dataHistoryCursor()][Sell];	
}

const Instrument::BookSide& Instrument::buySide() const
{
	return marketDataHistory_[dataHistoryCursor()][Buy];
}

const Instrument::BookSide& Instrument::sellSide() const
{
	return marketDataHistory_[dataHistoryCursor()][Sell];
}

Instrument::BuySellBook& Instrument::currentBuySellBook()
{
	return marketDataHistory_[dataHistoryCursor()];

}


const Instrument::BookSide& Instrument::previousBuyBook(int lookBack) const
{
	return marketDataHistory_[previousIndex_(lookBack)].at(Buy);
}

const Instrument::BookSide& Instrument::previousSellBook(int lookBack) const
{
	return marketDataHistory_[previousIndex_(lookBack)].at(Sell);
}

const Price& Instrument::previousLastPrice(int lookBack) const
{
	return lastPrice_[previousIndex_(lookBack)];
}

const Quantity Instrument::previousLastQuantity(int lookBack) const
{
	return lastQuantity_[previousIndex_(lookBack)];
}

const Quantity Instrument::previousLastQulifier(int lookBack) const
{
	return lastQualifiers_[previousIndex_(lookBack)];
}
const Quantity Instrument::previousAccVol(int lookBack) const
{
	return accVols_[previousIndex_(lookBack)];
}

Price Instrument::tickUp(Price price, int ticks) const
{
	return tickLadder_=>up(price, ticks);
}

Price Instrument::tickDown(Price price, int ticks) const
{
	return tickLadder_->down(price, ticks);

}

Price Instrument::roundUp(Price price) const
{
	return tickLadder_->roundUp(price);

}

Price Instrument::roundDown(Price price) const
{
	return tickLadder_->roundDown(price);

}

int Instrument::checkConsistency(bool verbose, std::int64_t when) const
{
	int result =0;
	bool buyIsConsistent = true;
	for (auto bsIt = buySide().begin(); buyIsConsistent && bsIt!= buySide().end(); ++bsIt)
	{
		buyIsConsistent =(liveOrders_[Buy].marketQuantityAt(bsIt->first)==bsIt->second);
	}

	buyIsConsistent = buyIsConsistent && liveOrders_[Buy].marketBookDepth()<=MARKET_DATA_DEPTH;
	if(!buyIsConsistent)
	{
		result |= 1 <<Buy;
		if(UNLIKELY(verbose))
		{
			std::cerr <<when<< " : mkt data best bid = "<<bestBidQuantity() << '@' <<bestBid()<<std::endl;
			for (int h=0; h<DATA_HISTORY;++h)
			{
				for (auto& pbo:previousBuyBook(h))
				{
					std::cerr <<' '<<pbo.second <<'@'<<pbo.first;

				}
				std::cerr<<' '<<previousLastQuantity(h)<<'@'<<prviousLastPrice(h)<<" AV = "<<previousAccVol(h)<<std::endl;
			}
		}
	}
	bool sellIsConsistent = true;
	for (auto ssIt = sellSide().begin(); sellIsConsistent&& ssIt!= sellSide().end(); ++ssIt)
	{
		sellIsConsistent = (liveOrders_[Sell].marketQuantityAt(ssIt->first)==ssIt->second);

	}

	sellIsConsistent = sellIsConsistent&& liveOrders_[Sell].marketBookDepth()<=MARKET_DATA_DEPTH;
	if(!sellIsConsistent)
	{
		result|=1<<Sell;
		if(UNLIKELY(verbose))
		{
			std::cerr<<when<<" : market data best ask= "<<bestAskQuantity()<<'@'<<bestAsk()<<" != order book mkt best ask =" <<std::endl;
			for (int h =0;h<DATA_HISTORY; ++h)
			{
				for (auto& pbo:previousSellBook(h))
				{
					std::cerr<<' '<<pbo.second<<'@'<<pbo.first;
				}
				std::cerr<<' '<<previousLastQuantity(h)<<'@'<<previousLastPrice(h)<<" AV = "<<previousAccVol(h)<<std::endl;
			}
		}
	}
	return result;
}
std::string Instrument::strMarketPhase(MarketPhase mp)
{
	static const std::string strPhases[Instrument::__mp_size__+1]={"BeforeOpening", "Morning", "LunchBreak","Afternoon", "Close"};
	return strPhases[std::min(mp, __mp_size__)];
}

RWInstrument::RWInstrument(StrategyBase& parent, const Quantity roundLotSize):Instrument(parent, roundLotSize)
{
	ADD_PARAMETER(String, "ric", ric_, "", "Instrument RIC identifier");
	ADD_PARAMETER(Quantity, "round_lot_size", roundLotSize, 100LL, "Instrument round lot size");
	ADD_PARAMETER(Quantity, "contract_size", contractSize_, 1LL, "Instrument contract size");
	ADD_PARAMETER(Quantity, "morning_open", mpts_[mpMorning],0, "Nanosecond time stamp of market open, morning session");
	ADD_PARAMETER(Quantity, "morning_close", mpts_[mpLunchBreak], std::numeric_limits<Quantity>::max(), "Nanosecond time stamp of lunch break");
	ADD_PARAMETER(Quantity, "afternoon_open", mpts_[mpAfternoon], std::numeric_limits<Quantity>::max(), "Nanosecond time stamp of afternoon_open");
	ADD_PARAMETER(Quantity, "afternoon_close", mpts_[mpClose], std::numeric_limits<Quantity>::max(), "Nanosecond time stamp of close");
}

void RWInstrument::configurePrices(const std::vector<std::string>& columns)
{
	for (int h =0; h<DATA_HISTORY;++h)
	{
		auto& prices = prices_[h];
		prices.resize(columns.size(), nullptr);
		for (std::size_t i =0ULL, s= columns.size(); i<s; ++i)
		{
			if (columns.at(i)=="bid")
			{
				prices[i] = &unusedPrices_;
			}
			else if (columns.at(i)=="ask")
			{
				prices[i]= &unusedPrices_;
			}
			else if (columns.at(i).find("bidprice")!=std::string::npos)
			{
				int pos(0);
				if(1==std::sscanf(columns.at(i).c_str(), "1%dbidprice", &pos)&& pos>0&& pos<= MARKET_DATA_DEPTH)
				{
					prices[i]=&marketDataHistory_[h].at(Buy).at(pos-1).first;
				}
			}
			else if (columns.at(i).find("askprice")!=std::string::npos)
			{
				int pos(0);
				if(1==std::sscanf(columns.at(i).c_str(), "1%daskprice", &pos)&& pos>0&& pos<= MARKET_DATA_DEPTH)
				{
					prices[i]=&marketDataHistory_[h].at(Buy).at(pos-1).first;
				}
			}
			else if (columns.at(i)=="price")
			{
				prices[i] = &lastPrices_[h];
			}
			else
			{
				throw std::runtime_error("Sorry but this column "+columns.at()+ " is not handled in RWInstrument");
			}

		}

	}
}

void RWInstrument::configureQuantities(const std::vector<std::string>& columns)
{

	for (int h =0; h<DATA_HISTORY;++h)
	{
		auto& quantities = quantities_[h];
		quantities.resize(columns.size(), nullptr);
		for (std::size_t i =0ULL, s= columns.size(); i<s; ++i)
		{
			if (columns.at(i)=="bidsize")
			{
				quantities[i] = &unusedQuantity_;
			}
			else if (columns.at(i)=="asksize")
			{
				quantities[i]= &unusedQuantity_;
			}
			else if (columns.at(i).find("bidsize")!=std::string::npos)
			{
				int pos(0);
				if(1==std::sscanf(columns.at(i).c_str(), "1%dbidsize", &pos)&& pos>0&& pos<= MARKET_DATA_DEPTH)
				{
					quantities[i]=&marketDataHistory_[h].at(Buy).at(pos-1).second;
				}
			}
			else if (columns.at(i).find("asksize")!=std::string::npos)
			{
				int pos(0);
				if(1==std::sscanf(columns.at(i).c_str(), "1%dasksize", &pos)&& pos>0&& pos<= MARKET_DATA_DEPTH)
				{
					quantities[i]=&marketDataHistory_[h].at(Buy).at(pos-1).second;
				}
			}
			else if (columns.at(i)=="size")
			{
				quantities[i] = &lastQuantities_[h];
			}
			else if (columns.at(i)== "accvol")
			{
				quantities[i]= &accVols_[h];

			}
			else if (columns.at(i)== "condtrade")
			{
				quantities[i]=&lastQualifiers_[h];
			}
			else
			{
				throw std::runtime_error("Sorry but this column "+columns.at()+ " is not handled in RWInstrument");

			}

		}

	}



}

void RWInstrument::onInit()
{
	if(mpts_[mpMorning]!=std::numeric_limits<std::int64_t>::max())
	{
		marketPhase_ = mpBeforeOpen;
	}
}

bool RWInstrument::sendOrder(const Side buySell, const Order& order, const std::int64_t when)
{
	return ourPendingOrders_[buySell].insertOrder(order);
}

bool RWInstrument::acceptOrder(const Side buySell, const Order& order, const std::int64_t when, const Validity validity)
{
	ourPendingOrders_[buySell].removeOrder(order.id());
	auto oppositeSide = static_cast<Side> (1-buySell);

	auto self = shared_from_this();
	auto remainingQuantity = order.quantity();
	auto& oppositeSideLiveOrders = liveOrders_[oppositeSide];
	Price worstMarketPriceOppSide;
	Quantity worstMarketQuantityOppSide(0);

	for (auto oppIt = oppositeSideLiveOrders.begin(); oppIt!= oppositeSideLiveOrders.end()&& remainingQuantity>0LL && matching)//not end yt
	{
		auto matchQuantity = std::min(remainingQuantity, oppIt->quantity());
		if(UNLIKELY(validity == vFOK && matchQuantity != remainingQuantity))
		{
			//FOK failed
			return true;
		}

		remainingQuantity -= matchQuantity;
		parent_.manageTrade(self, oppositeSide, matchQuantity, oppIt->price(), oppIt->completedId(), when);
		if(!oppIt->quantity())
		{
			oppIt = oppositeSideLiveOrders.erase(oppIt);
		}
		else
		{
			++oppIt;
		}
	}
	else
	{
		worstMarketPriceOppSide = oppIt->price();
		worstMarketQuantityOppSide = oppIt->quantity();
		++oppIt;
	}

	if(remainingQuantity&& validity== vDay)
	{
		if(worstMarketQuantityOppSide)
		{
			for (;remainingQuantity;)
			{
				auto matchQuantity = std::min(remainingQuantity, worstMarketQuantityOppSide);
				remainingQuantity -= matchQuantity;
				parent_.manageTrade(self, buySell, matchQuantity, worstMarketPriceOppSide, remainingQuantity?0LL:order.id, when);
			}
		}
		else
		{
			liveOrders_[buySell].insertOrder(order);
			parent_.manageOrderAck(self, order.id(), buySell, order.quantity(), order.price(),when);
		}
		return true;
	}

	parent_.manageOrderAck(self, order.id(), buySell, 0LL, 0.0, when);
	return validity;

}

bool RWInstrument::cancelOrder(const Order::Id id, const std::int64_t when)
{
	if(outPendingOrders_[Buy].removeOrder(id))
	{
		return true;
	}
	else if(ourPendingOrders_[Sell].removeOrder(id))
	{
		return true;
	}
	else if(liveOrders_[Buy].removeOrder(id))
	{
		parent_.manageOrderAck(shared_from_this(),id, Buy,0,0.0,when);
		return true;
	}

	else if(liveOrders_[Sell].removeOrder(id))
	{
		parent_.manageOrderAck(shared_from_this(),id, Sell, 0,0.0, when);
		return true;
	}
	return false;

}

void RWInstrument::manageOrderBookUpdate(const std::int64_t when)
{
	auto self = shared_from_this();
	if(UNLIKELY(when >= mpts_[marketPhase_+1]))
	{
		int& mp = *reintepret_cast<int*>(&marketPhase_);
		++mp;
		parent_.onMarketPhaseUpdate(self, when);
	}

	// 1 - match own orders vs market and add market orders in live books

	const auto& previousBuy = previousBuyBook();
	const auto& previousSell = previousSellBook();
	int p_buy_i =0;
	int p_sell_i=0;
	auto& liveBuyOrders = liveOrders_[Buy];
	auto& liveSellOrders = liveOrders_[Sell];
	Price mktBuyPrice, worstMktBuyPrice;
	Price mktSellPrice, worstMktSellPrice(Price::max_value());

	for (int i =0; i<MARKET_DATA_DEPTH; ++i)
	{
		mktBuyPrice = buySide()[i].first;
		const Quantity mktBuyQuantity = buySide()[i].second;
		mktSellPrice = sellSide()[i].first;
		const Quantity mktSellQuantity = sellSide()[i].second;
		if (mktBuyQuantity&& !!mktBuyPrice)
		{
			worstMktBuyPrice = mktBuyPrice;
			for (; p_buy_i < MARKET_DATA_DEPTH && previousBuy[p_buy_i].first>mktBuyPrice;)
			{
				liveBuyOrders.handleMarketOrder(previousBuy[p_buy_i].first, 0LL, previousBuy[p_buy_i].second);
				++p_buy_i;
			}
			if(p_buy_i<MARKET_DATA_DEPTH && previousBuy[p_buy_i].first == mktBuyPrice)
			{
				liveBuyOrders.handleMarketOrder(mktBuyPrice, mktBuyQuantity, previousBuy[p_buy_i].second);
				++p_buy_i;
			}
			else
			{
				liveBuyOrders.handleMarketOrder(mktBuyPrice, mktBuyQuantity, 0LL);
			}

			//execute our sell side if relevant
			for (auto sellIt = liveSellOrders.begin(); sellIt!= liveSellOrders.end()&& mktBuyPrice>=sellIt->price();)
			{
				if(!sellIt->isOurs())
				{
					++sellIt;
					continue;
				}
				while(sellIt->quantity())
				{
					auto matchQuantity = std::min(mktBuyQuantity, sellIt->quantity());
					sellIt->reduceQuantity(matchQuantity);
					parent_.manageTrade(self, Sell, matchQuantity, sellIt->price(), sellIt->completedId(), when);

				}
				sellIt = liveSellOrders.erase(sellIt);
			}
		}

		//handle market orders cancel when is more aggressive than current best
		if (mktSellQuantity&& !! mktSellPrice)
		{
			worstMktSellPrice = mktSellPrice;
			for (; p_sell_i<MARKET_DATA_DEPTH && previousSell[p_sell_i].second&& previousSell[p_sell_i].first <mktSellPrice;)
			{
				liveSellOrders.handleMarketOrder(previousSell[p_sell_i].first, 0LL, previousSell[p_sell_i].second);
				++p_sell_i;
			}
			if(p_sell_i<MARKET_DATA_DEPTH && previousSell[p_sell_i].first == mktSellPrice)
			{
				liveSellOrders.handleMarketOrder(mktSellPrice, mktSellQuantity, previousSell[p_sell_i].second);
				++p_sell_i;
			}
			else
			{
				liveSellOrders.handleMarketOrder(mktSellPrice, mktSellQuantity, 0LL);
			}

			for (auto buyIt = liveBuyOrders.begin(); buyIt!=liveBuyOrders.end()&& mktSellPrice<=buyIt->price();)
			{
				if(!buyIt->isOurs())
				{
					++buyIt;
					continue;
				}

				while(buyIt->quantity())
				{
					auto matchQuantity = std::min(mktSellQuantity, buyIt->quantity());
					buyIt->reduceQuantity(matchQuantity);
					parent_.manageTrade(self, Buy, matchQuantity, buyIt->price(), buyIt->completedId(), when);
				}
				buyIt = liveBuyOrders.erase(buyIt);
			}
		}
	}
	for (; p_buy_i <MARKET_DATA_DEPTH; ++p_buy_i)
	{
		liveBuyOrders.handleMarketOrder(previousBuy[p_buy_i].first, 0LL, previousBuy[p_buy_i].second);
	}

	for (; p_sell_i<MARKET_DATA_DEPTH; ++p_sell_i)
	{
		liveSellOrders.handleMarketOrder(previousSell[p_sell_i].first, 0LL, previousSell[p_sell_i].second);


	}

	for (auto it = liveBuyOrders.upper_bound(worstMktBuyPrice); it!=liveBuyOrders.end();)
	{
		parent_.manageOrderAck(self, it->id(), Buy, 0LL,0.0, when);
		it = liveBuyOrders.erase(it);
	}

	for (auto it = liveSellOrders.upper_bound(worstMktSellPrice); it!=liveSellOrders.end();)
	{
		parent_.manageOrderAck(self, it->id(), Sell, 0LL,0.0, when);
		it = liveSellOrders.erase(it);
	}

	// 2 - anticipate market behaviour if last has changed and is coming to us
	if (accVol()>previousAccVol())
	{
		//last moved take out our orders if enough quantity in last if we are ahead in the queue
		auto lastQuantiyToMatch = lastQuantity();
		if(lastPrice()<=ourBestBid())
		{
			for (auto it = liveOrders_[Buy].begin(); lastQuantiyToMatch>0LL && it !=liveOrders_[Buy].end()&& )// need to check 
			{
				auto matchedQuantity = std::min(lastQuantiyToMatch,it->quantity());
				if(it->isOurs())
				{
					it->reduceQuantity(matchedQuantity);
					parent_.manageTrade(self, Buy, matchedQuantity, it->price(), it->completedId(), when);
					if(!it->quantity())
					{
						it = liveOrders_[Buy].erase(it);
					}
					else
					{
						++it;
					}
				}
				else
				{
					++it;
				}
				lastQuantiyToMatch -= matchedQuantity;
			}
		}
		else if(lastPrice()>= ourBestAsk())
		{
			for (auto it = liveOrders_[Sell].begin(); lastQuantiyToMatch>0LL && it != liveOrders_[Sell].end()&& it->price()) //need to check
			{
				auto matchedQuantity = std::min(lastQuantiyToMatch,it->quantity());
				if(it->isOurs())
				{
					it->reduceQuantity(matchedQuantity);
					parent_.manageTrade(self, Sell, matchedQuantity, it->price(), it->completedId(), when);
					if(!it->quantity())
					{
						it = liveOrders_[Sell].erase(it);
					}
					else
					{
						++it;
					}
				}
				else
				{
					++it;
				}
				lastQuantiyToMatch -= matchedQuantity;
			}

		}
	}

}


void RWInstrument::accountTrade(const Side buySell, const Price& price, const Quantity quantity)
{
	tradedValues_[buySell] +=(price*quantity*contractSize_);
	tradedQuantities_[buySell] ==quantity;
}

void RWInstrument::setTickLadder(PTickLadder tickLadder)
{
	tickLadder_ = tickLadder;
}