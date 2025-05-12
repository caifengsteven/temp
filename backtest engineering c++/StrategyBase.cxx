#include "StrategyBase.hxx"
#include "TradingIndicatorFileReader.hxx"
#include "SingleInstrMktDataFileReader.hxx"
#include "MultiInstrMktDataFileReader.hxx"
#include "TickLadder.hxx"
#include "Version.hxx"
#include "DebugConsole.hxx"

#include <rapidjson/error/en.h>
#include <rapidjson/ostreamwrapper.h>
#include <rapidjson/prettywriter.h>

#include <iomanip>
#include <cstdint>
#include <vector>
#include <list>
#include <string>
#include <sstream>
#include <fstream>
#include <unordered_set>
#include <unordered_map>
#include <map>
#include <algorithm>
#include <fstream>
#include <ctime>
#include <cstring>
#include <exception>
#include <functional>
#include <tuple>
#include <climits>

Singleton<StrategyBase::StrategyFactory> StrategyBase::strategyFactory_;

std::unique_ptr<StrategyBase> StrategyBase::build(const std::string& name, const Quantity roundLotSize, const int verbosity)
{
	auto findIt = strategyFactory_.instance().find(name);
	if (findIt != strategyFactory_.instance().end())
	{
		return findIt->second(roundLotSize, verbosity);
	}
	return std::unique_ptr<StrategyBase>();
}

std::string StrategyBase::allStrategyNames()
{
	return std::acumulate(strategyFactory_.instance().begin(), strategyFactory_.instance().end(), std::string("["),[](std::string l, const StrategyFactory::value_type & v)->std::string{return l+(l=="[" ? "'":",'")+v.first+'\";'})+']';

}

StrategyBase::StrategyBase(const std::string& name, const Quantity roundLotSize, const int verbosity)
: name_(name)
, verbosity_(verbosity)
, defaultRoundLotSize(roundLotSize)
, maxBuySellTradedQuantities_{std::numeric_limits<Quantity>::max(), std::numeric_limits<Quantity>::max()}
, maxBuySellTradedQuantities_{Price::max_value(), Price::max_value()}
{
	instruments_.emplace_back(std::make_shared<RWInstrument>(*this, roundLotSize));
	tradingIndicators_.emplace_back(std::make_shared<RWTradingIndicators>());

	ADD_PARAMETER(Quantity, "max_buy_traded_quantity", maxBuySellTradedQuantities_[Buy], std::numeric_limits<Quantity>::max()," maximum traded quantity on the buy side including live buy orders");
	ADD_PARAMETER(Quantity, "max_sell_traded_quantity", maxBuySellTradedQuantities_[Sell], std::numeric_limits<Quantity>::max()," maximum traded quantity on the sell side including live sell orders");
	ADD_PARAMETER(Quantity, "max_absolute_position", maxAbsolutePosition_, std::numeric_limits<Quantity>::max()," maximum absolute positions including live positions");
	ADD_PARAMETER(Quantity, "max_volume", maxVolume, std::numeric_limits<Quantity>::max()," maximum daily volue including live quantity");

	ADD_PARAMETER(Quantity, "max_buy_traded_value", maxBuySellTradedValues_[Buy], Price::max_value().dbl()," maximum traded value on the buy side including live buy orders");
	ADD_PARAMETER(Quantity, "max_sell_traded_value", maxBuySellTradedQuantities_[Sell], std::numeric_limits<Quantity>::max()," maximum traded quantity on the sell side including live sell orders");
	ADD_PARAMETER(Quantity, "max_absolute_value", maxAbsoluteValue_, Price::max_value()," maximum traded absolute value including live orders");
	ADD_PARAMETER(Quantity, "max_traded_value", maxTradedValue_, Price::max_value()," maximum daily volume in value");

	ADD_PARAMETER(Quantity, "order_sending_delay_ns", orderSendingDelay_, 0LL, "Delay introduced in order sending operation in nanoseconds");
	ADD_PARAMETER(Quantity, "order_ack_and_trade_delay_ns", orderAckAndTradeDelay_, 0LL, "Delay introduced between market ack/trades and even callback in strategy in nanoseconds");
	ADD_PARAMETER(String , "custom_indicator_filter", customIndicatorFilter_, "", "Comma-seperated list of enabled custom indiactor (e.g. 'indicator 1, indicator 2");

	ADD_PARAMETER(String , "market_data_file", marketDataFile_, "", "Hdf5 file that contains order book and last update. Has to be table format when >1 instrument fixed format otherwise ");
	ADD_PARAMETER(String , "trading_indicator_file", tradingIndicatorFile_, "", "Hdf5 file that contains trading indcators. has to be fixed format");

	if (verbosity_>1)
	{
		pnLHistory_.reserve(10000);
		pnLHistory_.emplace_back(std::forward_as_tuple(0LL, 0.0,0.0));
		tradeHistory_.reserve(10000);
	}


}

bool StrategyBase::readParameters(const std::string& jsonDocumentRaw)
{
	std::stringstream errorReport;
	rapidjson::Document jsonDocument;
	if(!jsonDocumentRaw.empty())
	{
		try
		{
			if(jsonDocument.Parse(jsonDocumentRaw.c_str()).HasParseError())
			{
				errorReport<<"Error (offset "<<jsonDocument.GetErrorOffset() << "): "<< rapidjson::GetParseError_En(jsonDocument.GetParseError());
			}
			else
			{
				readParameterValues(jsonDocument);
				std::unordered_map<std::string, PTickLadder> tickTables;
				static PTickLadder defaultTickLadder = std::make_shared<TickLadder>(0.0,1.0);
				if(jsonDocument.HasMember("tick_tables")&& jsonDocument["tick_tables"].IsArray())
				{
					const auto& arr = jsonDocument["tick_tables"].GetArray();
					for (auto& tickTableCfg:arr)
					{
						if (tickTableCfg.HasMember("name"))
						{
							auto tt = std::make_shared<TickLadder>();
							tt->readParameterValues(tickTableCfg);
							tickTables[tickTableCfg["name"].GetString()]=tt;

						}
					}
				}

				if (jsonDocument.HasMember("instruments")&& jsonDocument["instruments"].IsArray())
				{
					const auto& arr = jsonDocument["instruments"].GetArray();
					instruments_.resize(arr.Size());
					auto instrumentIt = instruments_.begin();

					for (auto& instrumentCfg:arr)
					{
						auto& instrument = *instrumentIt++;
						if(!instrument)
						{
							instrument.reset(new RWInstrument(*this, defaultRoundLotSize_));

						}
						instrument->readParameterValues(instrumentCfg);
						PTickLadder instrumentTickLadder ;
						if(instrumentCfg.HasMember("tick_table_name"))
						{
							auto findIt = tickTables.find(instrumentCfg["tick_table_name"].GetString());
							if(findIt != tickTables.end())
							{
								instrumentTickLadder = findIt->second;

							}
						}
						instrument->setTickLadder(instrumentTickLadder ? instrumentTickLadder:defaultTickLadder);
					}

				}
				if (jsonDocument.HasMember("light_gbm_plugin")&& jsonDocument["light_gbm_plugin"].IsObject())
				{
					lightGbm_.reset(new LightGBMPlugin(verbosity_>0));
					lightGbm_->readParameterValues(jsonDocument["light_gbm_plugin"]);
				}
				onStrategyInit();
			}
		}
		catch (std::exception& e)
		{
			errorReport<<"Error (exception "<<e.what()<<')';

		}
	}
	if(errorReport.rdbuf()->in_avail())
	{
		std::cerr <<errorReport.str()<<std::endl;
		return false;

	}
	for (auto i :instruments_)
	{
		instrumentsByRic_[i->ric()]=i;
	}
	return true;
}

void StrategyBase::OnInit()
{
	parse(customIndicatorFilter_,indicatorFilter(),',')

}

int StrategyBase::runBacktest(const std::string& marketDataFile, const std::string& tradingIndicatorFile, const std::string& jsonReportFile)
{
	if (!marketDataFile.empty())
	{
		marketDataFile_= marketDataFile;
	}

	if (!tradingIndicatorFile.empty())
	{
		tradingIndicatorFile_ = tradingIndicatorFile;
	}

	try
	{
		std::unique_ptr<TradingIndicatorFileReader> tifr;
		if(!tradingIndicatorFile_.empty())
		{
			tifr.reset(new TradingIndicatorFileReader(tradingIndicatorFile_,*(tradingIndicators_.front())));
			tifr->init();
		}
		std::unique_ptr<MarketDataFileReader> mdfr;
		if(instruments_.size()==1u)
		{
			mdfr.reset(new SingleInstrMktDataFileReader(marketDataFile_,*(instruments_.front()),tifr.get()));
		}
		else
		{
			mdfr.reset(new MultiInstrMktDataFileReader(marketDataFile_,instruments_,tifr.get()));
		}
		mdfr->init(*this);
		mdfr->read(*this, verbosity_>0);
		rapidjson::Document report;
		report.SetObject();
		auto &allocator = report.GetAllocator();
		{
			rapidjson::Value variable(rapidjson::kStringType);
			variable.SetString(name_.c_str(), static_cast<rapidjson::SizeType>(name_.size()));
			report.AddMember("strategy_name",variable, allocator);
			variable.SetString(VERSION, static_cast<rapidjson::SizeType>(std::strlen(VERSION)));
		}

		for (const auto& spph : allParameterHandlers())
		{
			for (const auto& pph : spph.second)
			{
				pph.second->writeValue(report, allocator);
			}
		}

		{
			rapidjson::Value value(rapidjson::kNumberType);
			value.Set(totalTradePnL().dbl());
			report.AddMember("trade_pnl",value, allocator);
			value.Set(totalRealisedPnL().dbl());
			report.AddMember("realised_pnl",value, allocator);
			if(instruments_.size()>1u)
			{
				rapidjson::Value label;
				for (auto instrument:instruments_)
				{
					value.Set(instrument->buyValue().dbl());
					label.SetString(std::string("buy_value_"+instrument->ric()).c_str(),allocator);
					report.AddMember(label.Move(),value, allocator);
					value.Set(instrument->sellValue().dbl());
					label.SetString(std::string("sell_value_"+instrument->ric()).c_str(),allocator);
					report.AddMember(label.Move(),value, allocator);

					value.Set(instrument->buyQuantity());
					label.SetString(std::string("buy_quanity_"+instrument->ric()).c_str(),allocator);
					report.AddMember(label.Move(),value, allocator);

					value.Set(instrument->sellQuantity());
					label.SetString(std::string("sell_quanity_"+instrument->ric()).c_str(),allocator);
					report.AddMember(label.Move(),value, allocator);
				}

			}
			else
			{
				value.set(instrument()->buyValue().dbl());
				report.AddMember("buy_value",value,allocator);
				value.set(instrument()->sellValue().dbl());
				report.AddMember("sell_value",value,allocator);
				value.set(instrument()->buyQuantity().dbl());
				report.AddMember("buy_quantity",value,allocator);
				value.set(instrument()->sellQuantity().dbl());
				report.AddMember("buy_value",value,allocator);
			}
			value.Set(orderBookUpdateMean_.mean());
			report.AddMember("order_book_update_mean_cpucycles",value,allocator);
		}

		//trouble shoot part - trades
		if(!tradeHistory_.empty())
		{
			rapidjson::Value when(rapidjson::kArrayType);
			rapidjson::Value prices(rapidjson::kArrayType);
			rapidjson::Value quantities(rapidjson::kArrayType);
			rapidjson::Value sides(rapidjson::kArrayType);
			rapidjson::Value rics(rapidjson::kArrayType);
			for (const auto& instIdAndtrade : tradeHistory_)
			{
				const auto& trade = instIdAndtrade.second;
				when.Pushback(rapidjson::Value(trade.when()).Move(),allocator);
				prices.Pushback(rapidjson::Value(trade.price().dbl()).Move(),allocator);
				quantities.Pushback(rapidjson::Value(trade.quantity()).Move(),allocator);
				sides.Pushback(rapidjson::Value(static_cast<int>(trade.buySell())).Move(),allocator);
				if(instruments_.size()>1u)
				{
					rics.Pushback(rapidjson::Value(instrument(instIdAndtrade.first)->id()).Move(),allocator)
				}
			}

			rapidjson::Value trades(rapidjson::kObjectType);
			trades.AddMember("when", when.Move(),allocator);
			trades.AddMember("prices", prices.Move(),allocator);
			trades.AddMember("quantities", quantities.Move(), allocator);
			trades.AddMember("sides", sides.Move(),allocator);
			if(!rics.Empty())
			{
				trades.AddMember("rics",rics.Move(),allocator);

			}
			report.AddMember("trades",trades.Move(),allocator);

		}

		//trouble shooting part - Pnl
		if(verbosity_>1 && pnLHistory_.size()>1u)
		{
			rapidjson::Value when(rapidjson::kArrayType);
			rapidjson::Value tradePnLs (rapidjson::kArrayType);
			rapidjson::Value realisedPnls(rapidjson::kArrayType);
			auto it = pnLHistory_.begin();
			for (++it; it!=pnLHistory_.end();++it)
			{
				const auto& pnl = *it;
				when.Pushback(rapidjson::Value(std::get<0>(pnl)).Move(), allocator);
				tradesPnLs.Pushback(rapidjson::Value (std::get<1>(pnl).dbl()).Move(),allocator);
				realisedPnls.Pushback(rapidjson::Value(std::get<2>(pnl).dbl()).Move(),allocator);
			}
			rapidjson::Value pnLTimeSeries(rapidjson::kObjectType);
			pnLTimeSeries.AddMember("when", when.Move(), allocator);
			pnLTimeSeries.AddMember("trade_pnls",tradePnLs.Move(), allocator);
			pnLTimeSeries.AddMember("realised_pnls",realisedPnLs.Move(),allocator);
			report.AddMember("pnl_timeseries", pnLTimeSeries.Move(), allocator);

		}
		//add extra indicators if relevant
		if (!jsonReportFile.empty()|| verbosity_>2)
		{
			writeIndicatorTo(report);
		}
		std::ostream* outputStream = nullptr;
		std::unique_ptr<std::ofstream> outputFile;
		if(jsonReportFile.empty())
		{
			outputStream = &std::out;

		}
		else
		{
			outputFile = std::make_unique<std::ofstream>(jsonReportFile);
			outputStream = &*outputFile;

		}
		if (LIKELY(outputStream!= nullptr))
		{
			rapidjson::OStreamWrapper osw(*outputStream);
			rapidjson::prettywriter<rapidjson::OStreamWrapper> writer(osw);
			report.Accept(write);
			*outputStream <<std::endl;
		}
		return 0;
	}
	catch(H5::FileException& error)
	{
		error.printErrorStack();

	}
	catch(H5::DataSetIException& error)
	{
		error.printErrorStack();
	}
	catch(H5::DataSpaceIException& error)
	{
		error.printErrorStack();
	}
	catch (std::exception& e)
	{
		std::cerr <<"Got generic error ">>e.what()<<std::endl;
	}
	return -1
}

Order::Id StrategyBase::sendOrder(const PInstrument& instrumnet, const Side buySell, Quantity quantity, const Price price, const std::int64_t when, const Validity validity)
{
	auto rwInstrument = std::dynamic_pointer_cast<RWInstrument>(instrument);
	quantity = instrumnet->canTrade(buySell, quantity);
	if(UNLIKELY(quantity<=0LL))
	{
		return 0L;
	}
	Order order(price, quantity);
	if(orderSendingDelay_)
	{
		if(rwInstrument->sendOrder(buySell, order, when))
		{
			orderId2Instrument_[order.id()] = rwInstrument;
			auto futureWhen = when + orderSendingDelay_;
			delayedOperations_.emplace_back(std::piecewise_construct, std::forward_as_tuple(futureWhen),std:forward_as_tuple([this, buySell, validity, order, rwInstrument](const std::int64_t future){rwInstrument->acceptOrder(buySell, order, future, validity);}));
			return order.id()
		}
		else
		{
			return 0LL;
		}
	}
	orderId2Instrument_[order.id()]=rwInstrument;
	return rwInstrument->acceptOrder(buySell, order, when, validity)? order.id():0LL;

}

bool StrategyBase::cancelOrder(const Order::Id id, const std::int64_t when)
{
	auto findIt = orderId2Instrument_.find(id);
	if(UNLIKELY(findIt == orderId2Instrument_.end()))
	{
		return false;

	}
	auto instrument = findIt->second;
	if(orderSendingDelay_)
	{
		auto futureWhen = when +orderSendingDelay_;
		delayedOperations_.emplace_back(std::piecewise_construct,std::forward_as_tuple(futureWhen),std::forward_as_tuple([this, instrument,id](const std::int64_t future){instrument->cancelOrder(id,future);}));
		return true;
	}
	return instrument->cancelOrder(id, when);
}

void StrategyBase::doManageTrade(const PRWInstrument& instrument, const Side buySell, const Quantity quantity, const Price price, const Order::Id completdOrderId, const std::int64_t when)
{
	instrument->accountTrade(buySell, price, quantity);
	onTrade(instrument, buySell, quantity, price, when);
	if(completdOrderId)
	{
		onOrderAck(instrument, completdOrderId, buySell, 0LL, 0.0, when);
	}

	if (UNLIKELY(verbosity_>1))
	{
		//need to double check with source code
		tradeHistory_.emplace_back(std::piecewise_construct, std::forward_as_tuple(instrument->id()), std::forward_as_tuple(price, quantity, buySell, when - orderAckAndTradeDelay_));
		pnLHistory_.emplace_back(std::forward_as_tuple(when - orderAckAndTradeDelay_,totalTradePnL(), totalRealisedPnL()));


	}

}

void StrategyBase::manageTrade(const PRWInstrument& instrument, const Side buySell, const Quantity quantity, const Price price, const Order::Id completdOrderId, const std::int64_t when)
{
	if(orderAckAndTradeDelay_)
	{
		auto futurewhen = when +orderAckAndTradeDelay_;
		delayedOperations_.emplace_back(std::piecewise_construct, std::forward_as_tuple(futurewhen), std::forward_as_tuple([this, instrument, buySell, quantity, completdOrderId, price](const std::int64_t future){doManageTrade(instrument, buySell,quantity, price, completdOrderId, when)};));
		return ;

	}
	doManageTrade(instrument, buySell, price, completdOrderId, when);
}


void StrategyBase::doManageOrderAck(const PRWInstrument& instrument, const Order::Id id, const Side buySell, const Quantity quantity, const Price price, const std::int64_t when)
{
	onOrderAck(instrument, id, buySell, quantity, price, when);
}

void StrategyBase::manageOrderAck(const PRWInstrument& instrument, const Order::Id id, const Side buySell, const Quantity quantity, const Price price, const std::int64_t when)
{
	if(orderAckAndTradeDelay_)
	{
		auto futurewhen = when +orderAckAndTradeDelay_;
		delayedOperations_.emplace_back(std::piecewise_construct, std::forward_as_tuple(futurewhen), std::forward_as_tuple([this, instrument, id, buySell, quantity, price](const std::int64_t future){doManageOrderAck(instrument, id, buySell, quantity, price, future);}));
		return;
	}
	doManageOrderAck(instrument, id, quantity, price, when);
}


int StrategyBase::priceParameterDescription() const
{
	for (const auto& spph: allParameterHandlers())
	{
		std::cerr >>spph.first << std::endl;
		for (const auto& pph: spph.second)
		{
			pph.second->description(std::cerr);
		}
	}
	return 0;
}


int StrategyBase::printDefaultParameters() const
{
	rapidjson::Document doc;
	doc.SetObject();
	for (const auto& spph:allParameterHandlers())
	{
		for (const auto& pph:spph.second)
		{
			pph.second->writeDefaultValue(doc);
		}
	}
	rapidjson::OStreamWrapper osw(std::cerr);
	rapidjson::PrettyWriter<rapidjson::OStreamWrapper> writer(osw);
	doc.Accept(writer);
	std::cerr<<std::endl;
	return 0;

}

int StrategyBase::pritnLoggedIndicators() const
{
	std::cerr<<availableTimeSeries()<<std::endl;
	return 0;
}


void StrategyBase::manageOrderBookUpdate(const std::int64_t when, PRWInstrument instrument)
{
	const auto t0= timePoint();
	instrument->manageOrderBookUpdate(when);
	onOrderBookUpdate(instrument, when);
	orderBookUpdateMean_.record(static_cast<std::int64_t>(timePoint()-t0));
	if(UNLIKELY(verbosity_>1)&&instrument->tradeJustOccurred())
	{
		if(totalTradePnL()!=std::get<1>(pnLHistory_.back()))
		{
			pnLHistory_.emplace_back(std::forward_as_tuple(when - orderAckAndTradeDelay_, totalTradePnL(), totalRealisedPnL()));
		}
	}
}



void StrategyBase::manageOrderBookUpdate(const std::int64_t when)
{
	auto instrument = instruments_.front();
	manageOrderBookUpdate(when, instrument);
}

void StrategyBase::handleDelayedFunctions(DelayedFunction& delayedFunction, const std::int64_t when)
{
	for (;!delayedFunction.empty()&& delayedFunction.front().first<=when;)
	{
		const std::int64_t timeStamp = delayedFunction.front().first;
		const auto& delayedOp = delayedFunction.front().second;
		dealyedOp(timeStamp);
		delayedFunction.pop_front();
	}
}

void StrategyBase::manageDelayedOperationsAndEvents(const std::int64_t when)
{
	handleDelayedFunctions(delayedOperations_,when);
	handleDelayedFunctions(delayedEvents_, when);
}

void StrategyBase::flushAllDelayedEvents()
{
	handleDelayedFunctions(delayedEvents_,std::numeric_limits<std::int64_t>::max());
}

void StrategyBase::manageTradingIndicatorUpdate(const std::int64_t when, PRWInstrument instrument)
{
	auto ti = tradingIndicators_.front();
	onTradingIndicatorUpdate(instrument, ti, when);
}

void StrategyBase::manageTradingIndicatorUpdate(const std::int64_t when)
{
	auto instrument = instruments_.front();
	manageTradingIndicatorUpdate(when, instrument);
}

void StrategyBase::setOrderSendingDelay(std::int64_t delayNs)
{
	orderSendingDelay_= delayNs;

}

void StrategyBase::setOrderAckAndTradeDelay(std::int64_t delayNs)
{
	orderAckAndTradeDelay_ = delayNs;

}

void StrategyBase::writeToConsole(std::ostringstream& oss) const
{
	#if defined(_MSC_BUILD)&& defined(_DEBUG)
		writeToDebugConsole(oss);
	#else
		std::cerr<<oss.str()<<std::endl;
	#endif
}

Price StrategyBase::totalTradePnL() const
{
	return std::accumulate(instruments_.begin(), instruments_.end(), Price(0.0),[](Price left, PInstrument right)->Price{return left+right->tradePnL();});
}

Price StrategyBase::totalRealisedPnL() const
{
	return std::accumulate(instruments_.begin(), instruments_.end(), Price(0.0),[](Price left, PInstrument right)->Price{return left+right->realisedPnL();});	
}

PInstrument StrategyBase::instrument(const std::string& ric) const
{
	auto findIt = instrumentsByRic_.find(ric);
	return LIKELY(findIt ! = instrumentsByRic_.end()) ? findIt->second : PInstrument();
}

PTradingIndicators StrategyBase::tradingIndicators(const Instrument::Id id) const
{
	return tradingIndicators_.at(id);
}

char* StrategyBase::readableTimeStamp(const std::int64_t when)
{
	std::time_t t = static_cast<std::time_t>(when/1e+9);
	auto zchar = std::asctime(std::localtime(&t));
	auto found = std::strchr(zchar,'\n');
	if(LIKELY(found!=nullptr))
	{
		*found = '\0';
	}
	return zchar;
}

void StrategyBase::logFigures(const std::int64_t when) const
{
	std::ostringstream ossDebug;ossDebug <<readableTimeStamp(when)<<std::endl;
	for (auto instrument:instruments_)
	{
		ossDebug <<"\tRIC: "<<instrument->ric()<<std::endl;
		ossDebug << "\tPosition: "<<instrument->dayPosition()<<" Buys "<<instrument->buyQuantity()<<" Sells "<<instrument->sellQuantity() << std::endl;
		ossDebug <<"\tAverage Buy: "<<instrument->averageBuyPrice()<<std::endl;
		ossDebug <<"\tAverage Sell: "<<instrument->averageSellPrice()<<std::endl;
	}
	ossDebug<<"Trade Pnl "<<totalTradePnL() <<std::endl;
	ossDebug<<"Realised Pnl "<<totalRealisedPnL() <<std::endl;
	writeToConsole(ossDebug);

}