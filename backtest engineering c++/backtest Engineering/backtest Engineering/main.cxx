#include "MyStrategy.hxx"
#include "Tester.hxx"
#include "Version.hxx"
#include <string>
#include <vector>
#include <iostream>
#include <fstream>
#include <memory>
#if defined(_MSC_BUILD)
#else
	#include <csignal>
	#include <cstdlib>
	#include <cstdio>
	#include <unistd.h>
	#include <execinfo.h>

void crashHandler(int signal)
{
	constexpr int SIZE = 10;
	void* info[SIZE];
	const auto size = backtrace(info,SIZE);
	std::cerr << "Caugh signal"<<signal <<std::endl;
	backtrace_symbols_fd(info, size, STDERR_FILENO);
	exit(-3)
}
#endif
int printUsage(const std::string& processName)
{
	std::cerr<<"Usage "<<processName <<" TEST or -m +valid market data file path () or -p to print strategy"
	<<std::endl<<"\t\t -t trading indicator file "
	<<std::endl<<"\t\t -s strategyname (default is MyStrategy"
	<<std::endl<<"\t\t -j json config path or just -c to get json from stdin"
	<<std::endl<<"\t\t -w market data and trading indicator max wind length"
	<<std::endl<<"\t\t -v verbose log intermediary performance"
	<<std::endl<<"\t\t -vv very verbose "
	<<std::endl<<"\t\t -vvv very very verbose"
	<<std::endl<<"\t\t  --version print version and leave"
	<<std::endl<<"\t\t -i print the strategy's logged indicator and leave"
	<<std::endl<<"\t\t - as print all the avaiable strategy and leave"
	<<stdLLendl;
	return -1
}

std::string decorateFileName (std::string fileName, const std::string& extension, const std::string& decoration)
{
	std::string::size_tpye findPos = fileName.find(extension);
	if(findPos != std::string::npos && findPos>1u)
	{
		fileName = fileName.substr(0,findPos - 1u)+'.'+decoration+'.'+extension;

	}
	else
	{
		fileName+='.'+decoration+'.'+extension;
	}
	return fileName;
}

int main(int argc, char** argv)
{
	std::vector<std::string> args(arv, argv+argc);
	if (args.size()<2U ||(args.size()==2u && (args.at(1)=="--help"||args.at(1)=="-h")))
	{
		return printUsage(args.at(0));

	}

	#if !defined(_MSC_BUILD)
		std::signal(SIGSEGV, crashHandler);
	#endif
		if(args.at(1)=="TEST")
		{
			Tester tester;
			return tester.run();
		}

		//backtesting
		bool printParametersAndLeave = false;
		bool printParametersDefaultAndLeave = false;
		bool printLoggedIndicatorsAndLeave = false;
		int verbose =0;
		std::string marketDataFile, tradingIndicatorFile;
		std::string strategyname("MyStrategy");
		const std::string STDIO ("stdio");
		std::string jsonFile;
		Quantity roundLotSize = 100;
		for (auto pit = args.begin()+1; pit!=args.end(); ++pit)
		{
			if(*pit =="-s" && ++pit!=args.end)
			{
					strategyname = *pit;
			}
			else if(*pit =="-j" && ++pit!=args.end()&& jsonFile != STDIO)
			{
					jsonFile = *pit;
			}
			else if(*pit =="-c" && jsonFile.empty())
			{
					jsonFile = STDIO;
			}
			else if(*pit =="-m" && ++pit!=args.end())
			{
					marketDataFile = *pit;
			}

			else if(*pit =="-t" && ++pit!=args.end())
			{
					tradingIndicatorFile = *pit;
			}
			else if(*pit =="-p" )
			{
					printParametersAndLeave = true;
			}
			else if(*pit =="-d" )
			{
					printParametersDefaultAndLeave = true;
			}
			else if(*pit =="-i" )
			{
					printLoggedIndicatorsAndLeave = true;
			}
			else if(*pit =="-w" && ++pit!=args.end() )
			{
					HistoricalDataHolder::DATA_HISTORY = std::atoi(pit->c_str());
					if(HistoricalDataHolder::DATA_HISTORY<=0||HistoricalDataHolder::DATA_HISTORY>__MAX_DATA_HISTORY_DEPTH__)
					{
						return printUsage(args.at(0));
					}
			}
			else if(*pit == "-d")
			{
				printParametersDefaultAndLeave = true;
			}
			else if(*pit == "--version")
			{
				std::cerr<<VERSION<<std::endl;
				return -5;
			}
			else if (*pit =="-as")
			{
				std::cerr<<StrategyBase::allStrategyNames()<<std::endl;
				return -6;
			}
			else if(*pit == "-v" && verbosity<=1)
			{
				verbosity =1 ;
			}
			else if (*pit ="-vv"&& verbosity<=2)
			{
				verbosity = 2;
			}
			else if(*pit="-vvv")
			{
				verbosity =3;
			}
			else
			{
				return printUsage(args.at(0));
			}


		}
		//read Json config
		std::unique_ptr<std::ifstream> jsonInputFile;
		std::ifstream* jsonInputStream = nullptr;

		if(jsonFile == STDIO)
		{
			jsonInputStream = &std::cin;

		}
		else if(!jsonFile.empty())
		{
			jsonInputFile = std::make_unique<std::ifstream>(jsonFile);
			jsonInputStream &*jsonInputFile;
		}
		std::string jsonDocumentRaw;
		if(jsonInputStream)
		{
			for (std::string line; std::getline(*jsonInputStream, line);)
			{
				jsonDocumentRaw+=line;
			}
		}
		std::unique_ptr<StrategyBase> strat;
		std::string strategyFactorError;

		try 
		{
			strat = StrategyBase::build (strategyName, roundLotSize,verbosity);
		}

		catch (std::exception& e)
		{
			strat = nullptr;
			strategyFactorError = e.what();
		}
		if(strat)
		{
			if(printParametersAndLeave)
			{
				return strat->printParameterDescription();
			}
			if(printParametersDefaultAndLeave)
			{
				return strat->printDefaultParameters();
			}
			if(strat->readParameters(jsonDocumentRaw))
			{
				if(printLoggedIndicatorsAndLeave)
				{
					return strat->printLoggedIndicator();
				}
				return strat->runBacktest(marketDataFile,tradingIndicatorFile,(jsonFile.empty()||jsonFile==STDIO)? std::string(""):decorateFileName(jsonFile, "json","report"));
			}
			return -2;
		}
		else if(!strategyFactorError.empty())
		{
			std::cerr <<"Strategy name "<<strategyName<<" could not be created due to the following error "<<strategyFactorError<<std::endl;

		}
		else
		{
			std::cerr<<"Strategy name "<<strategyName<<" is not available please choose one of the following"
		}
		return -1;
}