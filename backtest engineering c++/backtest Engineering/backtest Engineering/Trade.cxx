#include "Trade.hxx"

Trade::Id Trade::idCache_=0;

Trade::Trade(Price p, Quantity q, Side buySell, std::int64_t when):id_(++idCache_), price_(p), quantity_(q), buySell_(buySell), when_(when)
{}

