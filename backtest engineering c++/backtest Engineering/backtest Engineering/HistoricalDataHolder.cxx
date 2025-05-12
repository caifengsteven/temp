#include "HistoricalDataHolder.hxx"

#include <algorithm>

int HistoricalDataHolder::DATA_HISTORY = _MAX_DATA_HISTORY_DEPTH__;
int HistoricalDataHolder::previousIndex_(int lookBack) const
{
	const int maxLookBack = DATA_HISTORY-1;
	lookBack = std::max(0, std::min(maxLookBack, lookBack));
	return (dataHistoryCursor_ + DATA_HISTORY - lookBack)% DATA_HISTORY;

}

void HistoricalDataHolder::shiftHistoryCursor()
{
	dataHistoryCursor_=(++dataHistoryCursor_)%DATA_HISTORY;
}