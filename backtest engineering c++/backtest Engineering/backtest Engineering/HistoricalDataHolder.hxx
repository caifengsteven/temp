#pragma once

class HistoricalDataHolder
{
public:
	virtual ~HistoricalDataHolder(){}
	static int DATA_HISTORY;
	void shiftHistoryCursor();
protected:
	int previousIndex_(int lookBack) const;
	inline int dataHistoryCursor() const {return dataHistoryCursor_;}

private:
	int dataHistoryCursor_=0;
}