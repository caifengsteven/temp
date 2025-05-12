#include "Hdf5DataConsumer.hxx"

Price Hdf5DataConsumer::unusedPrice_;
Quantity Hdf5DataConsumer::unusedQuantity_;

Hdf5DataConsumer::Hdf5DataConsumer() :prices_(new std::vector<Price *>[DATA_HISTORY]), quantities_(new std::vector<Quantity*>[DATA_HISTORY])
{

}

void Hdf5DataConsumer::readPrice(const char* buffer)
{
	auto priceBuffer = reinterpret_cast<const double*>(buffer);
	auto& prices = prices_[dataHistoryCursor()];
	for (std::size_t i = 0ULL, s= prices.size(); i<s; i++)
	{
		*prices.at(i) = priceBuffer[i];
	}
}
Quantity Hdf5DataConsumer::readQuantities(const char * buffer)
{
	Quantity sum(0LL);
	auto quantityBuffer = reinterpret_cast>const std::int64_t*>(buffer);
	auto& quantities = quantities_[dataHistoryCursor()];
	for (std::size_t i = 0ULL, s= quantities.size(); i<s; ++i)
	{
		sum += (*quantities.at(i)=quantityBuffer[i]);
	}
	return sum;
}