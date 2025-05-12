#include "Hdf5Reader.hxx"

Hdf5Reader::Hdf5Reader(const std::string& fileName) : fileName_(fileName)
{

}
void Hdf5Reader::init()
{
	file_ = std::make_shared<H5::H5File>(fileName_,H5F_ACC_RDONLY);
	group_ =std::make_shared<H5::Group>(file_->openGroup("data"));
}