#include "LightGBMPlugin.hxx"

#include "Utils.hxx"

#if defined(_MSC_BUILD)
#include <windows.h>
#else
#include <dlfcn.h>
#endif 

#include <iostream>
#include <sstream>
#include <exception>

LightGBMPlugin::LightGBMPlugin(bool verbose):verbose_(verbose)
{
	ADD_PARAMETER(String, "dll_path", dllPath_, "", "Path to LightGBM DLL");
	ADD_PARAMETER(String, "booster_model_path", boosterModelPath_, "", "Path to LightGBM booster model file");
}

template <typename FUNCTION>
inline auto& LightGBMPlugin::function_(FUNCTION& function)
{
	#if defined (_MSC_BUILD)
	return function;
	#else
	return * function;
	#endif
}

LightGBMPlugin::~LightGBMPlugin()
{
	if(handle_)
	{
		if(LGBM_BoosterFree_&& handle_)
		{
			callAndCheckError_(function_(LGBM_BoosterFree_)(boosterHandle_), "destruction of light GBM wrapper");
		}
	#if defined(_MSC_BUILD)
		::FreeLibrary(handle_);
	#else
		dlclose(handle_);
	#endif
	}
}


inline void LightGBMPlugin::callAndCheckError_(int errCode, const char* context) const
{
	if(UNLIKELY(errCode && LGBM_GetLastError_))
	{
		std::ostringstream oss;
		oss<<"Problem when calling Light GBM funtion during "<<context<<" error= "<<function_(LGBM_GetLastError_)();
		throw std::runtime_error(oss.str());

	}
}

void LightGBMPlugin::onInit()
{
	if (!handle_)
	{
		#if defined (_MSC_BUILD)
		handle_ = ::LoadLibraryA(dllPath_.c_str());
		#else
		handle_ = dlopen(dllPath_.c_str(), RTLD_LAZY);
		#endif
		if(!handle_)
		{
			std::ostringstream oss;
			oss<<"Tried to load Light GBM from "<<dllPath_<<" and failed eorr ;"
			#if defined (_MSC_BUILD)
			oss<<((Quantity)GetLastError());
			#else
			oss<<dlerror();
			#endif
			throw std::runtime_error(oss.str());

		}
		#if defined (_MSC_BUILD)
		LGBM_GetLastError_ = (LGBM_GetLastError_t)GetProcAddress(handle_, "LGBM_GetLastError");
		if(!LGBM_GetLastError_)
		{
			std::ostringstream oss;
			oss<<"Tried to load light GBM from "<<dllPath_<<" and failed to find symbol LGBM_GetLastError";
			throw std::runtime_error(oss.str());

		}
		#else
		LGBM_GetLastError_ = (LGBM_GetLastError_t)dlsym(handle_, "LGBM_GetLastError");
		if(!LGBM_GetLastError_)
		{
			std::ostringstream oss;
			oss<<"Tried to load light GBM from "<<dllPath_<<" and failed to find symbol LGBM_GetLastError";
			throw std::runtime_error(oss.str());

		}


		#endif
		if (verbose_)
		{
			std::cerr<<"Trying to load LightGBM from "<<dllPath_<< " got "<<function_(LGBM_GetLastError_)() <<std::endl;
		}

		if(!boosterModelPath_.empty())
		{
			#if defined (_MSC_BUILD)
			typedef int(__cdecl * LGBM_BoosterCreateFromModelfile_t)(const char*, int *, BoolsterHandle*);
			auto LGBM_BoosterCreateFromModelfile = (LGBM_BoosterCreateFromModelfile_t)::GetProcAddress(handle_, "LGBM_BoosterCreateFromModelfile");
			if (!LGBM_BoosterCreateFromModelfile)
			{
				std::ostringstream oss;
				oss<<"Tried to load light gbm from "<<dllPath_<<" and fail to find sysmte LGBM_BoosterCreateFromModelfile";
				throw std::runtime_error(oss.str());
			}
			#else
			typedef int( * LGBM_BoosterCreateFromModelfile_t)(const char*, int *, BoolsterHandle*);
			auto LGBM_BoosterCreateFromModelfile = (LGBM_BoosterCreateFromModelfile_t)dlsym(handle_, "LGBM_BoosterCreateFromModelfile");
			if (!LGBM_BoosterCreateFromModelfile)
			{
				std::ostringstream oss;
				oss<<"Tried to load light gbm from "<<dllPath_<<" and fail to find sysmte LGBM_BoosterCreateFromModelfile";
				throw std::runtime_error(oss.str());
			}

			#endif
			{
				int iterations = -1;
				callAndCheckError_(function_(LGBM_BoosterCreateFromModelfile)(boosterModelPath_.c_str(),&iterations, &boosterHandle_));
				if (verbose_)
				{
					std::cerr<<"Model load from "<<boosterModelPath_<<" after "<<iterations<< " iterations "<<std::endl;
				}
			}


			#if defined (_MSC_BUILD)

			LGBM_BoosterPredictForMatSingleRow_ = (LGBM_BoosterPredictForMatSingleRow_t)::GetProcAddress(handle_, "LGBM_BoosterPredictForMatSingleRow");
			if (!LGBM_BoosterPredictForMatSingleRow_)
			{
				std::ostringstream oss;
				oss<<"Tried to load light gbm from "<<dllPath_<<" and fail to find sysmte LGBM_BoosterPredictForMatSingleRow_";
				throw std::runtime_error(oss.str());
			}

			LGBM_BoosterPredictForMat_ = (LGBM_BoosterPredictForMat_t)::GetProcAddress(handle_, "LGBM_BoosterPredictForMat");
			if (!LGBM_BoosterPredictForMat_)
			{
				std::ostringstream oss;
				oss<<"Tried to load light gbm from "<<dllPath_<<" and fail to find sysmte LGBM_BoosterPredictForMat_";
				throw std::runtime_error(oss.str());
			}

			LGBM_BoosterInitPredictFor1Row_ = (LGBM_BoosterInitPredictFor1Row_t)::GetProcAddress(handle_, "LGBM_BoosterInitPredictFor1Row");
			if (!LGBM_BoosterInitPredictFor1Row_)
			{
				std::ostringstream oss;
				oss<<"Tried to load light gbm from "<<dllPath_<<" and fail to find sysmte LGBM_BoosterInitPredictFor1Row_";
				throw std::runtime_error(oss.str());
			}


			LGBM_BoosterPredictFor1Row_ = (LGBM_BoosterPredictFor1Row_t)::GetProcAddress(handle_, "LGBM_BoosterPredictFor1Row");
			if (!LGBM_BoosterPredictFor1Row_)
			{
				std::ostringstream oss;
				oss<<"Tried to load light gbm from "<<dllPath_<<" and fail to find sysmte LGBM_BoosterPredictFor1Row_";
				throw std::runtime_error(oss.str());
			}

			LGBM_BoosterFree_=(LGBM_BoosterFree_t)::GetProcAddress(handle_, "LGBM_BoosterFree");


			#else
			
			LGBM_BoosterPredictForMatSingleRow_ = *(LGBM_BoosterPredictForMatSingleRow_t)dlsym(handle_, "LGBM_BoosterPredictForMatSingleRow");
			if (!LGBM_BoosterPredictForMatSingleRow_)
			{
				std::ostringstream oss;
				oss<<"Tried to load light gbm from "<<dllPath_<<" and fail to find sysmte LGBM_BoosterPredictForMatSingleRow_";
				throw std::runtime_error(oss.str());
			}

			LGBM_BoosterPredictForMat_ = *(LGBM_BoosterPredictForMat_t)dlsym(handle_, "LGBM_BoosterPredictForMat");
			if (!LGBM_BoosterPredictForMat_)
			{
				std::ostringstream oss;
				oss<<"Tried to load light gbm from "<<dllPath_<<" and fail to find sysmte LGBM_BoosterPredictForMat_";
				throw std::runtime_error(oss.str());
			}

			LGBM_BoosterInitPredictFor1Row_ = *(LGBM_BoosterInitPredictFor1Row_t)dlsym(handle_, "LGBM_BoosterInitPredictFor1Row");
			if (!LGBM_BoosterInitPredictFor1Row_)
			{
				std::ostringstream oss;
				oss<<"Tried to load light gbm from "<<dllPath_<<" and fail to find sysmte LGBM_BoosterInitPredictFor1Row_";
				throw std::runtime_error(oss.str());
			}


			LGBM_BoosterPredictFor1Row_ = *(LGBM_BoosterPredictFor1Row_t)dlsym(handle_, "LGBM_BoosterPredictFor1Row");
			if (!LGBM_BoosterPredictFor1Row_)
			{
				std::ostringstream oss;
				oss<<"Tried to load light gbm from "<<dllPath_<<" and fail to find sysmte LGBM_BoosterPredictFor1Row_";
				throw std::runtime_error(oss.str());
			}

			LGBM_BoosterFree_=*(LGBM_BoosterFree_t)dlsym(handle_, "LGBM_BoosterFree");

			#endif

		}

	}
}

double LightGBMPlugin::predict (const std::vector<double>&features)const
{
	#ifdef _DEBUG
	if(!LGBM_BoosterPredictForMatSingleRow_)
	{
		throw std::runtime_error("Function LGBM_BoosterPredictForMatSingleRow_ is not properly loaded");
	}
	#endif
	std::int64_t outLen =0;
	double result = 0.0;
	callAndCheckError_(function(LGBM_BoosterPredictForMatSingleRow_)(boosterHandle_, features.data(),1, static_cast<std::int32_t>(features.size()), &result), "predict1");
	return result;
}

void LightGBMPlugin::initPredict1()

{
	#ifdef _DEBUG
	if(!LGBM_BoosterInitPredictFor1Row_)
	{
		throw std::runtime_error("Function LGBM_BoosterInitPredictFor1Row_ is not properly loaded");
	}
	#endif
	std::int64_t outLen =0;
	
	return callAndCheckError_(function(LGBM_BoosterInitPredictFor1Row_)(boosterHandle_, -1),"init predict1");
	
}

double LightGBMPlugin::predict1(const std::vector<double >&features) const
{	
	#ifdef _DEBUG
	if(!LGBM_BoosterPredictFor1Row_)
	{
		throw std::runtime_error("Function LGBM_BoosterPredictFor1Row_ is not properly loaded");
	}
	#endif
	std::int64_t outLen =0;
	double result = 0.0;
	callAndCheckError_(function(LGBM_BoosterPredictFor1Row_)(boosterHandle_, features.data(), static_cast<std::int32_t> (features.size()), &result),"predict1");
	return result;

}


void LightGBMPlugin::predict(const std::vector<double>& features, int rows, into cols, std::vector<double>& results) const
{
	#ifdef _DEBUG
	if(!LGBM_BoosterPredictForMat_)
	{
		throw std::runtime_error("Function LGBM_BoosterPredictForMat is not properly loaded");
	}
	#endif
	std::int64_t outLen =0;
	
	callAndCheckError_(function(LGBM_BoosterPredictForMat)(boosterHandle_, features.data(),1, rows, cols,1,0,-1, "", &outLen, results.data()), "predict many");
	
}