#pragma once
#include "Configurable.hxx"
#include <string>
#include <memory>
#include <vector>
#include <cstdint>

#if defined(_MSC_BUILD)
struct HINSTANCE__;
#endif 

class LightGBMPlugin:protected Configurable
{
	const bool verbose_;
	std::string dllPath_;
	std::string boolsterModelPath_;
#if defined (_MSC_BUILD)
	typedef HINSTANCE__* Handle;
#else
	typedef void* Handle;

	Handle handle_ = nullptr;
	typedef void * BoosterHandle;
	BoosterHandle boosterHandle_ = nullptr;

#if defined (_MSC_BUILD)
	typedef const char* (__cdecl * LGBM_GetLastError_t) (void);
	typedef int(__cdecl * LGBM_BoosterPredictForMatSingleRow_t)(BoosterHandle, const void *, int, std::int32_t, int, int, int, const char *, std::int64_t*, double *);
	typedef int(__cdecl * LGBM_BoosterPredictForMat_t)(BoosterHandle, const void *, int, std::int32_t, std::int32_t, int, int, int, const char*, std::int64_t*,double *);
	typedef int(__cdecl * LGBM_BoosterInitPredictFor1Row_t)(BoosterHandle, int);
	typedef int(__cdecl * LGBM_BossterPredictFor1Row_t)(BoosterHandle, const double * , int, double * );
	typedef int(__cdecl* LGBM_BoosterFree_t)(BoosterHandle);

#else

	typedef const char* (* LGBM_GetLastError_t) ();
	typedef int(* LGBM_BoosterPredictForMatSingleRow_t)(BoosterHandle, const void *, int, std::int32_t, int, int, int, const char *, std::int64_t*, double *);
	typedef int(* LGBM_BoosterPredictForMat_t)(BoosterHandle, const void *, int, std::int32_t, std::int32_t, int, int, int, const char*, std::int64_t*,double *);
	typedef int(* LGBM_BoosterInitPredictFor1Row_t)(BoosterHandle, int);
	typedef int(* LGBM_BossterPredictFor1Row_t)(BoosterHandle, const double * , int, double * );
	typedef int(* LGBM_BoosterFree_t)(BoosterHandle);
#endif

	LGBM_GetLastError_t LGBM_GetLastError_ = nullptr;
	LGBM_BoosterPredictForMatSingleRow_t LGBM_BoosterPredictForMatSingleRow_ = nullptr;
	LGBM_BoosterPredictForMat_t LGBM_BoosterPredictForMat_ =nullptr;
	LGBM_BoosterInitPredictFor1Row_t LGBM_BoosterInitPredictFor1Row_ = nullptr;
	LGBM_BoosterPredictFor1Row_t LGBM_BoosterPredictFor1Row_ = nullptr;
	LGBM_BoosterFree_t LGBM_BoosterFree_ = nullptr;

public: 
	using Configurable::readParameterValues;
	LightGBMPlugin(bool verbose);
	~LightGBMPlugin();

	virtual void onInit() override;
	bool checkPredictIsPossible() cosnt {return LGBM_BoosterPredictForMatSingleRow_ && LGBM_BoosterPredictForMat_;}
	void predict (const std::vector<double>& features) const;
	void predict( const std::vector <double>& features, int rows, int cols, std::vector<double>& results)const;
	bool checkPredict1IsPossible() const{return LGBM_BoosterInitPredictFor1Row_ && LGBM_BoostPredictFor1Row_;}
	void initPredict1();
	double predict1(const std::vector<double>& features) const;

private:
	template<typename FUNCTION>
	static auto& function_(FUNCTION& function);
	void callAndCheckError_(int errCode, const char* context) const;


};
typedef std::unique_ptr<LightGBMPlugin> PLightGBMPlugin;