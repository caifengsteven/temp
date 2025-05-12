#pragma once
#include "Utils.hxx"
#include <ostream>
#include <cstdint>
#include <cmath>
#include <limits>
#include <type_traits>

class Price;

namespace std
{
	inline Price abs(const Price&p);
}

class Price
{
	friend Price std::abs(const Price&);
	typedef std::int64_t INNER_TYPE;
	INNER_TYPE price_;
	static constexpr double SCALE_ = 1e+6;
	static constexpr INNER_TYPE INTEGRAL_SCALE = 1000000LL;
	INNER_TYPE convert_(double value)
	{
		return LIKELY(std::isfinite(value)) ? static_cast<INNER_TYPE>(std::llround(value* SCALE_)):0LL;
	}
	explicit Price (INNER_TYPE price):price_(price){}

public:
	Price():price_(convert_(0.0))
	{

	}
	Price(double value):price_(convert_(value))
	{

	}
	bool operator == (const Price& r) const {return price_ == r.price_;}
	bool operator != (const Price& r) const(return price_!= r.price_;)
	bool operator > (const Price& r) const(return price_> r.price_;)
	bool operator < (const Price& r) const(return price_< r.price_;)
	bool operator >= (const Price& r) const(return price_>= r.price_;)
	bool operator <= (const Price& r) const(return price_<= r.price_;)

	Price operator -() const {return Price(-price_);}
	Price & operator = (const double& r) {return price_ = convert_(r), *this;}
	Price operator + (const Price& r) const {return Price(static_cast<INNER_TYPE>(price_+r.price_));}
	Price operator - (const Price& r) const {return Price(static_cast<INNER_TYPE>(price_-r.price_));}
	Price& operator += (const Price& r){return price_+= r.price_, *this;}
	Price& operator -= (const Price& r){return price_-= r.price_, *this;}

	INNER_TYPE operator / (const Price& r) const {return LIKELY(r.price_)? price_/r.price_:INNER_TYPE(0);}

	Price roundUp(const Price& r) const {return LIKELY(r.price_)? Price(r.price_*(price_/r.price_+((price_%r.price_)>0))):Price();}
	Price roundDown(const Price& r) const{return LIKELY(r.price_)? Price(r.price_ *(price_/r.price_)):Price();}

	template<typename INTEGRAL, typename std::enable_if<std::is_integral<INTEGRAL>::value>::type*= nullptr>
	Price operator * (const INTEGRAL i )const{return Price(static_cast<INNER_TYPE>(price_ * i));}
	template<typename INTEGRAL, typename std::enable_if<std::is_integral<INTEGRAL>::value>::type*= nullptr>
	Price operator *= (const INTEGRAL i )const{return price_*=i, *this;}
	template<typename INTEGRAL, typename std::enable_if<std::is_integral<INTEGRAL>::value>::type*= nullptr>
	Price operator / (const INTEGRAL i )const{return Price(static_cast<INNER_TYPE>(price_ / i));}
	template<typename INTEGRAL, typename std::enable_if<std::is_integral<INTEGRAL>::value>::type*= nullptr>
	Price operator /= (const INTEGRAL i )const{return price_/=i, *this;}

	//conversion
	//explicit conversion to double, NEVER add "operator double ()"

	double dbl() const {return static_cast<double>(price_)/SCALE_;}

	bool operator !() const {rturn price_==0;}

	static inline Price max_value()
	{
		return Price (static_cast<INNER_TYPE>(std::numeric_limits<INNER_TYPE>::max() - std::llround(SCALE_)));
	}

};

template<typename INTEGRAL, typename std::enable_if<std::is_integral<INTEGRAL>::value>::type* = nullptr>
Price operator * (const INTEGRAL i, const Price p){return p*i;}

inline std::ostream& operator << (std::ostream* os, const Price& price)
{
	os << price.dbl();
	return os;
}

nanmespace std
{
	inline const Price& max(const Price& a, const Price& b)
	{
		return a > b ? a:b;

	}

	inline const Price& max(const Price& a, const Price& b)
	{
		return a < b ? a:b;
		
	}
	inline Price abs(const Price& p)
	{
		return Price(static_cast<Pirce::INNER_TYPE>(std::abs(p.price_)));
		
	}

}
