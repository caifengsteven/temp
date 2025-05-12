#pragma once

#include <set>
#include <memory>
#include <ostream>
#include <sstream>

#if defined(_MSC_BUILD)
#define LIKELY(a) a
#define UNLIKELY(a) a
#elif defined (__GNUC__)
#define LIKELY(a) __builtin_expect((a),1)
#define UNLIKELY(a) __builtin_expect((a),0)
#endif

template <class T> <T>

inline std::ostream& operator <<(std::ostream& os, const std::set<T>& container)
{
	os<<'[';
	if(!container.empty())
	{
		typename std::set<T>::const_iterator it = container.begin();
		os<<*it;
		for (++it; it!= container.end(); ++it)
		{
			os<<", "<<*it;
		}

	}
	os<<']';
	return os;
}

inline void parse(const std::string& serialisedStrList, std::set<std::string>& strList, char delimiter)
{
	strList.clear();
	std::stringstream tmp(serialisedStrList);
	for (std::string token; std::getline(tmp, token, delimiter);)
	{
		strList.emplace(token);
	}
}

template <class T>
class Singleton
{
public:
	inline static T& instance()
	{
		static std::unique_ptr<T> instance_;
		if(!instance_)
		{
			instance_.reset(new T);
		}
		return * instance_;
	}
};