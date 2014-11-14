#ifndef SIZE_HPP_
#define SIZE_HPP_

#include "size.h"
#include <cassert>
#include <stdexcept>

template<typename T>
inline bool Size<T>::isEmpty() const
{ return !isValid() || wd == T(0) || ht == T(0); }

template<typename T>
inline bool Size<T>::isValid() const
{ return wd >= T(0) && ht >= T(0); }

template<typename T>
inline T Size<T>::getWidth() const
{ return wd; }

template<typename T>
inline T Size<T>::getHeight() const
{ return ht; }

template<typename T>
inline void Size<T>::setWidth(T w)
{ wd = w; }

template<typename T>
inline void Size<T>::setHeight(T h)
{ ht = h; }

template<typename T>
inline T Size<T>::getArea() const
{
	if (wd <= T(0) || ht <= T(0))
		return T(0);
	return wd * ht;
}

template<typename T>
inline Size<T> &Size<T>::operator+=(const Size<T> &s)
{ wd += s.wd; ht += s.ht; return *this; }

template<typename T>
inline Size<T> &Size<T>::operator-=(const Size<T> &s)
{ wd -= s.wd; ht -= s.ht; return *this; }



template<typename T>
template<typename T2>
inline Size<T> &Size<T>::operator*=(T2 c)
{ wd *= c; ht *= c; return *this; }

template<typename T>
template<typename T2>
inline Size<T> &Size<T>::operator/=(T2 c)
{
    assert(c != T(0));
    wd = wd/c; ht = ht/c;
    return *this;
}

template<>
template<>
inline Size<int> &Size<int>::operator*=(float c)
{ wd = static_cast<int>(NumericFunctions::roundf(c * wd)); ht = static_cast<int>(NumericFunctions::roundf(c * ht)); return *this; }

template<>
template<>
inline Size<int> &Size<int>::operator/=(float c)
{
    assert(c != float(0));
    wd = static_cast<int>(NumericFunctions::roundf(wd / c)); ht = static_cast<int>(NumericFunctions::roundf(ht / c));
    return *this;
}

template<>
template<>
inline Size<long> &Size<long>::operator*=(float c)
{ wd = static_cast<long>(NumericFunctions::roundf(c * wd)); ht = static_cast<long>(NumericFunctions::roundf(c * ht)); return *this; }

template<>
template<>
inline Size<long> &Size<long>::operator/=(float c)
{
    assert(c != float(0));
    wd = static_cast<long>(NumericFunctions::roundf(wd / c)); ht = static_cast<long>(NumericFunctions::roundf(ht / c));
    return *this;
}

template<>
template<>
inline Size<int> &Size<int>::operator*=(double c)
{ wd = static_cast<int>(NumericFunctions::round(c * wd)); ht = static_cast<int>(NumericFunctions::round(c * ht)); return *this; }

template<>
template<>
inline Size<int> &Size<int>::operator/=(double c)
{
    assert(c != double(0));
    wd = static_cast<int>(NumericFunctions::round(wd / c)); ht = static_cast<int>(NumericFunctions::round(ht / c));
    return *this;
}

template<>
template<>
inline Size<long> &Size<long>::operator*=(double c)
{ wd = static_cast<long>(NumericFunctions::round(c * wd)); ht = static_cast<long>(NumericFunctions::round(c * ht)); return *this; }

template<>
template<>
inline Size<long> &Size<long>::operator/=(double c)
{
    assert(c != double(0));
    wd = static_cast<long>(NumericFunctions::round(wd / c)); ht = static_cast<long>(NumericFunctions::round(ht / c));
    return *this;
}



template<typename T>
inline bool operator==(const Size<T> &s1, const Size<T> &s2)
{ return s1.getWidth() == s2.getWidth() && s1.getHeight() == s2.getHeight(); }

template<typename T>
inline bool operator!=(const Size<T> &s1, const Size<T> &s2)
{ return s1.getWidth() != s2.getWidth() || s1.getHeight() != s2.getHeight(); }

template<>
inline bool operator==(const Size<float> &s1, const Size<float> &s2)
{ return NumericFunctions::fuzzyEqual(s1.getWidth(), s2.getWidth()) && NumericFunctions::fuzzyEqual(s1.getHeight(), s2.getHeight()); }

template<>
inline bool operator!=(const Size<float> &s1, const Size<float> &s2)
{ return !NumericFunctions::fuzzyEqual(s1.getWidth(), s2.getWidth()) || !NumericFunctions::fuzzyEqual(s1.getHeight(), s2.getHeight()); }

template<>
inline bool operator==(const Size<double> &s1, const Size<double> &s2)
{ return NumericFunctions::fuzzyEqual(s1.getWidth(), s2.getWidth()) && NumericFunctions::fuzzyEqual(s1.getHeight(), s2.getHeight()); }

template<>
inline bool operator!=(const Size<double> &s1, const Size<double> &s2)
{ return !NumericFunctions::fuzzyEqual(s1.getWidth(), s2.getWidth()) || !NumericFunctions::fuzzyEqual(s1.getHeight(), s2.getHeight()); }




template<typename T>
inline const Size<T> operator+(const Size<T> & s1, const Size<T> & s2)
{ return Size<T>(s1.getWidth()+s2.getWidth(), s1.getHeight()+s2.getHeight()); }

template<typename T>
inline const Size<T> operator-(const Size<T> &s1, const Size<T> &s2)
{ return Size<T>(s1.getWidth()-s2.getWidth(), s1.getHeight()-s2.getHeight()); }

template<typename T, typename T2>
inline const Size<T> operator*(const Size<T> &s, T2 c)
{
	Size<T> size = s;
	size *= c;
	return size;
}

template<typename T, typename T2>
inline const Size<T> operator*(T2 c, const Size<T> &s)
{
	Size<T> size = s;
	size *= c;
	return size;
}

template<typename T>
inline const Size<T> operator/(const Size<T> &s, T c)
{
	Size<T> size = s;
	size /= c;
	return size;
}

template<typename T>
inline void Size<T>::transpose()
{
    T tmp = wd;
    wd = ht;
    ht = tmp;
}

template<typename T>
template<typename T2>
inline Size<T>::operator Size<T2>() const
{
	return Size<T2>(static_cast<T2>(getWidth()), static_cast<T2>(getHeight()));
}

template<>
template<>
inline Size<float>::operator Size<int>() const
{
	return Size<int>(static_cast<int>(NumericFunctions::roundf(getWidth())), static_cast<int>(NumericFunctions::roundf(getHeight())));
}

template<>
template<>
inline Size<float>::operator Size<long>() const
{
	return Size<long>(static_cast<long>(NumericFunctions::roundf(getWidth())), static_cast<long>(NumericFunctions::roundf(getHeight())));
}


template<>
template<>
inline Size<double>::operator Size<int>() const
{
	return Size<int>(static_cast<int>(NumericFunctions::round(getWidth())), static_cast<int>(NumericFunctions::round(getHeight())));
}

template<>
template<>
inline Size<double>::operator Size<long>() const
{
	return Size<long>(static_cast<long>(NumericFunctions::round(getWidth())), static_cast<long>(NumericFunctions::round(getHeight())));
}


template<typename T>
inline std::ostream& operator<<(std::ostream& o, const Size<T>& s)
{
	o << s.getWidth() << " x " << s.getHeight();
	return o;
}

template<typename T>
std::istream& operator>>(std::istream& i, Size<T>& s)
{
	T w, h;
	i.setf(std::ios_base::skipws);
	i >> w;
	char c = i.peek();
	while (c == '\t' || c == ' ' || c == ',' || c == 'x') {
		i.get(c);
		c = i.peek();
		if (!i.good())
			throw std::runtime_error("Size: unexpected error/end reading the stream");
	}
	i >> h;
	s = Size<T>(w, h);
	return i;
}


#endif //SIZE_HPP_