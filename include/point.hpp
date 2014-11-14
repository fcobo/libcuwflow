#ifndef POINT_HPP_
#define POINT_HPP_

#include "point.h"
#include <cassert>


template<typename T>
inline bool Point<T>::isNull() const
{
    return xp == T(0) && yp == T(0);
}

template<typename T>
inline T Point<T>::getX() const
{
    return xp;
}

template<typename T>
inline T Point<T>::getY() const
{
    return yp;
}

template<typename T>
inline void Point<T>::setX(T xpos)
{
    xp = xpos;
}

template<typename T>
inline void Point<T>::setY(T ypos)
{
    yp = ypos;
}

template<typename T>
inline Point<T> &Point<T>::operator+=(const Point<T> &p)
{
    xp += p.xp;
    yp += p.yp;
    return *this;
}

template<typename T>
inline Point<T> &Point<T>::operator-=(const Point<T> &p)
{
    xp-=p.xp; yp-=p.yp; return *this;
}



template<typename T>
template<typename T2>
inline Point<T> &Point<T>::operator*=(T2 c)
{
    xp = static_cast<T>(xp * c);
    yp = static_cast<T>(yp * c);
    return *this;
}

template<>
template<>
inline Point<int> &Point<int>::operator*=(float c)
{
    xp = static_cast<int>(NumericFunctions::roundf(xp * c));
    yp = static_cast<int>(NumericFunctions::roundf(yp * c));
    return *this;
}

template<>
template<>
inline Point<long> &Point<long>::operator*=(float c)
{
    xp = static_cast<long>(NumericFunctions::roundf(xp * c));
    yp = static_cast<long>(NumericFunctions::roundf(yp * c));
    return *this;
}

template<>
template<>
inline Point<int> &Point<int>::operator*=(double c)
{
    xp = static_cast<int>(NumericFunctions::round(xp * c));
    yp = static_cast<int>(NumericFunctions::round(yp * c));
    return *this;
}

template<>
template<>
inline Point<long> &Point<long>::operator*=(double c)
{
    xp = static_cast<long>(NumericFunctions::round(xp * c));
    yp = static_cast<long>(NumericFunctions::round(yp * c));
    return *this;
}



template<typename T>
template<typename T2>
inline Point<T> &Point<T>::operator/=(T2 c)
{
    assert(!fuzzyEqual(c, 0));
    xp = static_cast<T>(xp / c);
    yp = static_cast<T>(yp / c);
    return *this;
}

template<>
template<>
inline Point<int> &Point<int>::operator/=(float c)
{
    assert(c != float(0));
    xp = static_cast<int>(NumericFunctions::round(xp / c));
    yp = static_cast<int>(NumericFunctions::round(yp / c));
    return *this;
}

template<>
template<>
inline Point<long> &Point<long>::operator/=(float c)
{
    assert(c != float(0));
    xp = static_cast<long>(NumericFunctions::round(xp / c));
    yp = static_cast<long>(NumericFunctions::round(yp / c));
    return *this;
}

template<>
template<>
inline Point<int> &Point<int>::operator/=(double c)
{
    assert(c != double(0));
    xp = static_cast<int>(NumericFunctions::round(xp / c));
    yp = static_cast<int>(NumericFunctions::round(yp / c));
    return *this;
}

template<>
template<>
inline Point<long> &Point<long>::operator/=(double c)
{
    assert(c != double(0));
    xp = static_cast<long>(NumericFunctions::round(xp / c));
    yp = static_cast<long>(NumericFunctions::round(yp / c));
    return *this;
}



template<typename T>
inline bool operator==(const Point<T> &p1, const Point<T> &p2)
{
    return p1.getX() == p2.getX() && p1.getY() == p2.getY();
}

template<>
inline bool operator==(const Point<float> &p1, const Point<float> &p2)
{
    return NumericFunctions::fuzzyEqual(p1.getX(), p2.getX()) && NumericFunctions::fuzzyEqual(p1.getY(), p2.getY());
}

template<>
inline bool operator==(const Point<double> &p1, const Point<double> &p2)
{
    return NumericFunctions::fuzzyEqual(p1.getX(), p2.getX()) && NumericFunctions::fuzzyEqual(p1.getY(), p2.getY());
}



template<typename T>
inline bool operator!=(const Point<T> &p1, const Point<T> &p2)
{
    return p1.getX() != p2.getX() || p1.getY() != p2.getY();
}

template<>
inline bool operator!=(const Point<float> &p1, const Point<float> &p2)
{
    return !NumericFunctions::fuzzyEqual(p1.getX(), p2.getX()) || !NumericFunctions::fuzzyEqual(p1.getY(), p2.getY());
}

template<>
inline bool operator!=(const Point<double> &p1, const Point<double> &p2)
{
    return !NumericFunctions::fuzzyEqual(p1.getX(), p2.getX()) || !NumericFunctions::fuzzyEqual(p1.getY(), p2.getY());
}



template<typename T>
inline const Point<T> operator+(const Point<T> &p1, const Point<T> &p2)
{
    return Point<T>(p1.getX()+p2.getX(), p1.getY()+p2.getY());
}

template<typename T>
inline const Point<T> operator-(const Point<T> &p1, const Point<T> &p2)
{
    return Point<T>(p1.getX()-p2.getX(), p1.getY()-p2.getY());
}

template<typename T, typename T2>
inline const Point<T> operator*(const Point<T> &p, T2 c)
{
    Point<T> point = p;
    point *= c;
    return point;
}

template<typename T, typename T2>
inline const Point<T> operator*(T2 c, const Point<T> &p)
{
    Point<T> point = p;
    point *= c;
    return point;
}

template<typename T>
inline const Point<T> operator-(const Point<T> &p)
{
    return Point<T>(-p.getX(), -p.getY());
}

template<typename T, typename T2>
inline const Point<T> operator/(const Point<T> &p, T2 c)
{
	Point<T> point = p;
	point /= c;
	return point;
}



template<typename T>
template<typename T2>
inline Point<T>::operator Point<T2>() const
{
	return Point<T2>(static_cast<T2>(getX()), static_cast<T2>(getY()));
}

template<>
template<>
inline Point<float>::operator Point<int>() const
{
	return Point<int>(static_cast<int>(NumericFunctions::roundf(getX())), static_cast<int>(NumericFunctions::roundf(getY())));
}

template<>
template<>
inline Point<float>::operator Point<long>() const
{
	return Point<long>(static_cast<long>(NumericFunctions::roundf(getX())), static_cast<long>(NumericFunctions::roundf(getY())));
}

template<>
template<>
inline Point<double>::operator Point<int>() const
{
	return Point<int>(static_cast<int>(NumericFunctions::round(getX())), static_cast<int>(NumericFunctions::round(getY())));
}

template<>
template<>
inline Point<double>::operator Point<long>() const
{
	return Point<long>(static_cast<long>(NumericFunctions::round(getX())), static_cast<long>(NumericFunctions::round(getY())));
}



template<typename T>
inline std::ostream& operator<<(std::ostream& o, const Point<T>& p)
{
	o << p.getX() << ", " << p.getY();
	return o;
}

template<typename T>
std::istream& operator>>(std::istream& i, Point<T>& p)
{
	T x, y;
	i.setf(std::ios_base::skipws);
	i >> x;
	i.ignore(256, ',');
	i >> y;
	p = Point<T>(x, y);
	return i;
}


#endif
