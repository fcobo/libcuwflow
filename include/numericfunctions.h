#ifndef NUMERIC_FUNCTIONS_H
#define NUMERIC_FUNCTIONS_H

/**
* \class NumericFunctions
*
* \brief This class contains a group of mathematical functions which are needed
* in w-flow dense trajectories algorithm
*/
class NumericFunctions{

public:

	static double round(double val);
	static float roundf(float x);
	static bool fuzzyEqual(const double& x1, const double& x2);
	static bool fuzzyLower(const double& x1, const double& x2);
	static bool fuzzyLowerEqual(const double& x1, const double& x2);
	static bool fuzzyGreater(const double& x1, const double& x2);
	static bool fuzzyGreaterEqual(const double& x1, const double& x2);
	static bool fuzzyEqual(const float& x1, const float& x2);
	static bool fuzzyLower(const float& x1, const float& x2);
	static bool fuzzyLowerEqual(const float& x1, const float& x2);	
	static bool fuzzyGreater(const float& x1, const float& x2);
	static bool fuzzyGreaterEqual(const float& x1, const float& x2);
	static double interpolateXMaximum(double x1, double y1, double x2, double y2, double x3, double y3);
};

inline double NumericFunctions::round(double val)
{    
    return floor(val + 0.5);
}

inline float NumericFunctions::roundf(float x)
{
   return x >= 0.0f ? floorf(x + 0.5f) : ceilf(x - 0.5f);
}
	
inline bool NumericFunctions::fuzzyEqual(const double& x1, const double& x2)
{
    return fabs(x1 - x2) <= 1e-10 * fabs(x1 + x2);
}

inline bool NumericFunctions::fuzzyLower(const double& x1, const double& x2)
{
    return x1 < x2 - 1e-10 * fabs(x1 + x2);
}

inline bool NumericFunctions::fuzzyLowerEqual(const double& x1, const double& x2)
{
    return x1 <= x2 + 1e-10 * fabs(x1 + x2);
}

inline bool NumericFunctions::fuzzyGreater(const double& x1, const double& x2)
{
    return x1 > x2 + 1e-10 * fabs(x1 + x2);
}

inline bool NumericFunctions::fuzzyGreaterEqual(const double& x1, const double& x2)
{
    return x1 >= x2 - 1e-10 * fabs(x1 + x2);
}

inline bool NumericFunctions::fuzzyEqual(const float& x1, const float& x2)
{
    return fabs(x1 - x2) <= 1e-5 * fabs(x1 + x2);
}

inline bool NumericFunctions::fuzzyLower(const float& x1, const float& x2)
{
    return x1 < x2 - 1e-5 * fabs(x1 + x2);
}

inline bool NumericFunctions::fuzzyLowerEqual(const float& x1, const float& x2)
{
    return x1 < x2 + 1e-5 * fabs(x1 + x2);
}

inline bool NumericFunctions::fuzzyGreater(const float& x1, const float& x2)
{
    return x1 > x2 + 1e-5 * fabs(x1 + x2);
}

inline bool NumericFunctions::fuzzyGreaterEqual(const float& x1, const float& x2)
{
    return x1 > x2 - 1e-5 * fabs(x1 + x2);
}

inline double NumericFunctions::interpolateXMaximum(double x1, double y1, double x2, double y2, double x3, double y3)
{
	double x1Sqr = pow(x1, 2);
	double x2Sqr = pow(x2, 2);
	double x3Sqr = pow(x3, 2);
	return 0.5 * (y2 * (x3Sqr - x1Sqr) + y3 * (x1Sqr - x2Sqr) + y1 * (x2Sqr - x3Sqr)) /
			(y2 * (x3 - x1) + y3 * (x1 - x2) + y1 * (x2 - x3));
}

#endif //NUMERIC_FUNCTIONS_H