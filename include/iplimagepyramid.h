#ifndef IPLIMAGEPYRAMID_H_
#define IPLIMAGEPYRAMID_H_

// STL
#include <vector>

//my stuff
#include "iplimagewrapper.h"


class IplImagePyramid {
protected:
	std::vector<IplImageWrapper> _imagePyramid;
	std::vector<double> _scaleFactors;
	std::vector<double> _scaleFactorsInv;
	std::vector<double> _xScaleFactors;
	std::vector<double> _xScaleFactorsInv;
	std::vector<double> _yScaleFactors;
	std::vector<double> _yScaleFactorsInv;
	double _scaleFactor;
	double _epsilon;

protected:
//	IplImagePyramid(std::vector<IplImageWrapper> imagePyramid, std::vector<double> correctScaleFactors);

public:
	IplImagePyramid();

	IplImagePyramid(const IplImagePyramid& pyramid);

	/**
	 * NOTE: the image is referenced on the lowest pyramid level!
	 */
	IplImagePyramid(IplImageWrapper image, double scaleFactor);

	/**
	 * Build an empty pyramid (pixel values are set to zero).
	 */
	IplImagePyramid(CvSize initSize, int depth, int nChannels, double scaleFactor);

	~IplImagePyramid();

	IplImagePyramid& operator=(const IplImagePyramid& pyramid);

	operator const bool() const;

	operator bool();

	std::size_t numOfLevels() const;

	double getScaleFactor() const;

	/**
	 * round == 0  =>  take the closest level (i.e., rounding)
	 * round < 0   =>  take the next level with a smaller factor (i.e., flooring)
	 * round > 0   =>  take the next level with a bigger factor (i.e., ceiling)
	 */
	std::size_t getIndex(double scaleFactor, int round = 0) const;

	double getScaleFactor(std::size_t index) const;

	double getScaleFactorInv(std::size_t index) const;

	double getXScaleFactor(std::size_t index) const;

	double getXScaleFactorInv(std::size_t index) const;

	double getYScaleFactor(std::size_t index) const;

	double getYScaleFactorInv(std::size_t index) const;

	IplImageWrapper& getImage(std::size_t index);

	const IplImageWrapper& getImage(std::size_t index) const;

	/**
	 * @param round  see getIndex()
	 */
	IplImageWrapper getImage(double scaleFactor, int round = 0);
	const IplImageWrapper& getImage(double scaleFactor, int round = 0) const;

	/**
	 * rebuilds the pyramid (re-using the already allocated space) with the given 
	 * image
	 * NOTE: this image needs to have the exact sames size as the initial scale
	 */
	void rebuild(IplImageWrapper image);

private:
	void init(IplImageWrapper image, double scaleFactor);

	/**
	 * Build an empty pyramid (pixel values are set to zero).
	 */
	void init(CvSize initSize, int depth, int nChannels, double scaleFactor);

};


inline IplImagePyramid::IplImagePyramid()
	: _imagePyramid(), _scaleFactors(), _scaleFactorsInv(), _xScaleFactors(), _xScaleFactorsInv(),
	_yScaleFactors(), _yScaleFactorsInv(), _scaleFactor(0), _epsilon(0)
{ }

inline IplImagePyramid::IplImagePyramid(const IplImagePyramid& pyramid)
	: _imagePyramid(pyramid._imagePyramid.begin(), pyramid._imagePyramid.end()),
	_scaleFactors(pyramid._scaleFactors.begin(), pyramid._scaleFactors.end()),
	_scaleFactorsInv(pyramid._scaleFactorsInv.begin(), pyramid._scaleFactorsInv.end()),
	_xScaleFactors(pyramid._xScaleFactors.begin(), pyramid._xScaleFactors.end()),
	_xScaleFactorsInv(pyramid._xScaleFactorsInv.begin(), pyramid._xScaleFactorsInv.end()),
	_yScaleFactors(pyramid._yScaleFactors.begin(), pyramid._yScaleFactors.end()),
	_yScaleFactorsInv(pyramid._yScaleFactorsInv.begin(), pyramid._yScaleFactorsInv.end()),
	_scaleFactor(pyramid._scaleFactor), _epsilon(pyramid._epsilon)
{ }

inline IplImagePyramid::IplImagePyramid(IplImageWrapper image, double scaleFactor)
{
	assert(image);
	init(image, scaleFactor);
}

inline IplImagePyramid::IplImagePyramid(CvSize initSize, int depth, int nChannels, double scaleFactor)
{
	init(initSize, depth, nChannels, scaleFactor);
}

inline IplImagePyramid::~IplImagePyramid()
{
	// everything is destroyed automatically :)
}

inline IplImagePyramid& IplImagePyramid::operator=(const IplImagePyramid& pyramid)
{
	_imagePyramid.clear();
	_imagePyramid = pyramid._imagePyramid;
	_scaleFactor = pyramid._scaleFactor;
	_epsilon = pyramid._epsilon;
	_scaleFactors = pyramid._scaleFactors;
	_scaleFactorsInv = pyramid._scaleFactorsInv;
	_xScaleFactors = pyramid._xScaleFactors;
	_xScaleFactorsInv = pyramid._xScaleFactorsInv;
	_yScaleFactors = pyramid._yScaleFactors;
	_yScaleFactorsInv = pyramid._yScaleFactorsInv;
	return *this;
}

inline IplImagePyramid::operator const bool() const
{
	return _imagePyramid.size() > 0;
}

inline IplImagePyramid::operator bool()
{
	return _imagePyramid.size() > 0;
}

inline std::size_t IplImagePyramid::numOfLevels() const
{
	return _imagePyramid.size();
}

inline double IplImagePyramid::getScaleFactor() const
{
	return _scaleFactor;
}

inline double IplImagePyramid::getScaleFactor(std::size_t index) const
{
	assert(index < _scaleFactors.size());
	return _scaleFactors[index];
}

inline double IplImagePyramid::getScaleFactorInv(std::size_t index) const
{
	assert(index < _scaleFactors.size());
	return _scaleFactorsInv[index];
}

inline double IplImagePyramid::getXScaleFactor(std::size_t index) const
{
	assert(index < _scaleFactors.size());
	return _xScaleFactors[index];
}

inline double IplImagePyramid::getXScaleFactorInv(std::size_t index) const
{
	assert(index < _scaleFactors.size());
	return _xScaleFactorsInv[index];
}

inline double IplImagePyramid::getYScaleFactor(std::size_t index) const
{
	assert(index < _scaleFactors.size());
	return _yScaleFactors[index];
}

inline double IplImagePyramid::getYScaleFactorInv(std::size_t index) const
{
	assert(index < _scaleFactors.size());
	return _yScaleFactorsInv[index];
}

inline IplImageWrapper& IplImagePyramid::getImage(std::size_t index)
{
	assert(index < _imagePyramid.size());
	return _imagePyramid[index];
}

inline const IplImageWrapper& IplImagePyramid::getImage(std::size_t index) const
{
	assert(index < _imagePyramid.size());
	return _imagePyramid[index];
}

inline IplImageWrapper IplImagePyramid::getImage(double scaleFactor, int round)
{
	return getImage(getIndex(scaleFactor, round));
}

inline const IplImageWrapper& IplImagePyramid::getImage(double scaleFactor, int round) const
{
	return getImage(getIndex(scaleFactor, round));
}


#endif //IPLIMAGEPYRAMID_H_