#ifndef IPLIMAGEWRAPPER_H_
#define IPLIMAGEWRAPPER_H_

#include <opencv/cxcore.h>
#include <opencv/highgui.h>
#include <boost/optional.hpp>

#include <cstdlib>
#include <string>

#include "box.h"
#include "numericfunctions.h"

/**
 * Wrapper in order to use IplImages more easily in STL containers .. for that we
 * need a destructor that releases an image correctly using cvReleaseImage().
 * We also want to have smart pointers with a reference count in order to copy
 * containers.
 *
 * NOTE: Do not use generic algorithms with this class! (See discussions on auto_ptr,
 *       similar thoughts/problems apply to this smart pointer!)
 */

	
class IplImageWrapper {

protected:
	IplImage* _img;
	std::size_t* _nRefs;
	boost::optional<Box<int> > _mask;

protected:
	void decrementAndFree();

public:
	IplImageWrapper(IplImage *newImg = NULL, bool isOwner = true);

	IplImageWrapper(CvSize size, int depth, int channels);

	IplImageWrapper(std::string fileName);

	IplImageWrapper(const IplImageWrapper& newImg);

	~IplImageWrapper();

	IplImageWrapper clone() const;

	IplImageWrapper& operator=(const IplImageWrapper& img);

	operator IplImage*();

	operator const IplImage*() const;

	operator const bool() const;

	operator bool();

	IplImage* operator->();

	const IplImage* operator->() const;

	IplImage* getReference();

	const IplImage* getReference() const;

	std::size_t numOfReferences() const;

	bool hasMask() const;

	Box<int> getMask() const;

	void setMask(const Box<int>& mask);

	void clearMask();
};


inline void IplImageWrapper::decrementAndFree() {

	if (_nRefs) {
		--(*_nRefs);
		if (*_nRefs == 0) {
			if (_img)
				cvReleaseImage(&_img);
			delete _nRefs;
		}
	}
}

inline IplImageWrapper::IplImageWrapper(IplImage *newImg, bool isOwner)
	: _img(newImg), _nRefs(isOwner ? new std::size_t(1) : 0), _mask()
{
}

inline IplImageWrapper::IplImageWrapper(CvSize size, int depth, int channels)
	: _img(cvCreateImage(size, depth, channels)), _nRefs(new std::size_t(1)), _mask()
{
}

inline IplImageWrapper::IplImageWrapper(std::string fileName)
	: _img(cvLoadImage(fileName.c_str())), _nRefs(new std::size_t(1)), _mask()
{
}

inline IplImageWrapper::IplImageWrapper(const IplImageWrapper& newImg)
	: _img(newImg._img), _nRefs(newImg._nRefs), _mask(newImg._mask)
{
	++(*_nRefs);
}

inline IplImageWrapper::~IplImageWrapper()
{
	decrementAndFree();
}

inline IplImageWrapper IplImageWrapper::clone() const
{
	IplImage *iplImg = cvCloneImage(this->getReference());
	IplImageWrapper clonedImg(iplImg);
	if (_mask)
		clonedImg.setMask(*_mask);
	return clonedImg;
}

inline IplImageWrapper::operator IplImage*()
{
	return _img;
}

inline IplImageWrapper::operator const IplImage*() const
{
	return _img;
}

inline IplImageWrapper::operator const bool() const
{
	return _img;
}

inline IplImageWrapper::operator bool()
{
	return _img;
}

inline IplImage* IplImageWrapper::getReference()
{
	return _img;
}

inline const IplImage* IplImageWrapper::getReference() const
{
	return _img;
}

inline std::size_t IplImageWrapper::numOfReferences() const
{
	return _nRefs ? 0 : (*_nRefs);
}

inline IplImage* IplImageWrapper::operator->()
{
	return _img;
}

inline const IplImage* IplImageWrapper::operator->() const
{
	return _img;
}

inline bool IplImageWrapper::hasMask() const
{
	return _mask;
}

inline Box<int> IplImageWrapper::getMask() const
{
	return *_mask;
}

inline void IplImageWrapper::setMask(const Box<int>& mask)
{
	_mask = boost::optional<Box<int> >(mask);
}

inline void IplImageWrapper::clearMask()
{
	_mask = boost::optional<Box<int> >();
}

#endif /*IPLIMAGEWRAPPER_H_*/