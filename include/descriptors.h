/**
* \file descriptors.h
* This header file contains the declaration as well as the implementation of the methods that are used to 
* calculate the movement descriptors.
*/

#ifndef CUWFLOW_DESCRIPTORS_H_
#define CUWFLOW_DESCRIPTORS_H_

#include "densetrack.h"


/**
* \class Descriptors
*
* \brief This class contains a group of methods related with the descriptors calculation in the cpu 
*/
class Descriptors{

public:

/**
* This method gets the rectangle for computing the descriptor 
* \param point The interest point position
* \param size  The size of the image
* \param desc_info Parameters about the descriptor
* \return A rectangule
*/
	static CvScalar getRect(const CvPoint2D32f point, const CvSize size, const DescInfo desc_info);

/**
* This method computes integral histograms for the whole image 
* \param x_comp x gradient component
* \param y_comp y gradient component
* \param desc_mat Output integral histograms
* \param desc_info Parameters about the descriptor
*/
	static void buildDescMat(const IplImage* x_comp,const IplImage* y_comp,DescMat* desc_mat,const DescInfo );


/**
* This method gets a descriptor from the integral histogram 
* \param desc_mat Input integral histogram
* \param rec Rectangle area for the descriptor
* \param desc_info Parameters about the descriptor
* \return An array with the descriptor
*/
	static std::vector<float> getDesc(const DescMat* desc_mat, CvScalar rect, DescInfo desc_info);

/**
* This method calculates the hog descriptor
* \param img Array with the input image
* \param desc_mat Array for the output integral histogram
* \param desc_info Parameters about the descriptor
*/
	static void hogComp(IplImage* img, DescMat* desc_mat, DescInfo desc_info);

/**
* This method calculates the hof descriptor
* \param flow Array with the optical flow
* \param desc_mat Array for the output integral histogram
* \param desc_info Parameters about the descriptor
*/
	static void hofComp(IplImage* flow, DescMat* desc_mat, DescInfo desc_info);

/**
* This method calculates the mbh descriptor
* \param flow Array with the optical flow
* \param desc_mat_x Array for the output integral histogram for X component
* \param desc_mat_y Array for the output integral histogram for Y component 
* \param desc_info Parameters about the descriptor
*/
	static void mbhComp(IplImage* flow, DescMat* desc_mat_x, DescMat* desc_mat_y, DescInfo desc_info);

/**
* This method calculates the DCS descriptor
* \param flow Array with the optical flow
* \param desc_mat_curldiv Array for the output integral histogram for curldiv
* \param desc_mat_shear_c Array for the output integral histogram for shearC
* \param desc_mat_shear_d Array for the output integral histogram for shearD
* \param curldiv_info Parameters about the curldiv descriptor
* \param shearCD_info Parameters about the shearCD descriptor
*/
	static void kinemComp(IplImage* flow, DescMat* desc_mat_curldiv, DescMat* desc_mat_shear_c, DescMat* desc_mat_shear_d, DescInfo curldiv_info, DescInfo shearCD_info);

/**
* This method tracks the interest points by median filtering in the optical field 
* \param flow The optical field
* \param points_in An array with the input interest point positions
* \param points_out  An array with the output interest point positions
* \param status An array for successfully tracked or not
*/
	static void opticalFlowTracker(IplImage* flow, std::vector<CvPoint2D32f>& points_in, std::vector<CvPoint2D32f>& points_out, std::vector<int>& status);

/**
* This method checks whether a trajectory is valid or not
* \param track Array with the trajectory
* \param mean_x Average of the component x of the trajectory
* \param mean_y Average of the component y of the trajectory
* \param var_x  Variance of the component x of the trajectory
* \param var_y Variance of the component y of the trajectory 
* \param length Length parameter
*/
	static int isValid(std::vector<CvPoint2D32f>& track, float& mean_x, float& mean_y, float& var_x, float& var_y, float& length);

/**
* This method detects new feature points in the whole image 
*/
	static void cvDenseSample(IplImage* grey, IplImage* eig, std::vector<CvPoint2D32f>& points,const double quality, const double min_distance);
	
/**
* This method detects new feature points in a image without overlapping to previous points
*/	
	static void cvDenseSample(IplImage* grey, IplImage* eig, std::vector<CvPoint2D32f>& points_in,std::vector<CvPoint2D32f>& points_out, const double quality, const double min_distance);
};


inline CvScalar Descriptors::getRect(const CvPoint2D32f point, const CvSize size, const DescInfo desc_info) 
{
	int x_min = desc_info.blockWidth/2;
	int y_min = desc_info.blockHeight/2;
	int x_max = size.width - desc_info.blockWidth;
	int y_max = size.height - desc_info.blockHeight;

	CvPoint2D32f point_temp;

	float temp = point.x - x_min;
	point_temp.x = std::min<float>(std::max<float>(temp, 0.), x_max);

	temp = point.y - y_min;
	point_temp.y = std::min<float>(std::max<float>(temp, 0.), y_max);

	//return the rectangle
	CvScalar rect;
	rect.val[0] = point_temp.x;
	rect.val[1] = point_temp.y;
	rect.val[2] = desc_info.blockWidth;
	rect.val[3] = desc_info.blockHeight;

	return rect;
}

inline void Descriptors::buildDescMat(const IplImage* xComp,const IplImage* yComp,DescMat* descMat,const DescInfo desc_info) 
{
	// whether use full orientation or not
	float fullAngle = desc_info.fullOrientation ? 360 : 180;
	// one additional bin for hof
	int nBins = desc_info.flagThre ? desc_info.nBins-1 : desc_info.nBins;
	// angle stride for quantization
	float angleBase = fullAngle/float(nBins);

	int width = descMat->width;
	int height = descMat->height;
	int histDim = descMat->nBins;

	int index = 0;

	for(int i = 0; i < height; i++) {
		const float* xcomp = (const float*)(xComp->imageData + xComp->widthStep*i);
		const float* ycomp = (const float*)(yComp->imageData + yComp->widthStep*i);

		// the histogram accumulated in the current line
		std::vector<float> sum(histDim);
		for(int j = 0; j < width; j++, index++) {
			float shiftX = xcomp[j];
			float shiftY = ycomp[j];
			float magnitude0 = sqrt(shiftX*shiftX+shiftY*shiftY);

			float magnitude1 = magnitude0;
			int bin0, bin1;

			// for the zero bin of hof
			if(desc_info.flagThre == 1 && magnitude0 <= desc_info.threshold) {
				bin0 = nBins; // the zero bin is the last one
				magnitude0 = 1.0;
				bin1 = 0;
				magnitude1 = 0;
			}
			else {
				float orientation = cvFastArctan(shiftY, shiftX);
				if(orientation > fullAngle)
					orientation -= fullAngle;

				// split the magnitude to two adjacent bins
				float fbin = orientation/angleBase;
				bin0 = cvFloor(fbin);
				float weight0 = 1 - (fbin - bin0);
				float weight1 = 1 - weight0;
				bin0 %= nBins;
				bin1 = (bin0+1)%nBins;

				magnitude0 *= weight0;
				magnitude1 *= weight1;
			}

			sum[bin0] += magnitude0;
			sum[bin1] += magnitude1;

			int temp0 = index*descMat->nBins;
			if(i == 0) { // for the first line
				for(int m = 0; m < descMat->nBins; m++)
					descMat->desc[temp0++] = sum[m];			
			}
			else {
				int temp1 = (index - width)*descMat->nBins;
				for(int m = 0; m < descMat->nBins; m++)
					descMat->desc[temp0++] = descMat->desc[temp1++]+sum[m];
			}
		}
	}
}


inline std::vector<float> Descriptors::getDesc(const DescMat* desc_mat, CvScalar rect, DescInfo desc_info)  
{
	int descDim = desc_info.dim;
	int height = desc_mat->height;
	int width = desc_mat->width;

	boost::numeric::ublas::vector<double> vec(descDim);
	int xOffset = rect.val[0];
	int yOffset = rect.val[1];
	int xStride = rect.val[2]/desc_info.nxCells;
	int yStride = rect.val[3]/desc_info.nyCells;

	// iterate over different cells
	int iDesc = 0;
	for (int iX = 0; iX < desc_info.nxCells; ++iX)
	for (int iY = 0; iY < desc_info.nyCells; ++iY) {
		// get the positions of the rectangle
		int left = xOffset + iX*xStride - 1;
		int right = std::min<int>(left + xStride, width-1);
		int top = yOffset + iY*yStride - 1;
		int bottom = std::min<int>(top + yStride, height-1);

		// get the index in the integral histogram
		int TopLeft = (top*width+left)*desc_info.nBins;
		int TopRight = (top*width+right)*desc_info.nBins;
		int BottomLeft = (bottom*width+left)*desc_info.nBins;
		int BottomRight = (bottom*width+right)*desc_info.nBins;

		for (int i = 0; i < desc_info.nBins; ++i, ++iDesc) {
			double sumTopLeft(0), sumTopRight(0), sumBottomLeft(0), sumBottomRight(0);
			if (top >= 0) {
				if (left >= 0)
					sumTopLeft = desc_mat->desc[TopLeft+i];
				if (right >= 0)
					sumTopRight = desc_mat->desc[TopRight+i];
			}
			if (bottom >= 0) {
				if (left >= 0)
					sumBottomLeft = desc_mat->desc[BottomLeft+i];
				if (right >= 0)
					sumBottomRight = desc_mat->desc[BottomRight+i];
			}
			float temp = sumBottomRight + sumTopLeft
					   - sumBottomLeft - sumTopRight;
			vec[iDesc] = std::max<float>(temp, 0) + epsilon;
		}
	}

	if (desc_info.norm == 1) // L1 normalization
		vec *= 1 / boost::numeric::ublas::norm_1(vec);
	else // L2 normalization
		vec *= 1 / boost::numeric::ublas::norm_2(vec);

	std::vector<float> desc(descDim);
	for (int i = 0; i < descDim; i++)
		desc[i] = vec[i];
	return desc;
}


inline void Descriptors::hogComp(IplImage* img, DescMat* desc_mat, DescInfo desc_info)
{
	int width = desc_mat->width;
	int height = desc_mat->height;
	IplImage* imgX = cvCreateImage(cvSize(width,height), IPL_DEPTH_32F, 1);
	IplImage* imgY = cvCreateImage(cvSize(width,height), IPL_DEPTH_32F, 1);
	cvSobel(img, imgX, 1, 0, 1);
	cvSobel(img, imgY, 0, 1, 1);
	buildDescMat(imgX, imgY, desc_mat, desc_info);
	cvReleaseImage(&imgX);
	cvReleaseImage(&imgY);
}


inline void Descriptors::hofComp(IplImage* flow, DescMat* desc_mat, DescInfo desc_info)
{
	int width = desc_mat->width;
	int height = desc_mat->height;
	IplImage* xComp = cvCreateImage(cvSize(width, height), IPL_DEPTH_32F, 1);
	IplImage* yComp = cvCreateImage(cvSize(width, height), IPL_DEPTH_32F, 1);
	for(int i = 0; i < height; i++) {
		const float* f = (const float*)(flow->imageData + flow->widthStep*i);
		float* xf = (float*)(xComp->imageData + xComp->widthStep*i);
		float* yf = (float*)(yComp->imageData + yComp->widthStep*i);
		for(int j = 0; j < width; j++) {
			xf[j] = f[2*j];
			yf[j] = f[2*j+1];
		}
	}
	buildDescMat(xComp, yComp, desc_mat, desc_info);
	cvReleaseImage(&xComp);
	cvReleaseImage(&yComp);
}


inline void Descriptors::mbhComp(IplImage* flow, DescMat* desc_mat_x, DescMat* desc_mat_y, DescInfo desc_info)
{
	int width = desc_mat_x->width;
	int height = desc_mat_x->height;
	IplImage* flowX = cvCreateImage(cvSize(width,height), IPL_DEPTH_32F, 1);
	IplImage* flowY = cvCreateImage(cvSize(width,height), IPL_DEPTH_32F, 1);
	IplImage* flowXdX = cvCreateImage(cvSize(width,height), IPL_DEPTH_32F, 1);
	IplImage* flowXdY = cvCreateImage(cvSize(width,height), IPL_DEPTH_32F, 1);
	IplImage* flowYdX = cvCreateImage(cvSize(width,height), IPL_DEPTH_32F, 1);
	IplImage* flowYdY = cvCreateImage(cvSize(width,height), IPL_DEPTH_32F, 1);

	// extract the x and y components of the flow
	for(int i = 0; i < height; i++) {
		const float* f = (const float*)(flow->imageData + flow->widthStep*i);
		float* fX = (float*)(flowX->imageData + flowX->widthStep*i);
		float* fY = (float*)(flowY->imageData + flowY->widthStep*i);
		for(int j = 0; j < width; j++) {
			fX[j] = 100*f[2*j];
			fY[j] = 100*f[2*j+1];
		}
	}

	cvSobel(flowX, flowXdX, 1, 0, 1);
	cvSobel(flowX, flowXdY, 0, 1, 1);
	cvSobel(flowY, flowYdX, 1, 0, 1);
	cvSobel(flowY, flowYdY, 0, 1, 1);

	buildDescMat(flowXdX, flowXdY, desc_mat_x, desc_info);
	buildDescMat(flowYdX, flowYdY, desc_mat_y, desc_info);

	cvReleaseImage(&flowX);
	cvReleaseImage(&flowY);
	cvReleaseImage(&flowXdX);
	cvReleaseImage(&flowXdY);
	cvReleaseImage(&flowYdX);
	cvReleaseImage(&flowYdY);
}

inline void Descriptors::kinemComp(IplImage* flow, DescMat* desc_mat_curldiv, DescMat* desc_mat_shear_c, DescMat* desc_mat_shear_d, DescInfo curldiv_info, DescInfo shearCD_info)
{


	int width = desc_mat_curldiv->width;
	int height = desc_mat_curldiv->height;

		IplImage* flowX = cvCreateImage(cvSize(width,height), IPL_DEPTH_32F, 1);
        IplImage* flowY = cvCreateImage(cvSize(width,height), IPL_DEPTH_32F, 1);
        IplImage* flowXdX = cvCreateImage(cvSize(width,height), IPL_DEPTH_32F, 1);
        IplImage* flowXdY = cvCreateImage(cvSize(width,height), IPL_DEPTH_32F, 1);
        IplImage* flowYdX = cvCreateImage(cvSize(width,height), IPL_DEPTH_32F, 1);
        IplImage* flowYdY = cvCreateImage(cvSize(width,height), IPL_DEPTH_32F, 1);


	// extract the x and y components of the flow
	for(int i = 0; i < height; i++) {
                const float* f = (const float*)(flow->imageData + flow->widthStep*i);
                float* fX = (float*)(flowX->imageData + flowX->widthStep*i);
                float* fY = (float*)(flowY->imageData + flowY->widthStep*i);
                for(int j = 0; j < width; j++) {
                        fX[j] = 100*f[2*j];
                        fY[j] = 100*f[2*j+1];
                }
        }


	    cvSobel(flowX, flowXdX, 1, 0, 1);
        cvSobel(flowX, flowXdY, 0, 1, 1);
        cvSobel(flowY, flowYdX, 1, 0, 1);
        cvSobel(flowY, flowYdY, 0, 1, 1);


	IplImage* shearImg = cvCreateImage(cvSize(width,height), IPL_DEPTH_32F, 1);
	IplImage* curlImg = cvCreateImage(cvSize(width,height), IPL_DEPTH_32F, 1);
	IplImage* divImg = cvCreateImage(cvSize(width,height), IPL_DEPTH_32F, 1);

	for(int i = 0; i < height; i++) {
				const float* xdx = (const float*)(flowXdX->imageData + flowXdX->widthStep*i);
                const float* xdy = (const float*)(flowXdY->imageData + flowXdY->widthStep*i);
                const float* ydx = (const float*)(flowYdX->imageData + flowYdX->widthStep*i);
                const float* ydy = (const float*)(flowYdY->imageData + flowYdY->widthStep*i);
		float* shear = (float*)(shearImg->imageData + shearImg->widthStep*i);
		float* curl = (float*)(curlImg->imageData + curlImg->widthStep*i);
		float* div = (float*)(divImg->imageData + divImg->widthStep*i);
		for(int j = 0; j < width; j++) {
                        div[j] = xdx[j] + ydy[j];
                        curl[j] = -xdy[j] + ydx[j];
                        float hyp1 = xdx[j] - ydy[j];
                        float hyp2 = xdy[j] + ydx[j];
                        shear[j] = sqrt(hyp1*hyp1+hyp2*hyp2);
		}
	}

        buildDescMat(divImg, curlImg, desc_mat_curldiv, curldiv_info);
        buildDescMat(curlImg, shearImg, desc_mat_shear_c, shearCD_info);
        buildDescMat(divImg, shearImg, desc_mat_shear_d, shearCD_info);


        cvReleaseImage(&flowX);
        cvReleaseImage(&flowY);
        cvReleaseImage(&flowXdX);
        cvReleaseImage(&flowXdY);
        cvReleaseImage(&flowYdX);
        cvReleaseImage(&flowYdY);
        cvReleaseImage(&shearImg);
        cvReleaseImage(&curlImg);
        cvReleaseImage(&divImg);
}

inline void Descriptors::opticalFlowTracker(IplImage* flow, std::vector<CvPoint2D32f>& points_in, std::vector<CvPoint2D32f>& points_out, std::vector<int>& status)  
{

	if(points_in.size() != points_out.size())
		fprintf(stderr, "the numbers of points don't match!");
	if(points_in.size() != status.size())
		fprintf(stderr, "the number of status doesn't match!");
	int width = flow->width;
	int height = flow->height;

	for(int i = 0; i < (signed)points_in.size(); i++) {
		CvPoint2D32f point_in = points_in[i];
		std::list<float> xs;
		std::list<float> ys;
		int x = cvFloor(point_in.x);
		int y = cvFloor(point_in.y);
		for(int m = x-1; m <= x+1; m++)
		for(int n = y-1; n <= y+1; n++) {
			int p = std::min<int>(std::max<int>(m, 0), width-1);
			int q = std::min<int>(std::max<int>(n, 0), height-1);
			const float* f = (const float*)(flow->imageData + flow->widthStep*q);
			xs.push_back(f[2*p]);
			ys.push_back(f[2*p+1]);
		}

		//Order the arrays
		xs.sort();
		ys.sort();
		
		//Median filter
		int size = xs.size()/2;
		for(int m = 0; m < size; m++) {
			xs.pop_back();
			ys.pop_back();
		}
		

		CvPoint2D32f offset;
		offset.x = xs.back();
		offset.y = ys.back();
		CvPoint2D32f point_out;
		point_out.x = point_in.x + offset.x;
		point_out.y = point_in.y + offset.y;


		points_out[i] = point_out;
		if( point_out.x > 0 && point_out.x < width && point_out.y > 0 && point_out.y < height)
			status[i] = 1;
		else
			status[i] = -1;
	}

}


inline int Descriptors::isValid(std::vector<CvPoint2D32f>& track, float& mean_x, float& mean_y, float& var_x, float& var_y, float& length)
{
	int size = track.size();
	for(int i = 0; i < size; i++) {
		mean_x += track[i].x;
		mean_y += track[i].y;
	}
	mean_x /= size;
	mean_y /= size;

	for(int i = 0; i < size; i++) {
		track[i].x -= mean_x;
		var_x += track[i].x*track[i].x;
		track[i].y -= mean_y;
		var_y += track[i].y*track[i].y;
	}
	var_x /= size;
	var_y /= size;
	var_x = sqrt(var_x);
	var_y = sqrt(var_y);
	// remove static trajectory
	if(var_x < min_var && var_y < min_var)
		return 0;
	// remove random trajectory
	if( var_x > max_var || var_y > max_var )
		return 0;

	for(int i = 1; i < size; i++) {
		float temp_x = track[i].x - track[i-1].x;
		float temp_y = track[i].y - track[i-1].y;
		length += sqrt(temp_x*temp_x+temp_y*temp_y);
		track[i-1].x = temp_x;
		track[i-1].y = temp_y;
	}

	float len_thre = length*0.7;
	for( int i = 0; i < size-1; i++ ) {
		float temp_x = track[i].x;
		float temp_y = track[i].y;
		float temp_dis = sqrt(temp_x*temp_x + temp_y*temp_y);
		if( temp_dis > max_dis && temp_dis > len_thre )
			return 0;
	}

	track.pop_back();
	// normalize the trajectory
	for(int i = 0; i < size-1; i++) {
		track[i].x /= length;
		track[i].y /= length;
	}
	return 1;
}



inline void Descriptors::cvDenseSample(IplImage* grey, IplImage* eig, std::vector<CvPoint2D32f>& points,const double quality, const double min_distance)
{
	int width = cvFloor(grey->width/min_distance);
	int height = cvFloor(grey->height/min_distance);
	double maxVal = 0;
	cvCornerMinEigenVal(grey, eig, 3, 3);
	cvMinMaxLoc(eig, 0, &maxVal, 0, 0, 0);
	const double threshold = maxVal*quality;

	int offset = cvFloor(min_distance/2);
	for(int i = 0; i < height; i++) 
	for(int j = 0; j < width; j++) {
		int x = cvFloor(j*min_distance+offset);
		int y = cvFloor(i*min_distance+offset);
		if(CV_IMAGE_ELEM(eig, float, y, x) > threshold) 
			points.push_back(cvPoint2D32f(x,y));
	}
}


inline void Descriptors::cvDenseSample(IplImage* grey, IplImage* eig, std::vector<CvPoint2D32f>& points_in,
				   std::vector<CvPoint2D32f>& points_out, const double quality, const double min_distance)
{
	int width = cvFloor(grey->width/min_distance);
	int height = cvFloor(grey->height/min_distance);
	double maxVal = 0;
	cvCornerMinEigenVal(grey, eig, 3, 3);
	cvMinMaxLoc(eig, 0, &maxVal, 0, 0, 0);
	const double threshold = maxVal*quality;

	std::vector<int> counters(width*height);
	for(int i = 0; i < (signed)points_in.size(); i++) {
		CvPoint2D32f point = points_in[i];
		if(point.x >= min_distance*width || point.y >= min_distance*height)
			continue;
		int x = cvFloor(point.x/min_distance);
		int y = cvFloor(point.y/min_distance);
		counters[y*width+x]++;
	}

	int index = 0;
	int offset = cvFloor(min_distance/2);
	for(int i = 0; i < height; i++) 
	for(int j = 0; j < width; j++, index++) {
		if(counters[index] == 0) {
			int x = cvFloor(j*min_distance+offset);
			int y = cvFloor(i*min_distance+offset);
			if(CV_IMAGE_ELEM(eig, float, y, x) > threshold) 
				points_out.push_back(cvPoint2D32f(x,y));
		}
	}
}


#endif // CUWFLOW_DESCRIPTORS_H_
