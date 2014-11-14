/**
* \file densetrack.h
*
* This header file contains the declaration of all the parameters that the algorithm will need to be executed
*/

#ifndef CUWFLOW_DENSETRACK_H_
#define CUWFLOW_DENSETRACK_H_

#include <opencv/cv.h>
#include <opencv/highgui.h>
#include <opencv2/gpu/gpu.hpp>

#include <ctype.h>
#include <algorithm>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <fstream>
#include <iostream>
#include <vector>
#include <list>
#include <string>

#include "cuda_runtime.h"
#include "cuda.h"
#include "device_launch_parameters.h"

#include "iplimagewrapper.h"
#include "iplimagepyramid.h"
#include <boost/numeric/ublas/vector.hpp>
#include <boost/numeric/ublas/io.hpp>

using namespace std;



//! Set show_track = 1, if you want to visualize the trajectories
int show_track = 0; 

//! Name of the video
std::string video_name;

//! Path model
std::string  model_path;

//! Parameter for the optical flow in the gpu
bool flow_gpu = false;
//! Parameter for the optical flow tracker in the gpu
bool optical_flow_tracker_gpu = false;
//! Parameter for the optical flow in the gpu
bool integral_histograms_gpu = false;

//! Number of levels of a pyramid instance that will be used for the gpu optical flow
int flow_gpu_range = 0;
 //! Number of levels of a pyramid instance that will be used for the gpu integral histograms
int histograms_gpu_range = 0;
//! Number of levels of a pyramid instance that will be used for the gpu optical flow tracker
int optical_flow_tracker_gpu_range = 0;

//! Float scale values
float* fscales = 0; 		

//! Parameter for descriptors
int patch_size = 32;
//! Parameter for descriptors
int nxy_cell = 2;
//! Parameter for descriptors
int nt_cell = 3;
//! Parameter for descriptors
bool full_orientation = true;
//! Parameter for descriptors
float epsilon = 0.05;
//! Parameter for descriptors
const float min_flow = 0.4*0.4;

//! First frame that will be tracked
int start_frame = 0;
//! Last frame that will be tracked
int end_frame = 99999;
//! Parameter for tracking
double quality = 0.001;
//! Parameter for tracking
double min_distance = 5;
//! Parameter for tracking
int init_gap = 1;
//! Parameter for tracking
int track_length = 15;

//! 0: use Optical flow for features | 1: use wFlow (Optical flow - Affine flow) to compute w-features (version 1 is disabled)
int w_flow = 0;		
//! 0: traj, hog, hof/w-hof (hof hofwflow), mbh | 1: traj, dcs (curldiv shearC shearD)
int feat_comb = 1;	
//! Write Affine model files if 1 else compute descriptors
int compute_models = 0;		

//! Parameter for the trajectory descriptor
const float min_var = sqrt(3);
//! Parameter for the trajectory descriptor
const float max_var = 50;
//! Parameter for the trajectory descriptor
const float max_dis = 20;

//! Parameter for multi-scale
int scale_num = 8;  
//! Parameter for multi-scale
const float scale_stride = sqrt(2);

//! Parameter to save hog descriptor
bool hogdesc = false;
//! Parameter to save hof descriptor
bool hofdesc = false;
//! Parameter to save mbh descriptor
bool mbhdesc = false;
//! Parameter to save curldiv descriptor
bool curldivdesc = false;
//! Parameter to save shearC descriptor
bool shearCdesc = false;
//! Parameter to save shearD descriptor
bool shearDdesc = false;


/** 
* \struct TimesAlgorithm
* This struct is used to manage the algorithm times. It has two
* version depending on the operative system used.
*/
#if defined _WIN32
 struct TimesAlgorithm{

	float optical_flow_time; //Optical flow time
	float optical_flow_tracker_time; //Optical flow tracker time
	float integral_histograms_time; //Integral histograms time
	float get_descriptors_time;//Get descriptors time
	float total_time; //Total time of the algorithm

	//Variables to calculate the times
	float start; 
	float end;
	float start_total;
	float end_total;

};
#else
#include <sys/time.h>
 struct TimesAlgorithm{

	float optical_flow_time; //Optical flow time
	float optical_flow_tracker_time; //Optical flow tracker time
	float integral_histograms_time; //Integral histograms time
	float get_descriptors_time;//Get descriptors time
	float total_time; //Total time of the algorithm

	//Variables to calculate the times
	timeval start;
	timeval end;	
	timeval start_total;
	timeval end_total;
};
#endif


 /**
 * This function gets the start time in some point of the algorithm
 * \param times Struct with the times
 * \param total True when the total time is being calculated, otherwise, false
 */
 void startTime(struct TimesAlgorithm *times, bool total){

#if defined _WIN32
	 if(total)
		times->start_total = clock();
	 else
		 times->start = clock();
#else
	 if(total)
		gettimeofday(&times->start_total,NULL);
	 else
		 gettimeofday(&times->start,NULL);
#endif
 }

  /**
 * This function gets the end time in some point of the algorithm
 * \param times Struct with the times
 * \param total True when the total time is being calculated, otherwise, false
 */
void endTime(struct TimesAlgorithm *times,  bool total){

#if defined _WIN32
	if(total)
		times->end_total = clock();
	else
		times->end = clock();
#else
	if(total)
		gettimeofday(&times->end_total,NULL);
	else
		gettimeofday(&times->end,NULL);
#endif
 }


 /**
 * This function calculates the time difference between a start time and an end time
 * \param times Struct with the times
 * \param total True when the total time is being calculated, otherwise, false
 * \return The time calculated
 */
double calculateTime(struct TimesAlgorithm *times, bool total){

	double result = 0;

#if defined _WIN32
	if(total)
		result = difftime(times->end_total,times->start_total);
	else
		result = difftime(times->end,times->start);
#else
	if(total){
		result += (times->end_total.tv_sec - times->start_total.tv_sec) * 1000.0;
		result += (times->end_total.tv_usec - times->start_total.tv_usec) / 1000.0;
	}
	else{
		result += (times->end.tv_sec - times->start.tv_sec) * 1000.0;
		result += (times->end.tv_usec - times->start.tv_usec) / 1000.0;
	}
#endif

	return result;
}


typedef struct TrackerInfo
{
    int trackLength; // length of the trajectory
    int initGap; // initial gap for feature detection
}TrackerInfo;

typedef struct DescInfo
{
    int nBins; // number of bins for vector quantization
    int fullOrientation; // 0: 180 degree; 1: 360 degree
    int norm; // 1: L1 normalization; 2: L2 normalization
    float threshold; //threshold for normalization
	int flagThre; // whether thresholding or not
    int nxCells; // number of cells in x direction
    int nyCells; 
    int ntCells;
	int dim; // dimension of the descriptor
    int blockHeight; // size of the block for computing the descriptor
    int blockWidth;
}DescInfo; 

typedef struct DescMat
{
    int height;
    int width;
    int nBins;
    float* desc;

}DescMat;

class PointDesc
{
public:
    std::vector<float> hog;
    std::vector<float> hof;
    std::vector<float> hof_wFlow;
    std::vector<float> mbhX;
    std::vector<float> mbhY;
    std::vector<float> curldiv;
    std::vector<float> shearC;
    std::vector<float> shearD;
    CvPoint2D32f point;

    PointDesc(const DescInfo& hogInfo, const DescInfo& hofInfo, const DescInfo& mbhInfo, const CvPoint2D32f& point_)
        : hog(hogInfo.nxCells * hogInfo.nyCells * hogInfo.nBins),
        hof(hofInfo.nxCells * hofInfo.nyCells * hofInfo.nBins),
        mbhX(mbhInfo.nxCells * mbhInfo.nyCells * mbhInfo.nBins),
                mbhY(mbhInfo.nxCells * mbhInfo.nyCells * mbhInfo.nBins),
        point(point_)
    {}

    PointDesc(int dummy, const DescInfo& hogInfo, const DescInfo& hofInfo, const DescInfo& mbhInfo, const CvPoint2D32f& point_)
        : hog(hogInfo.nxCells * hogInfo.nyCells * hogInfo.nBins),
        hof(hofInfo.nxCells * hofInfo.nyCells * hofInfo.nBins),
	hof_wFlow(hofInfo.nxCells * hofInfo.nyCells * hofInfo.nBins),
        mbhX(mbhInfo.nxCells * mbhInfo.nyCells * mbhInfo.nBins),
                mbhY(mbhInfo.nxCells * mbhInfo.nyCells * mbhInfo.nBins),
        point(point_)
    {}

    PointDesc(const DescInfo& curldivInfo, const DescInfo& shearCDInfo, const CvPoint2D32f& point_)
        : curldiv(curldivInfo.nxCells * curldivInfo.nyCells * curldivInfo.nBins),
        shearC(shearCDInfo.nxCells * shearCDInfo.nyCells * shearCDInfo.nBins),
        	shearD(shearCDInfo.nxCells * shearCDInfo.nyCells * shearCDInfo.nBins),
        point(point_)
    {}
};

class Track
{
public:
    std::list<PointDesc> pointDescs;
    int maxNPoints;

    Track(int maxNPoints_)
        : maxNPoints(maxNPoints_)
    {}

    void addPointDesc(const PointDesc& point)
    {
        pointDescs.push_back(point);
        if ((signed)pointDescs.size() > maxNPoints + 2) {
            pointDescs.pop_front();
		}
    }
};

#endif //CUWFLOW_DENSETRACK_H_
