/**
* \file descriptorsgpu.h
*
* This header file contains the declaration of some methods which are implemented with CUDA to acelerate some CPU
* functions from the original code. Futhermore, it contains memory allocations, kernel calls, methods from the gpu
* module of OpenCV, etc. Due to the big amount of gpu memory which is managed during these processes, some structs have been declarated.
*	
*	\author Fernando Cobo Aguilera
*	\date October 2014
*/

#ifndef WFLOW_DESCRIPTORS_GPU_H
#define WFLOW_DESCRIPTORS_GPU_H

#include "iplimagepyramid.h"

#include <iostream>
#include <vector>
#include <opencv2/gpu/gpu.hpp>
#include <opencv/cv.h>

#include "cuda_runtime.h"
#include "cuda.h"
#include "device_launch_parameters.h"

#include "kernels.cuh"


using namespace std;

/** 
* \struct GpuMemoryOpticalFlow
* This struct is used to manage the gpu/cpu memory in order to calculate the optical flow. This requires gpu memory for
* the images (dataX & dataY), gpu memory to save the optical flow (opticalFlowX & opticalFlowY) and last,
* cpu memory to download the optical flow (opticalFlowCPU).
*/

struct GpuMemoryOpticalFlow{

	//gpu memory
	cv::gpu::GpuMat *d_data_x;
	cv::gpu::GpuMat *d_data_y;

	//gpu memory
	cv::gpu::GpuMat d_optical_flow_x;
	cv::gpu::GpuMat d_optical_flow_y;
	
	//cpu memory
	std::vector<vector<cv::Mat> > h_optical_flow;
	
	cv::gpu::FarnebackOpticalFlow optical_flow;	
};


/** 
* \struct GpuMemoryOpticalFlowTracker
* This struct is used to manage the gpu/cpu memory in order to calculate the optical flow tracker,
* that is to say, median filters. For its implementation, the cpu memory is splitted in two arrays.
* The lists from the cpu version whose size was 9 elements, are transformed into 8-elements arrays and
* one element arrays (9th element from the cpu lists). 
*/
struct GpuMemoryOpticalFlowTracker{
	
	//cpu memory
	float *h_arr_x;
	float *h_arr_y;
	float *h_middle_x;
	float *h_middle_y;

	//To upload arrays to gpu memory
	float *d_middle;
	float *d_arr;

	//Stream for the asynchronous version
	cudaStream_t stream;
};



/** 
* \struct GpuMemoryIntegralHistogram
* This struct is used to manage the gpu/cpu memory in order to calculate the integral histograms.
* To upload sobel derivates to gpu memory asynchronously, page locked memory is needed 
* (h_flow_x_dx, h_flow_x_dy, h_flow_y_dx, h_flow_y_dy). To keep this data in the gpu, GpuMats are used.
* The struct is used for either culrDiv, shearC and shearD histograms or mbh and hog histograms.
*/
struct GpuMemoryIntegralHistogram{

	//culrDiv, shearC, shearD, mbh histograms

	//Sobel derivates in the gpu
	cv::gpu::GpuMat *d_flow_x_dx;
	cv::gpu::GpuMat *d_flow_x_dy;
	cv::gpu::GpuMat *d_flow_y_dx;
	cv::gpu::GpuMat *d_flow_y_dy;

	//Sobel derivates in the cpu
	cv::gpu::CudaMem *h_flow_x_dx;
	cv::gpu::CudaMem *h_flow_x_dy;
	cv::gpu::CudaMem *h_flow_y_dx;
	cv::gpu::CudaMem *h_flow_y_dy;

	//Hog histograms
	cv::gpu::GpuMat *d_hog_x;
	cv::gpu::GpuMat *d_hog_y;
	cv::gpu::CudaMem *h_hog_x;
	cv::gpu::CudaMem *h_hog_y;
};



/**
* \class GpuDescriptors
*
* \brief This class contains a group of methods related with the descriptors calculation in the gpu 
*
* \author Fernando Cobo Aguilera
* \date 30/09/2014
*/
class GpuDescriptors{

public:

/**
* The function allocates arrays in gpu memory 
* \param num_scale Number of arrays that will be allocated
* \return Pointer to these arrays
*/
	static cv::gpu::GpuMat * createPointers(int num_scale);

/**
* The function initializes an instance of Farneback optical flow gpu with deafult values as
* the cpu version does.
 \param flow Pointer to a Farneback optical flow instance
*/
	static void initializeFarnebackOpticalFlow(cv::gpu::FarnebackOpticalFlow *flow);
	
/**
* This function uploads the data of a set of images to gpu memory
* \param pyramid Set or pyramid of images which will be uploaded
* \param num_scale Number of levels of the pyramid or, the total number of images
* \param arr Gpu memory destination
*/	
	static void uploadImagesToGpu(IplImagePyramid pyramid, int num_scale, cv::gpu::GpuMat *arr);

/**
* This function uploads an image to gpu, calculates the optical flow and last, 
* the flow is downloaded to the cpu memory. In order to use this function, the user needs to
* have called <tt>uploadImagesToGpu</tt> before to upload to gpu memory the first frame of a video.
* Afer this, in every iteration, one image will be uploaded while the other one, for the optical flow,
* is already in gpu memory.
* \param data Array with the image that must be uploaded
* \param frame_num The frame number in relation with all the sequence of a video
* \param scale Total number of levels
* \param memory Struct to manage the memory
*/
	static cv::Mat opticalFlowGpu(cv::Mat data,int frame_num, int scale, struct GpuMemoryOpticalFlow *memory);
	
/**
* This function  tracks interest points by median filtering in the optical field using the GPU. 
* This is the synchronous version, that is to say, the cpu will wait until the gpu finishs. Thus,
* the output interest point positions, the status for successfully tracked and the median filters
* are calculated.
* \param flow Array with the optical flow of two images
* \param points_in Array whith the input interest point positions 
* \param points_out  Array whith the output interest point positions
* \param status Array with the status of every <tt>points_out<\tt> for successfully tracked or not
*/	
	static void opticalFlowTrackerGpuSync(IplImage* flow, std::vector<CvPoint2D32f>& points_in, std::vector<CvPoint2D32f>& points_out,std::vector<int>& status,struct GpuMemoryOpticalFlowTracker *memory);
	
/**
* This function  tracks interest points by median filtering in the optical field using the GPU. 
* This is the asynchronous version, that is to say, the cpu will not be blocked. Nevertheless, the output
* interest point positions and the status for successfully tracked will not be calculated inside the function. 
* It will be necessary to calculate them in the main program.
* \param flow Array with the optical flow of two images
* \param points_in Array whith the input interest point positions 
* \param memory Struct for the memory managing
*/	
	static void opticalFlowTrackerGpuAsync(IplImage* flow,std::vector<CvPoint2D32f>& points_in, struct GpuMemoryOpticalFlowTracker memory);
	
/**
* This function releases the gpu memory from the optical flow tracker. While the synchronous version does it itself, the asynchronous
* version needs to call this function.
* \param memory Struct with all the pointers to cpu and gpu memory
*/	
	static void freeMemoryOpricalFlowTrackerGpu(struct GpuMemoryOpticalFlowTracker memory);
	
/**
* This function calculates the the DCS descriptor using the GPU. This version is synchronous.
* Despite the fact that the uploads and kernels are called asynchronously, scan operations are
* calculated in the cpu, thus, the asynchronicity desappears
* \param vec Array with the optical flow splitted in the two components.
* \param stream Cuda stream for the asynchronous version
* \param n_bins nBins parameter from the descriptor
* \param parameters Array with some parameters about the descriptor
* \param d_x_dx Gpu array for the sobel derivate xdx 
* \param d_x_dy Gpu array for the sobel derivate xdy
* \param d_y_dx Gpu array for the sobel derivate ydy
* \param d_y_dy Gpu array for the sobel derivate ydy
* \param h_x_dx Pinned memory array for the sobel derivate xdx 
* \param h_x_dy Pinned memory array for the sobel derivate xdy
* \param h_y_dx Pinned memory array for the sobel derivate ydx 
* \param h_y_dy Pinned memory array for the sobel derivate ydy
* \param desc_mat_curldiv Array for the output integral histogram for curldiv
* \param desc_mat_shear_c Array for the output integral histogram for shearC
* \param desc_mat_shear_d Array for the output integral histogram for shearD
*/	
	static void kinemCompGpuSync(std::vector<cv::Mat> vec, cudaStream_t stream,  int n_bins, float *parameters,
				  cv::gpu::GpuMat d_x_dx, cv::gpu::GpuMat d_x_dy, cv::gpu::GpuMat d_y_dx,  cv::gpu::GpuMat d_y_dy,
				  cv::gpu::CudaMem h_x_dx,cv::gpu::CudaMem h_x_dy,cv::gpu::CudaMem h_y_dx,cv::gpu::CudaMem h_y_dy,
				  float * desc_mat_curldiv, float *desc_mat_shear_c, float *desc_mat_shear_d);
	
/**
* This function calculates the the DCS descriptor using the GPU. This version is asynchronous.
*  All the calculations are done in the gpu asynchronously, even the scan operation
* \param vec Array with the optical flow splitted in the two components.
* \param stream Cuda stream for the asynchronous version
* \param n_bins nBins parameter from the descriptor
* \param parameters Array with some parameters about the descriptor
* \param d_sum Auxiliar array to download the output integral histograms
* \param d_x_dx Gpu array for the sobel derivate xdx 
* \param d_x_dy Gpu array for the sobel derivate xdy
* \param d_y_dx Gpu array for the sobel derivate ydy
* \param d_y_dy Gpu array for the sobel derivate ydy
* \param h_x_dx Pinned memory array for the sobel derivate xdx 
* \param h_x_dy Pinned memory array for the sobel derivate xdy
* \param h_y_dx Pinned memory array for the sobel derivate ydx 
* \param h_y_dy Pinned memory array for the sobel derivate ydy
* \param desc_mat_curldiv Array for the output integral histogram for curldiv
* \param desc_mat_shear_c Array for the output integral histogram for shearC
* \param desc_mat_shear_d Array for the output integral histogram for shearD

*/	
	static void kinemCompGpuAsync(std::vector<cv::Mat> vec, cudaStream_t stream, int n_bins, float *parameters, float *d_sum,
				  cv::gpu::GpuMat d_x_dx, cv::gpu::GpuMat d_x_dy, cv::gpu::GpuMat d_y_dx,  cv::gpu::GpuMat d_y_dy,
				  cv::gpu::CudaMem h_x_dx,cv::gpu::CudaMem h_x_dy,cv::gpu::CudaMem h_y_dx,cv::gpu::CudaMem h_y_dy,
				  float * desc_mat_curldiv, float *desc_mat_shear_c, float *desc_mat_shear_d);


/**
* This function calculates mbh and hog descriptors using the GPU. This version is asynchronous.
*  All the calculations are done in the gpu asynchronously, even the scan operation
* \param vec Array with the optical flow splitted in the two components.
* \param img Array with an image for the hog descriptors
* \param stream Cuda stream for the asynchronous version
* \param n_bins nBins parameter from the descriptor
* \param parameters Array with some parameters about the descriptor
* \param d_sum Auxiliar array to download the output integral histograms
* \param d_x_dx Gpu array for the sobel derivate xdx of mbh descriptor
* \param d_x_dy Gpu array for the sobel derivate xdy of mbh descriptor
* \param d_y_dx Gpu array for the sobel derivate ydy of mbh descriptor
* \param d_y_dy Gpu array for the sobel derivate ydy of mbh descriptor
* \param d_hog_x Gpu array for the sobel derivate dx of hog descriptor
* \param d_hog_y Gpu array for the sobel derivate dy of hog descriptor
* \param h_x_dx Pinned memory array for the sobel derivate xdx of mbh descriptor 
* \param h_x_dy Pinned memory array for the sobel derivate xdy of mbh descriptor
* \param h_y_dx Pinned memory array for the sobel derivate ydx of mbh descriptor 
* \param h_y_dy Pinned memory array for the sobel derivate ydy of mbh descriptor
* \param h_hog_x Pinned memory array for the sobel derivate dx of hog descriptor
* \param h_hog_y Pinned memory array for the sobel derivate dy of hog descriptor
* \param desc_mat_x Array for the output integral histogram for the component X of mbh
* \param desc_mat_y Array for the output integral histogram for the component Y of mbh
* \param desc_mat Array for the output integral histogram for hog
*/
	static void hogMbhCompGpuAsync(std::vector<cv::Mat> vec, IplImage *img, cudaStream_t stream, int n_bins,float *parameters, float *d_sum,
			   cv::gpu::GpuMat d_x_dx, cv::gpu::GpuMat d_x_dy, cv::gpu::GpuMat d_y_dx,  cv::gpu::GpuMat d_y_dy,
			   cv::gpu::GpuMat d_hog_x, cv::gpu::GpuMat d_hog_y,
			   cv::gpu::CudaMem h_x_dx,cv::gpu::CudaMem h_x_dy,cv::gpu::CudaMem h_y_dx,cv::gpu::CudaMem h_y_dy,
			   cv::gpu::CudaMem h_hog_x,cv::gpu::CudaMem h_hog_y,
			   float * desc_mat_x, float *desc_mat_y, float *desc_mat);
};



#endif //WFLOW_DESCRIPTORS_GPU_H
