#ifndef CUWFLOW_OWNKERNELS_H_
#define CUWFLOW_OWNKERNELS_H_


#include "cuda_runtime.h"
#include "cuda.h"
#include "device_launch_parameters.h"

#include <opencv/cv.h>
#include <opencv/highgui.h>
#include <opencv2/gpu/gpu.hpp>

/**
* \file ownkernels.cuh
*  This header file contains the definition of some functions that will be 
* used to acelerate the algorithm
*/



/**
* \fn void copyToConstantMemory(float *parameters, cudaStream_t stream)
* \brief This method loads some parameters to constant memory. Parameters
 which are needed in kernels like BuildDescMatKernel or BuildDescMatKernelv2
 * \param parameters Parameters which will be loaded
 * \param stream Cuda stream for the asynchronous version
*/
void copyToConstantMemory(float *parameters, cudaStream_t stream);

/**
* \fn void buildDescMatMethodv2(int height, int width, cv::gpu::GpuMat m1, cv::gpu::GpuMat m2, float *sum, cudaStream_t stream)
* \brief This method calls the kernel BuildDescMatKernelv2
* \param height Kernel blocks
* \param width Kernel threads
* \param d_m1 x gradient component
* \param d_m2 y gradient component
* \param stream Cuda stream for the asynchronous version
* \param d_sum Output integral histograms
*/
void buildDescMatMethodv2(int height, int width, cv::gpu::GpuMat d_m1, cv::gpu::GpuMat d_m2, cudaStream_t stream, float *d_sum);


/**
* \fn void BuildDescMatMethod(int height, int width, cv::gpu::GpuMat m1, cv::gpu::GpuMat m2, float *sum, cudaStream_t stream)
* \brief This method calls the kernel BuildDescMatKernel
* \param height Kernel blocks
* \param width Kernel threads
* \param d_m1 x gradient component
* \param d_m2 y gradient component
* \param stream Cuda stream for the asynchronous version
* \param d_sum Output integral histograms
*/
void buildDescMatMethod(int height, int width, cv::gpu::GpuMat d_m1, cv::gpu::GpuMat d_m2, cudaStream_t stream, float *d_sum);

/**
* \fn void KinemCompMethod(int height, int width, cv::gpu::GpuMat xDx,cv::gpu::GpuMat xDy, cv::gpu::GpuMat yDx,cv::gpu::GpuMat yDy, 
					 cv::gpu::GpuMat divImg, cv::gpu::GpuMat curlImg,cv::gpu::GpuMat shearImg,cudaStream_t stream)
* \brief This method calls the kernel KinemCompKernel
* \param height Kernel blocks
* \param width Kernel threads
* \param x_dx Array with the sobel derived xdx
* \param x_dy Array with the sobel derived xdy
* \param y_dx Array with the sobel derived ydx
* \param y_dy Array with the sobel derived ydy
* \param stream Cuda stream for the asynchronous version
* \param div_img Destination array of the first descriptor
* \param curl_img Destination array of the second descriptor
* \param shear_img Destination array of the third descriptor
*/
void kinemCompMethod(int height, int width, cv::gpu::GpuMat x_dx,cv::gpu::GpuMat x_dy, cv::gpu::GpuMat y_dx,cv::gpu::GpuMat y_dy, 
					 cudaStream_t stream, cv::gpu::GpuMat div_img, cv::gpu::GpuMat curl_img,cv::gpu::GpuMat shear_img);


#endif //CUWFLOW_OWNKERNELS_H_