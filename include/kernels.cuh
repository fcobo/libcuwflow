/**
* \file kernels.cuh
*  This header file contains the definition of some functions that will be 
* used to acelerate the algorithm
*/

#ifndef CUWFLOW_KERNELS_H_
#define CUWFLOW_KERNELS_H_


#include "cuda_runtime.h"
#include "cuda.h"
#include "device_launch_parameters.h"

#include <opencv/cv.h>
#include <opencv/highgui.h>
#include <opencv2/gpu/gpu.hpp>



			
/*
* \fn void scankernel(float *g_odata, float *g_idata, cudaStream_t stream,int width, dim3 dims)
* \brief This method calls the kernel scan
* \param d_idata Source array
* \param stream Cuda stream for the asynchronous version
* \param width Kernel threads
* \param dims Kernel blocks
* \param d_odata Destination array
*/
void scanKernel(float *d_idata, cudaStream_t stream,int width, dim3 dims, float *d_odata);

/**
* \fn void mergeSort(float *d_SrcKey,float* d_elementNine, cudaStream_t stream, int nBlocks, int nThreads, int sharedMemory)
* \brief This method calls the kernel mergeSortSharedKernel
* \param d_src_key Source array with some 8-elements vectors
* \param d_element_nine Source array with the ninth element of each 8-elements vector. The results will be saved in this array.
* \param stream Cuda stream for the asynchronous version
* \param n_blocks Kernel blocks
* \param n_threads Kernel threads
* \param shared_memory Size of the shared memory
*/
void mergeSort(float *d_src_key, cudaStream_t stream, int n_blocks, int n_threads, int shared_memory,float* d_element_nine);

#endif //CUWFLOW_KERNELS_H_