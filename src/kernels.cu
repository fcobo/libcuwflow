/*
 * Copyright 1993-2013 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */


/**
* \file kernels.cu
*  This file contains the implementation of kernels which will be used to acelerate the algorithm.
*
* This file has a set of kernels that can be used to acelerate the algorithm. 
* Every main kernel has an individual method to call it
*/


#include "kernels.cuh"


/**
* CUDA constant memory variable
*
* This variable is needed when the user runs BuildDescMatKernel or
* BuildDescMatKernelv2. They both need a set of parameters, thus,
* the constant memory is ideal in these cases.
*/
__constant__ float parameters_kinem[3];


/**
* \fn int binarySearchInclusive(float val, float *data, int L, int stride)
* \brief Helper function of mergeSortSharedKernel kernel which finds the right position of a 
  specified input value to get an ordered vector (Bounds of the array included)
* \param val Specified input value
* \param data Array which i been used
* \param stride Level number of the merge sort (power of two)
* \return the position found  
*/
 __device__ int binarySearchInclusive(float val, float *data, int stride)
{

    int pos = 0;

    for (; stride > 0; stride >>= 1)
    {
        int newPos = stride;

        if (data[newPos - 1] <= val)
            pos = newPos;
    }

    return pos;
}

/**
* \fn  int binarySearchExclusive(float val, float *data, int stride)
* \brief Helper function of mergeSortSharedKernel kernel which finds the right position of a 
  specified input value to get an ordered vector. (Bounds of the array excluded)
* \param val Specified input value
* \param data Array which is been used
* \param stride Level number of the merge sort (power of two)
*/
 __device__ int binarySearchExclusive(float val, float *data, int stride)
{

    int pos = 0;

    for (; stride > 0; stride >>= 1)
    {
        int newPos = stride;

        if (data[newPos - 1] < val)
            pos = newPos;
    }

    return pos;
}

 /**
* \fn void mergeSortSharedKernel(float *d_SrcKey, float* d_elementNine, int sizeSharedMemory)
* \brief Bottom-level merge sort (binary search-based) kernel. It is a variante of the original
   code from CUDA kit, adapted to merge several arrays of 9 elements.
* \param d_src_key Source vector with all the 8-elements arrays to order. The size must be a multiple of 8
* \param d_element_nine Source vector with all the ninth elements of each 8-elements array from d_SrcKey
* \param size_shared_memory Size of the shared memory due to the memory is allocated dynamically

*/
__global__ void mergeSortSharedKernel(float *d_src_key, float* d_element_nine, int size_shared_memory)
{
	//Dynamic shared memory
	extern __shared__ float s_key[];

    d_src_key += blockIdx.x * size_shared_memory + threadIdx.x;
 
	//Loading array, two accesses per thread
    s_key[threadIdx.x +                      0] = d_src_key[0];
    s_key[threadIdx.x + (size_shared_memory / 2)] = d_src_key[(size_shared_memory / 2)];


	int stride;

	//Loop adapted to order arrays of 8 elements, rather than a whole array of 8*x elements
    for (stride = 1; stride < 8; stride *= 2)
    {

        int     lPos = threadIdx.x & (stride - 1);
        float *baseKey = s_key + 2 * (threadIdx.x - lPos);

        __syncthreads();
        float keyA = baseKey[lPos +      0];
        float keyB = baseKey[lPos + stride];
        int posA = binarySearchExclusive(keyA, baseKey + stride, stride) + lPos;
        int posB = binarySearchInclusive(keyB, baseKey +      0, stride) + lPos;

        __syncthreads();
        baseKey[posA] = keyA;
        baseKey[posB] = keyB;
    }
	

    __syncthreads();

	//Extra code from the original version to compare the ninth element of each array with the 3th and 4th element
	//of each 8-elements array, which are already ordered.
	float aux;
	if( threadIdx.x % 4 == 0){

		float lowerBound  = s_key[threadIdx.x*2 + 3];
		float higherBound = s_key[threadIdx.x*2 + 4];
		
		//The ninth element is considered as the medianis to say
		aux = d_element_nine[blockIdx.x*256 + (threadIdx.x / 4)];

		//Check if the 3th element is bigger than the ninth element
		if(aux <= lowerBound)
			d_element_nine[blockIdx.x*256 + (threadIdx.x / 4)] = lowerBound;
		else{
			//Check if the 4th element is smaller than the ninth element
		    if(aux >= higherBound)
			 d_element_nine[blockIdx.x*256 + (threadIdx.x / 4)] = higherBound;	
		}
	}
}

/**
* \fn void scan(float *g_odata, float *g_idata)
* \brief This kenel calculates the parallel prefix sum or scan of several arrays.
   The elements of every array are non-contiguos, that is to say, there is a stride 
* \param g_odata Output array
* \param g_idata Input array
*/
__global__ void scan(float *g_odata, float *g_idata){

	extern __shared__ float temp[]; // allocated on invocation

	int tid = threadIdx.x;

	//Loading array to shared memory with the corresponding stride
	temp[tid] = g_idata[threadIdx.x*gridDim.x + blockIdx.x + (blockIdx.y*gridDim.x*blockDim.x)];

	float aux;

	__syncthreads();

	for(int offset = 1; offset < blockDim.x; offset *= 2){

		if( tid >= offset){

			aux = temp[tid-offset];
			__syncthreads();

			temp[tid] += aux;
		}
		else{
			temp[tid] = temp[tid];
		}
	}

	//Loading results to output array
	g_odata[threadIdx.x*gridDim.x + blockIdx.x + (blockIdx.y*gridDim.x*blockDim.x)] = temp[tid]; // write output 
}

/**
* \fn void kinemCompKernel(cv::gpu::PtrStepSz<float> x_dx,cv::gpu::PtrStepSz<float> x_dy, 
								cv::gpu::PtrStepSz<float> y_dx,cv::gpu::PtrStepSz<float> y_dy, 
								cv::gpu::PtrStepSz<float> out1, cv::gpu::PtrStepSz<float> out2,cv::gpu::PtrStepSz<float> out3)
* \brief This kernel alludes a portion of KinemComp method from CPU version. It helps to accelerate some calculations like
   square root. The results are splitted in three arrays
* \param x_dx Source array, correponding with the output of a Sobel derived (xdx)
* \param x_dy Source array, correponding with the output of a Sobel derived (xdy)
* \param y_dx Source array, correponding with the output of a Sobel derived (ydx)
* \param y_dy Source array, correponding with the output of a Sobel derived (ydy)
* \param out1 First output array
* \param out2 Second output array
* \param out3 Third output array
*/
__global__ void kinemCompKernel(cv::gpu::PtrStepSz<float> x_dx,cv::gpu::PtrStepSz<float> x_dy, 
								cv::gpu::PtrStepSz<float> y_dx,cv::gpu::PtrStepSz<float> y_dy, 
								cv::gpu::PtrStepSz<float> out1, cv::gpu::PtrStepSz<float> out2,cv::gpu::PtrStepSz<float> out3){

	int block = blockIdx.x * blockDim.x;
	int j = block + threadIdx.x;

	float xdx = x_dx.ptr(0)[j];
	float xdy = x_dy.ptr(0)[j];
	float ydx = y_dx.ptr(0)[j];
	float ydy = y_dy.ptr(0)[j];

	out1[j] = xdx + ydy;
	out2[j] = -xdy + ydx;
                   
	float hyp1 = xdx - ydy;
    float hyp2 = xdy + ydx;
	
	//Square root
	float out = sqrt(hyp1*hyp1+hyp2*hyp2);
                      
	out3[j] = out;

	__syncthreads();
}

/**
* \fn void buildDescMatKernel(cv::gpu::PtrStepSz<float> in1, cv::gpu::PtrStepSz<float> in2, float *d_sum)
* \brief This kernel alludes the main loop of BuildDescMat method from CPU version. It helps to accelerate
   some calculations like square root and arctangent. 
* \param in1 x gradient component
* \param in2 y gradient component
* \param d_sum Output integral histograms
*/
__global__ void buildDescMatKernel(cv::gpu::PtrStepSz<float> in1, cv::gpu::PtrStepSz<float> in2, float *d_sum){

	int block = blockIdx.x * blockDim.x;
	int j = block + threadIdx.x;

	float magnitude0 = sqrt(in1[j]*in1[j]+in2[j]*in2[j]);
	float magnitude1 = magnitude0;
	float orientation = atan2(in2[j], in1[j])* 180/3.141592 +360.0;

    int bin0, bin1;
	
	if(orientation > parameters_kinem[0])
		orientation -= parameters_kinem[0];

	float fbin = orientation/ parameters_kinem[2];
	bin0 = floor(fbin);

	float weight0 = 1 - (fbin - bin0);
	float weight1 = 1 - weight0;

	bin0 %= (int)parameters_kinem[1];
	bin1 = (bin0+1)% (int)parameters_kinem[1];


	bin0 = bin0 + (8*j);
	bin1 = bin1 + (8*j);

	magnitude0 *= weight0;
	magnitude1 *= weight1;

	d_sum[bin0] = magnitude0;
	d_sum[bin1] = magnitude1;
}


/**
* \fn void buildDescMatKernelv2(cv::gpu::PtrStepSz<float> in1, cv::gpu::PtrStepSz<float> in2, float *d_sum)
* \brief This kernel alludes the main loop of BuildDescMat method from CPU version. It helps to accelerate
   some calculations like square root and arctangent. 
* \param in1 x gradient component
* \param in2 y gradient component
* \param d_sum Output integral histograms
*
*/
__global__ void buildDescMatKernelv2(cv::gpu::PtrStepSz<float> in1, cv::gpu::PtrStepSz<float> in2, float *d_sum){
	
	int block = blockIdx.x * blockDim.x;
	int j = block + threadIdx.x;

	 float magnitude0 = sqrt(in1[j]*in1[j]+in2[j]*in2[j]);
	 float magnitude1 = magnitude0;
	 float orientation = atan2(in2[j], in1[j])* 180/3.141592 +360.0;

	 int bin0, bin1;
	
	 bin0 = parameters_kinem[1]; 
	 magnitude0 = 1.0;
	 bin1 = 0;
	 magnitude1 = 0;

	d_sum[bin0] = magnitude0;
	d_sum[bin1] = magnitude1;
}

void mergeSort(float *d_src_key, cudaStream_t stream, int n_blocks, int n_threads, int shared_memory,float* d_element_nine)
{
	mergeSortSharedKernel<<<n_blocks, n_threads, shared_memory*sizeof(float),stream>>>(d_src_key,d_element_nine, shared_memory);
}


void scanKernel(float *d_idata, cudaStream_t stream,int width, dim3 dims, float *d_odata){

	scan<<<dims,width,sizeof(float)*width,stream>>>(d_odata, d_idata);
}


void kinemCompMethod(int height, int width, cv::gpu::GpuMat x_dx,cv::gpu::GpuMat x_dy, cv::gpu::GpuMat y_dx,cv::gpu::GpuMat y_dy, 
					 cudaStream_t stream, cv::gpu::GpuMat div_img, cv::gpu::GpuMat curl_img,cv::gpu::GpuMat shear_img)
{
	kinemCompKernel<<<height,width,0,stream>>>(x_dx, x_dy, y_dx, y_dy, div_img, curl_img, shear_img);
}


void buildDescMatMethod(int height, int width, cv::gpu::GpuMat d_m1, cv::gpu::GpuMat d_m2, cudaStream_t stream, float *d_sum){

	buildDescMatKernel<<<height,width,0,stream>>>(d_m1, d_m2, d_sum);
}


void buildDescMatMethodv2(int height, int width, cv::gpu::GpuMat d_m1, cv::gpu::GpuMat d_m2, cudaStream_t stream, float *d_sum){

	buildDescMatKernelv2<<<height,width,0,stream>>>(d_m1, d_m2, d_sum);
}

void copyToConstantMemory(float *parameters, cudaStream_t stream){

	cudaMemcpyToSymbolAsync(parameters_kinem,parameters, sizeof(float)*3,0,cudaMemcpyHostToDevice, stream);
}