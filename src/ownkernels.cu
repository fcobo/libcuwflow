/**
* \file ownkernels.cu
*  This file contains the implementation of kernels which will be used to acelerate the algorithm.
*
* @author Fernando Cobo Aguilera
* @date 30/11/2014
*/

#include "ownkernels.cuh"


/**
* CUDA constant memory variable
*
* This variable is needed when the user runs BuildDescMatKernel or
* BuildDescMatKernelv2. They both need a set of parameters, thus,
* the constant memory is ideal in these cases.
*/
__constant__ float parameters_kinem[3];


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