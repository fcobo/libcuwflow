#include "descriptorsgpu.h"

/* \file descriptorsgpu.cpp
*
* This soruce file contains the implementations of some methods which are implemented with CUDA to acelerate some CPU
* functions from the original code. 
*
* \author Fernando Cobo Aguilera
* \date 1/10/2014
*/


//
// Optical Flow Optimization
//

//!Max number of feature points 
int points_in_size = 0;

double oft_time[8] = {0,0,0,0,0,0,0,0};


void GpuDescriptors::initializeFarnebackOpticalFlow(cv::gpu::FarnebackOpticalFlow *flow){

		flow->pyrScale = sqrt(2)/2.0;
		flow->numLevels = 5;
		flow->winSize = 10;
		flow->numIters = 2;
		flow->polyN = 7;
		flow->polySigma = 1.5;
		flow->flags = cv::OPTFLOW_FARNEBACK_GAUSSIAN;
}

void GpuDescriptors::uploadImagesToGpu(IplImagePyramid pyramid, int num_scale, cv::gpu::GpuMat *arr){

	cv::Mat prev_grey_mat;
	IplImage *prev_grey_temp = 0;
	std::size_t temp_level;

	for(int i=0; i<num_scale; i++){

		//The level of the pyramid is selected
		temp_level = i;

		prev_grey_temp = cvCloneImage(pyramid.getImage(temp_level));
		prev_grey_mat = cv::cvarrToMat(prev_grey_temp); //From IplImage to Mat
		arr[i].upload(prev_grey_mat); //Upload to gpu memory
	}
}

cv::Mat GpuDescriptors::opticalFlowGpu(cv::Mat data, int frame_num, int scale,struct GpuMemoryOpticalFlow *memory){

	cv::Mat flow_mat;
	//This conditional is the key to upload the data in the right place. The conditional uses
	// framenum to check whether the last uploaded frame is in either dataY or dataX
	 if(frame_num % 2 == 1){
					
			memory->d_data_y[scale].upload(data);	
			//The last image uploaded is always the second parameter of Farneback optical flow
			//This last image will be used in the next iteration as first parameter
			memory->optical_flow(memory->d_data_x[scale],memory->d_data_y[scale],memory->d_optical_flow_x,memory->d_optical_flow_y);

	 }
	else{ 

			memory->d_data_x[scale].upload(data);				  
			memory->optical_flow(memory->d_data_y[scale],memory->d_data_x[scale],memory->d_optical_flow_x,memory->d_optical_flow_y);					
	}

	//The optical flow is downloaded to cpu memory
	memory->d_optical_flow_x.download(memory->h_optical_flow[scale][0]);
	memory->d_optical_flow_y.download(memory->h_optical_flow[scale][1]);

	//The optical flow is merged in a single Mat
	cv::merge(memory->h_optical_flow[scale],flow_mat);

	return flow_mat;
}





//
// Optical Flow Tracker Optimization
//

void GpuDescriptors::opticalFlowTrackerGpuAsync(IplImage* flow,std::vector<CvPoint2D32f>& points_in, struct GpuMemoryOpticalFlowTracker memory){

	int width = flow->width;
	int height = flow->height;
	int index =0;

	for(int i = 0; i < (signed)points_in.size(); i++){

		CvPoint2D32f point_in = points_in[i];

		int x = cvFloor(point_in.x);
		int y = cvFloor(point_in.y);

		for(int m = x-1; m <= x+1; m++){
		 for(int n = y-1; n <= y+1; n++){

			int p = std::min<int>(std::max<int>(m, 0), width-1);
			int q = std::min<int>(std::max<int>(n, 0), height-1);
			const float* f = (const float*)(flow->imageData + flow->widthStep*q);

			//The first eight numbers are saved in an array
			if(!(m==x+1 && n==y+1)){
				memory.h_arr_x[index]=f[2*p];
				memory.h_arr_y[index]=f[2*p+1];
				index++;	
			}
			else{ //The ninth element is saved in a different array
				memory.h_middle_x[i] = f[2*p];
				memory.h_middle_y[i] = f[2*p+1];
			}
		 }
		}
	}


	int nBlocks;
	int nThreads;
	int sharedMemory;

	//Calculate the number of blocks,threads and shared memory for the kernel 
	//depending on the total number of input interest point positions
	if(points_in.size()*8 <= 2048){ //One block is enough

		nBlocks = 1;
		nThreads = points_in.size()*8/2;
		sharedMemory = points_in.size()*8;
	}
	else{ //More than one block will be needed

		nBlocks = (int)points_in.size()*8/2048;
		nBlocks++;
		nThreads = 1024;
		sharedMemory = 2048;
	}

	//Uploading arrays to gpu memory
	if(cudaSuccess != cudaMemcpyAsync(memory.d_arr, memory.h_arr_x, sizeof(float) * points_in.size()*8, cudaMemcpyHostToDevice,memory.stream)){cout << "Error cudamemcpy" << endl;exit(EXIT_FAILURE);}
	if(cudaSuccess != cudaMemcpyAsync(memory.d_middle, memory.h_middle_x, sizeof(float) * points_in.size(), cudaMemcpyHostToDevice,memory.stream)){cout << "Error cudamemcpy" << endl;exit(EXIT_FAILURE);}
    //The kernel is executed
	mergeSort(memory.d_arr,memory.stream,nBlocks, nThreads, sharedMemory, memory.d_middle);
	//Downloading the results to the ninth-elements array
	if(cudaSuccess != cudaMemcpyAsync(memory.h_middle_x,memory.d_middle,sizeof(float) * points_in.size(),cudaMemcpyDeviceToHost,memory.stream)){cout << "Error cudamemcpy" << endl;exit(EXIT_FAILURE);}

	//Uploading arrays to gpu memory
	if(cudaSuccess != cudaMemcpyAsync(memory.d_arr, memory.h_arr_y, sizeof(float) * points_in.size()*8, cudaMemcpyHostToDevice,memory.stream)){cout << "Error cudamemcpy" << endl;exit(EXIT_FAILURE);}
	if(cudaSuccess != cudaMemcpyAsync(memory.d_middle, memory.h_middle_y, sizeof(float) * points_in.size(), cudaMemcpyHostToDevice,memory.stream)){cout << "Error cudamemcpy" << endl;exit(EXIT_FAILURE);}
	 //The kernel is executed
	mergeSort(memory.d_arr,memory.stream,nBlocks, nThreads, sharedMemory, memory.d_middle);
	//Downloading the results to the ninth-elements array
	if(cudaSuccess != cudaMemcpyAsync(memory.h_middle_y,memory.d_middle,sizeof(float) * points_in.size(),cudaMemcpyDeviceToHost, memory.stream)){cout << "Error cudamemcpy" << endl;exit(EXIT_FAILURE);}
}


void GpuDescriptors::opticalFlowTrackerGpuSync(IplImage* flow, std::vector<CvPoint2D32f>& points_in, std::vector<CvPoint2D32f>& points_out,std::vector<int>& status,struct GpuMemoryOpticalFlowTracker *memory){

	int width = flow->width;
	int height = flow->height;
	int index =0;

	//Memory is allocated if the total number of feature points is bigger than the current maximum value
	//Ex: if the first frame contained 1000 feature points and the second one contains 500, it's not needed
	//to allocate memory
	if((unsigned int)points_in_size < points_in.size()){
		points_in_size = points_in.size();

		memory->h_arr_x = (float*) malloc(sizeof(float)*points_in.size()*8);
		memory->h_arr_y = (float*) malloc(sizeof(float)*points_in.size()*8);
		memory->h_middle_x = (float*) malloc(sizeof(float)*points_in.size());
		memory->h_middle_y = (float*) malloc(sizeof(float)*points_in.size());

	if(cudaSuccess != cudaMalloc((void**)&memory->d_arr,         sizeof(float) * points_in.size()*8)){cout << "Error cudaMalloc" << endl;exit(EXIT_FAILURE);}
	if(cudaSuccess != cudaMalloc((void**)&memory->d_middle, sizeof(float) * points_in.size())){cout << "Error cudaMalloc" << endl;exit(EXIT_FAILURE);}
	}


	for(int i = 0; i < (signed)points_in.size(); i++){

		CvPoint2D32f point_in = points_in[i];

		int x = cvFloor(point_in.x);
		int y = cvFloor(point_in.y);

		for(int m = x-1; m <= x+1; m++){
		 for(int n = y-1; n <= y+1; n++){

			int p = std::min<int>(std::max<int>(m, 0), width-1);
			int q = std::min<int>(std::max<int>(n, 0), height-1);
			const float* f = (const float*)(flow->imageData + flow->widthStep*q);
			
			//The first eight numbers are saved in an array
			if(!(m==x+1 && n==y+1)){
				memory->h_arr_x[index] = f[2*p]; 
				memory->h_arr_y[index] = f[2*p+1]; 
				index++;	
			}
			else{ //The ninth element is saved in a different array
				memory->h_middle_x[i] = f[2*p];
				memory->h_middle_y[i] = f[2*p+1];
			}
		 }
		}
	}


	int nBlocks;
	int nThreads;
	int sharedMemory;

	//Calculate the number of blocks,threads and shared memory for the kernel 
	//depending on the total number of input interest point positions
	if(points_in.size()*8 <= 2048){ //One block is enough

		nBlocks = 1;
		nThreads = points_in.size()*8/2;
		sharedMemory = points_in.size()*8;
	}
	else{ //More than one block will be needed

		nBlocks = (int)points_in.size()*8/2048;
		nBlocks++;
		nThreads = 1024;
		sharedMemory = 2048;
	}

	//Uploading arrays to gpu memory
    if(cudaSuccess != cudaMemcpy(memory->d_arr, memory->h_arr_x, sizeof(float) * points_in.size()*8, cudaMemcpyHostToDevice)){cout << "Error cudamemcpy" << endl;	exit(EXIT_FAILURE);}
	if(cudaSuccess != cudaMemcpy(memory->d_middle, memory->h_middle_x, sizeof(float) * points_in.size(), cudaMemcpyHostToDevice)){cout << "Error cudamemcpy" << endl;exit(EXIT_FAILURE);}
	//The kernel is executed
	clock_t start,end;
	start = clock();
	 mergeSort(memory->d_arr,NULL,nBlocks, nThreads, sharedMemory, memory->d_middle);
	 end = clock();
	 oft_time[0] += difftime(start,end);
	//Downloading the results to the ninth-elements array
	 if(cudaSuccess != cudaMemcpy(memory->h_middle_x,memory->d_middle,sizeof(float) * points_in.size(),cudaMemcpyDeviceToHost)){cout << "Error cudamemcpy" << endl;exit(EXIT_FAILURE);}

	//Uploading arrays to gpu memory
	 if(cudaSuccess != cudaMemcpy(memory->d_arr, memory->h_arr_y, sizeof(float) * points_in.size()*8, cudaMemcpyHostToDevice)){cout << "Error cudamemcpy" << endl;exit(EXIT_FAILURE);}
	 if(cudaSuccess != cudaMemcpy(memory->d_middle, memory->h_middle_y, sizeof(float) * points_in.size(), cudaMemcpyHostToDevice)){cout << "Error cudamemcpy" << endl;exit(EXIT_FAILURE);}
	 //The kernel is executed
	 start = clock();
	 mergeSort(memory->d_arr,NULL,nBlocks, nThreads, sharedMemory,memory->d_middle);
	 end = clock();
	 oft_time[0] += difftime(start,end);
	//Downloading the results to the ninth-elements array
	 if(cudaSuccess != cudaMemcpy(memory->h_middle_y,memory->d_middle,sizeof(float) * points_in.size(),cudaMemcpyDeviceToHost)){cout << "Error cudamemcpy" << endl;exit(EXIT_FAILURE);}

	//The output points and the status are calculated
    CvPoint2D32f point_out;
	for(int i=0; i<(signed)points_in.size(); i++){

		point_out.x = points_in[i].x + memory->h_middle_x[i];
		point_out.y = points_in[i].y + memory->h_middle_y[i];

		points_out[i] = point_out;

		if(point_out.x > 0 && point_out.x < width && point_out.y > 0 && point_out.y < height)
		     status[i] = 1;
	     else
		     status[i] = -1;
	}
}

void GpuDescriptors::freeMemoryOpricalFlowTrackerGpu(struct GpuMemoryOpticalFlowTracker memory){
	

	if(memory.h_arr_x != NULL)
	 free(memory.h_arr_x);

	if(memory.h_arr_y != NULL)
	free(memory.h_arr_y);

	if(memory.h_middle_x != NULL)
	free(memory.h_middle_x);

	if(memory.h_middle_y != NULL)
	free(memory.h_middle_y);

	if(memory.d_arr != NULL)
	 if(cudaSuccess != cudaFree(memory.d_arr)){cout << "Error cudaFree" << endl;exit(EXIT_FAILURE);}

    if(memory.d_middle != NULL)
	 if(cudaSuccess != cudaFree(memory.d_middle)){cout << "Error cudaFree" << endl;exit(EXIT_FAILURE);}
}



//
// Integral histograms Optimization
//

void GpuDescriptors::kinemCompGpuSync(std::vector<cv::Mat> vec, cudaStream_t stream, int n_bins, float *parameters,
				  cv::gpu::GpuMat d_x_dx, cv::gpu::GpuMat d_x_dy, cv::gpu::GpuMat d_y_dx,  cv::gpu::GpuMat d_y_dy,
				  cv::gpu::CudaMem h_x_dx,cv::gpu::CudaMem h_x_dy,cv::gpu::CudaMem h_y_dx,cv::gpu::CudaMem h_y_dy,
				  float * desc_mat_curldiv, float *desc_mat_shear_c, float *desc_mat_shear_d){

	int height = vec[0].rows;
	int width = vec[0].cols;

	float *d_sum;

	dim3 dims,dims2;
	dims.x = n_bins;
	dims.y = height;

	dims2.x = width*n_bins;


	vec[0] = vec[0] * 100;
	vec[1] = vec[1] * 100;

	IplImage ipl1,ipl2;

	ipl1 = vec[0];
	ipl2 = vec[1];

	IplImage f1 = h_x_dx.createMatHeader();
	IplImage f2 = h_x_dy.createMatHeader();
	IplImage f3 = h_y_dx.createMatHeader();
	IplImage f4 = h_y_dy.createMatHeader();


	cvSobel(&ipl1,&f1,1,0,1);
	cvSobel(&ipl1,&f2,0,1,1);
	cvSobel(&ipl2,&f3,1,0,1);
	cvSobel(&ipl2,&f4,0,1,1);


	cv::gpu::GpuMat shearImg(cvSize(width,height), CV_32F);
	cv::gpu::GpuMat curlImg(cvSize(width,height), CV_32F);
	cv::gpu::GpuMat divImg(cvSize(width,height), CV_32F);

	if(cudaSuccess != cudaMalloc((void **)&d_sum,sizeof(float)*n_bins*height*width)){cout << "Error cudaMalloc" << endl;exit(EXIT_FAILURE);}



		if(cudaSuccess != cudaMemcpyAsync(d_x_dx.data,h_x_dx.data,width*height*sizeof(float),cudaMemcpyHostToDevice,stream)){cout << "Error cudaMemcpyAsync" << endl;exit(EXIT_FAILURE);}
		if(cudaSuccess != cudaMemcpyAsync(d_x_dy.data,h_x_dy.data,width*height*sizeof(float),cudaMemcpyHostToDevice,stream)){cout << "Error cudaMemcpyAsync" << endl;exit(EXIT_FAILURE);}
		if(cudaSuccess != cudaMemcpyAsync(d_y_dx.data,h_y_dx.data,width*height*sizeof(float),cudaMemcpyHostToDevice,stream)){cout << "Error cudaMemcpyAsync" << endl;exit(EXIT_FAILURE);}
		if(cudaSuccess != cudaMemcpyAsync(d_y_dy.data,h_y_dy.data,width*height*sizeof(float),cudaMemcpyHostToDevice,stream)){cout << "Error cudaMemcpyAsync" << endl;exit(EXIT_FAILURE);}

	
		kinemCompMethod(height,width, d_x_dx, d_x_dy, d_y_dx, d_y_dy, stream,divImg, curlImg, shearImg);


		//The auxiliar array is set to 0 and the parameteres are uploaded to constant gpu memory
		if(cudaSuccess != cudaMemsetAsync(d_sum,0,sizeof(float)*n_bins*height*width,stream)){cout << "Error cudaMemsetAsync" << endl;exit(EXIT_FAILURE);}
		copyToConstantMemory(parameters, stream);
		
			//The integral histogram is computed, except the scan or prefix sum
			buildDescMatMethod(height, width,divImg, curlImg, stream,  d_sum);

		//The results are downloaded to the auxiliar array		
			if(cudaSuccess != cudaMemcpyAsync(desc_mat_curldiv,d_sum,sizeof(float)*n_bins*height*width,cudaMemcpyDeviceToHost,stream)){cout << "Error cudaMemcpyAsync" << endl;exit(EXIT_FAILURE);} 

		//Even though the last three calls are asynchronous, we wait until all the operations in the stream have finished
			if(cudaSuccess != cudaStreamSynchronize(stream)){cout << "Error cudaStreamSynchronize" << endl;exit(EXIT_FAILURE);}	


		//Instead of calculating the scan or prefix sum, the auxiliar array is set to 0 again and 
		//the parameters are uploaded to constant gpu memory
		if(cudaSuccess != cudaMemsetAsync(d_sum,0,sizeof(float)*n_bins*height*width,stream)){cout << "Error cudaMemsetAsync" << endl;exit(EXIT_FAILURE);} 
		copyToConstantMemory((parameters+3), stream);
			
		//The integral histogram is computed again, except the scan or prefix sum
			buildDescMatMethod(height, width, curlImg, shearImg, stream, d_sum);

			//As the last three calls have been asynchronous, the scan of the first integral histogram is calculated in the cpu
			//Therefore, both cpu and gpu are working at the same time. If this scan operation had been called after the first cudaStreamSynchronize,
			//the gpu would have waited for the cpu to finish
			for(int i=0; i<height; i++)
				for(int m = 8; m < width*n_bins; m++)
					 desc_mat_curldiv[m + (i*n_bins*width)] += desc_mat_curldiv[m + (i*n_bins*width)-n_bins];
				
			for(int i=1; i<height; i++)
				for(int m=0; m<width*n_bins; m++)
				 desc_mat_curldiv[m +(i*width*n_bins)] +=  desc_mat_curldiv[m +(i*width*n_bins) - (width*n_bins)];


		if(cudaSuccess != cudaMemcpyAsync(desc_mat_shear_c,d_sum,sizeof(float)*8*height*width,cudaMemcpyDeviceToHost,stream)){cout << "Error cudaMemcpyAsync" << endl;exit(EXIT_FAILURE);} 
		if(cudaSuccess != cudaStreamSynchronize(stream)){cout << "Error cudaStreamSynchronize" << endl;exit(EXIT_FAILURE);}	

		//The same process is repeated with the rest of integral histograms


		if(cudaSuccess != cudaMemsetAsync(d_sum,0,sizeof(float)*8*height*width,stream)){cout << "Error cudaMemsetAsync" << endl;exit(EXIT_FAILURE);} 

			buildDescMatMethod(height, width,divImg, shearImg,  stream, d_sum);

			
			for(int i=0; i<height; i++)
				for(int m = 8; m < width*n_bins; m++)
					 desc_mat_shear_c[m + (i*n_bins*width)] += desc_mat_shear_c[m + (i*n_bins*width)-n_bins];
				
			for(int i=1; i<height; i++)
				for(int m=0; m<width*n_bins; m++)
				 desc_mat_shear_c[m +(i*width*n_bins)] +=  desc_mat_shear_c[m +(i*width*n_bins) - (width*n_bins)];



		if(cudaSuccess != cudaMemcpyAsync(desc_mat_shear_d,d_sum,sizeof(float)*n_bins*height*width,cudaMemcpyDeviceToHost,stream)){cout << "Error cudaMemcpyAsync" << endl;exit(EXIT_FAILURE);} 
		
		
		if(cudaSuccess != cudaStreamSynchronize(stream)){cout << "Error cudaStreamSynchronize" << endl;exit(EXIT_FAILURE);} 

	
			for(int i=0; i<height; i++)
				for(int m = n_bins; m < width*n_bins; m++)
					 desc_mat_shear_d[m + (i*n_bins*width)] += desc_mat_shear_d[m + (i*n_bins*width)-8];
				
			for(int i=1; i<height; i++)
				for(int m=0; m<width*n_bins; m++)
				 desc_mat_shear_d[m +(i*width*n_bins)] +=  desc_mat_shear_d[m +(i*width*n_bins) - (width*n_bins)];

	    //Release memory
		if(cudaSuccess != cudaStreamSynchronize(stream)){cout << "Error cudaStreamSynchronize" << endl;exit(EXIT_FAILURE);} 
		if(cudaSuccess != cudaFree(d_sum)){cout << "Error cudaFree" << endl;exit(EXIT_FAILURE);} 
}

void GpuDescriptors::kinemCompGpuAsync(std::vector<cv::Mat> vec, cudaStream_t stream, int n_bins, float *parameters, float *d_sum,
				  cv::gpu::GpuMat d_x_dx, cv::gpu::GpuMat d_x_dy, cv::gpu::GpuMat d_y_dx,  cv::gpu::GpuMat d_y_dy,
				  cv::gpu::CudaMem h_x_dx,cv::gpu::CudaMem h_x_dy,cv::gpu::CudaMem h_y_dx,cv::gpu::CudaMem h_y_dy,
				  float * desc_mat_curldiv, float *desc_mat_shear_c, float *desc_mat_shear_d){


	int height = vec[0].rows;
	int width = vec[0].cols;

	dim3 dims,dims2;
	dims.x = n_bins;
	dims.y = height;

	dims2.x = width*n_bins;

	vec[0] = vec[0] * 100;
	vec[1] = vec[1] * 100;

	IplImage ipl1,ipl2;

	ipl1 = vec[0];
	ipl2 = vec[1];

	IplImage f1 = h_x_dx.createMatHeader();
	IplImage f2 = h_x_dy.createMatHeader();
	IplImage f3 = h_y_dx.createMatHeader();
	IplImage f4 = h_y_dy.createMatHeader();
	
	
	cvSobel(&ipl1,&f1,1,0,1);
	cvSobel(&ipl1,&f2,0,1,1);
	cvSobel(&ipl2,&f3,1,0,1);
	cvSobel(&ipl2,&f4,0,1,1);


	cv::gpu::GpuMat shearImg(cvSize(width,height), CV_32F);
	cv::gpu::GpuMat curlImg(cvSize(width,height), CV_32F);
	cv::gpu::GpuMat divImg(cvSize(width,height), CV_32F);

	   //From this point, all the gpu calls are asynchronous, so the cpu will keep working in the main code
	   //The main requierement is that pinned memory is needed to upload and to download the data asynchronously
	    if(cudaSuccess != cudaMemcpyAsync(d_x_dx.data,h_x_dx.data,width*height*sizeof(float),cudaMemcpyHostToDevice,stream)){cout << "Error cudaMemcpyAsync" << endl;exit(EXIT_FAILURE);}
		if(cudaSuccess != cudaMemcpyAsync(d_x_dy.data,h_x_dy.data,width*height*sizeof(float),cudaMemcpyHostToDevice,stream)){cout << "Error cudaMemcpyAsync" << endl;exit(EXIT_FAILURE);}
		if(cudaSuccess != cudaMemcpyAsync(d_y_dx.data,h_y_dx.data,width*height*sizeof(float),cudaMemcpyHostToDevice,stream)){cout << "Error cudaMemcpyAsync" << endl;exit(EXIT_FAILURE);}
		if(cudaSuccess != cudaMemcpyAsync(d_y_dy.data,h_y_dy.data,width*height*sizeof(float),cudaMemcpyHostToDevice,stream)){cout << "Error cudaMemcpyAsync" << endl;exit(EXIT_FAILURE);}


		kinemCompMethod(height,width, d_x_dx, d_x_dy, d_y_dx, d_y_dy, stream, divImg, curlImg, shearImg);

		   //The auxiliar array is set to 0 and the parameteres are uploaded to constant gpu memory
			if(cudaSuccess != cudaMemsetAsync(d_sum,0,sizeof(float)*n_bins*height*width,stream)){cout << "Error cudaMemcpyAsync" << endl;exit(EXIT_FAILURE);}
			copyToConstantMemory(parameters, stream);

			//The integral histogram and the scan or prefix sum are computed in the gpu
			buildDescMatMethod(height, width,divImg, curlImg, stream, d_sum);
			scanKernel(d_sum,stream,width,dims, d_sum);
			scanKernel(d_sum,stream,height,dims2, d_sum);

        //The results are downloaded to the descriptor mat
		if(cudaSuccess != cudaMemcpyAsync(desc_mat_curldiv,d_sum,sizeof(float)*n_bins*height*width,cudaMemcpyDeviceToHost,stream)){cout << "Error cudaMemcpyAsync" << endl;exit(EXIT_FAILURE);}
	    
		//The same process is repeated two more times

			if(cudaSuccess != cudaMemsetAsync(d_sum,0,sizeof(float)*n_bins*height*width,stream)){cout << "Error cudaMemsetAsync" << endl;exit(EXIT_FAILURE);}	
		    copyToConstantMemory((parameters+3), stream);

			buildDescMatMethod(height, width, curlImg, shearImg, stream, d_sum);
			scanKernel(d_sum,stream,width,dims, d_sum);
			scanKernel(d_sum,stream,height,dims2, d_sum);

		if(cudaSuccess != cudaMemcpyAsync(desc_mat_shear_c,d_sum,sizeof(float)*n_bins*height*width,cudaMemcpyDeviceToHost,stream)){cout << "Error cudaMemcpyAsync" << endl;exit(EXIT_FAILURE);}
	

			if(cudaSuccess != cudaMemsetAsync(d_sum,0,sizeof(float)*n_bins*height*width,stream)){cout << "Error cudaMemsetAsync" << endl;exit(EXIT_FAILURE);}

			buildDescMatMethod(height, width,divImg, shearImg, stream, d_sum);
			scanKernel(d_sum,stream,width,dims,d_sum);
			scanKernel(d_sum,stream,height,dims2,d_sum);
			
		if(cudaSuccess != cudaMemcpyAsync(desc_mat_shear_d,d_sum,sizeof(float)*n_bins*height*width,cudaMemcpyDeviceToHost,stream)){cout << "Error cudaMemcpyAsync" << endl;exit(EXIT_FAILURE);}
		
}


void GpuDescriptors::hogMbhCompGpuAsync(std::vector<cv::Mat> vec, IplImage *img, cudaStream_t stream, int n_bins, float *parameters, float *d_sum,
			   cv::gpu::GpuMat d_x_dx, cv::gpu::GpuMat d_x_dy, cv::gpu::GpuMat d_y_dx,  cv::gpu::GpuMat d_y_dy,
			   cv::gpu::GpuMat d_hog_x, cv::gpu::GpuMat d_hog_y,
			   cv::gpu::CudaMem h_x_dx,cv::gpu::CudaMem h_x_dy,cv::gpu::CudaMem h_y_dx,cv::gpu::CudaMem h_y_dy,
			   cv::gpu::CudaMem h_hog_x,cv::gpu::CudaMem h_hog_y,
			   float * desc_mat_x, float *desc_mat_y, float *desc_mat ){

	int height = vec[0].rows;
	int width = vec[0].cols;

	vec[0] = vec[0] * 100;
	vec[1] = vec[1] * 100;

	dim3 dims,dims2;
	dims.x = n_bins;
	dims.y = height;

	dims2.x = width*n_bins;

	IplImage ipl1,ipl2;

	ipl1 = vec[0];
	ipl2 = vec[1];

	IplImage f1 = h_x_dx.createMatHeader();
	IplImage f2 = h_x_dy.createMatHeader();
	IplImage f3 = h_y_dx.createMatHeader();
	IplImage f4 = h_y_dy.createMatHeader();
	IplImage f5 = h_hog_x.createMatHeader();
	IplImage f6 = h_hog_y.createMatHeader();
	
	
	cvSobel(&ipl1,&f1,1,0,1);
	cvSobel(&ipl1,&f2,0,1,1);
	cvSobel(&ipl2,&f3,1,0,1);
	cvSobel(&ipl2,&f4,0,1,1);

	cvSobel(img,&f5 ,1,0,1);
    cvSobel(img,&f6 ,0,1,1);

	   //From this point, all the gpu calls are asynchronous, so the cpu will keep working in the main code
	   //The main requierement is that pinned memory is needed to upload and to download the data asynchronously
		if(cudaSuccess != cudaMemcpyAsync(d_x_dx.data,h_x_dx.data,width*height*sizeof(float),cudaMemcpyHostToDevice,stream)){cout << "Error cudaMemcpyAsync" << endl;exit(EXIT_FAILURE);}
		if(cudaSuccess != cudaMemcpyAsync(d_x_dy.data,h_x_dy.data,width*height*sizeof(float),cudaMemcpyHostToDevice,stream)){cout << "Error cudaMemcpyAsync" << endl;exit(EXIT_FAILURE);}
		if(cudaSuccess != cudaMemcpyAsync(d_y_dx.data,h_y_dx.data,width*height*sizeof(float),cudaMemcpyHostToDevice,stream)){cout << "Error cudaMemcpyAsync" << endl;exit(EXIT_FAILURE);}
		if(cudaSuccess != cudaMemcpyAsync(d_y_dy.data,h_y_dy.data,width*height*sizeof(float),cudaMemcpyHostToDevice,stream)){cout << "Error cudaMemcpyAsync" << endl;exit(EXIT_FAILURE);}


		if(cudaSuccess != cudaMemcpyAsync(d_hog_x.data,h_hog_x.data,width*height*sizeof(float),cudaMemcpyHostToDevice,stream)){cout << "Error cudaMemcpyAsync" << endl;exit(EXIT_FAILURE);}
		if(cudaSuccess != cudaMemcpyAsync(d_hog_y.data,h_hog_y.data,width*height*sizeof(float),cudaMemcpyHostToDevice,stream)){cout << "Error cudaMemcpyAsync" << endl;exit(EXIT_FAILURE);}

		//The auxiliar array is set to 0 and the parameteres are uploaded to constant gpu memory
		if(cudaSuccess != cudaMemsetAsync(d_sum,0,sizeof(float)*n_bins*height*width,stream)){cout << "Error cudaMemsetAsync" << endl;exit(EXIT_FAILURE);}
		copyToConstantMemory(parameters, stream);

		//The integral histogram and the scan or prefix sum are computed in the gpu
		buildDescMatMethod(height, width,d_x_dx, d_x_dy, stream, d_sum);
		scanKernel(d_sum,stream,width,dims,d_sum);
		scanKernel(d_sum,stream,height,dims2,d_sum);
		
		//The results are downloaded to the descriptor mat
		if(cudaSuccess != cudaMemcpyAsync(desc_mat_x,d_sum,sizeof(float)*8*height*width,cudaMemcpyDeviceToHost,stream)){cout << "Error cudaMemcpyAsync" << endl;exit(EXIT_FAILURE);}

		//The same process is repeated two more times

		if(cudaSuccess != cudaMemsetAsync(d_sum,0,sizeof(float)*n_bins*height*width,stream)){cout << "Error cudaMemsetAsync" << endl;exit(EXIT_FAILURE);}
		buildDescMatMethod(height, width,d_y_dx, d_y_dy, stream, d_sum);
		scanKernel(d_sum,stream,width,dims,d_sum);
		scanKernel(d_sum,stream,height,dims2,d_sum);

		if(cudaSuccess != cudaMemcpyAsync(desc_mat_y,d_sum,sizeof(float)*n_bins*height*width,cudaMemcpyDeviceToHost,stream)){cout << "Error cudaMemcpyAsync" << endl;exit(EXIT_FAILURE);}

		if(cudaSuccess != cudaMemsetAsync(d_sum,0,sizeof(float)*n_bins*height*width,stream)){cout << "Error cudaMemsetAsync" << endl;exit(EXIT_FAILURE);}
		buildDescMatMethod(height, width,d_hog_x, d_hog_y, stream, d_sum);
		scanKernel(d_sum,stream,width,dims,d_sum);
		scanKernel(d_sum,stream,height,dims2,d_sum);

		if(cudaSuccess != cudaMemcpyAsync(desc_mat,d_sum,sizeof(float)*n_bins*height*width,cudaMemcpyDeviceToHost,stream)){cout << "Error cudaMemcpyAsync" << endl;exit(EXIT_FAILURE);}
}





//
// Auxiliar functions
//

cv::gpu::GpuMat * GpuDescriptors::createPointers(int numScale){

	cv::gpu::GpuMat *arr;
	arr = new cv::gpu::GpuMat[numScale];

	return arr;
}