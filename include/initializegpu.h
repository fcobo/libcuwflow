#ifndef CUWFLOW_INITIALIZE_GPU_H_
#define CUWFLOW_INITIALIZE_GPU_H_

#include "densetrack.h"

/**
* This function initializes a descriptor instance with pinned memory
* \param height Total number of rows
* \param width Total number of cols
* \param nBins Total number of bins
*/
DescMat* InitDescMatPagedLocked(int height, int width, int nBins)
{
	DescMat* descMat = (DescMat*)malloc(sizeof(DescMat));
	descMat->height = height;
	descMat->width = width;
	descMat->nBins = nBins;
	cudaMallocHost((void **)&descMat->desc,height*width*nBins*sizeof(float));
	memset( descMat->desc, 0, height*width*nBins*sizeof(float));
	return descMat;
}


#endif //CUWFLOW_INITIALIZE_GPU_H_