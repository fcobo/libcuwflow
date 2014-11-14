/**
* \file initialize.h
*
* This header file contains some initialization functions as well as all the functions that
* are used to check errors in the command line parameters
*/


#ifndef CUWFLOW_INITIALIZE_H_
#define CUWFLOW_INITIALIZE_H_

#include "densetrack.h"

/**
* This function initializes a tracker
* \param tracker Tracker that will be initialized
* \param track_length Tracker length
* \param init_gap Initial gap
*/
void InitTrackerInfo(TrackerInfo* tracker, int track_length, int init_gap)
{
	tracker->trackLength = track_length;
	tracker->initGap = init_gap;
}

/**
* This function initializes a descriptor instance
* \param height Total number of rows
* \param width Total number of cols
* \param nBins Total number of bins
*/
DescMat* InitDescMat(int height, int width, int nBins)
{
	DescMat* descMat = (DescMat*)malloc(sizeof(DescMat));
	descMat->height = height;
	descMat->width = width;
	descMat->nBins = nBins;
	descMat->desc = (float*)malloc(height*width*nBins*sizeof(float));
	memset( descMat->desc, 0, height*width*nBins*sizeof(float));
	return descMat;
}



/**
* This function initializes a descriptor info instance 
* \param descInfo Descriptor info instance
* \param nBins nBins value
* \param flag Flag value
* \param orientation Orientation value
* \param size size value
* \param nxy_cell nxy_cell value
* \param nt_cell nt_cell value
*/
void InitDescInfo(DescInfo* descInfo, int nBins, int flag, int orientation, int size, int nxy_cell, int nt_cell)
{
	descInfo->nBins = nBins;
	descInfo->fullOrientation = orientation;
	descInfo->norm = 2;
	descInfo->threshold = min_flow;
	descInfo->flagThre = flag;
	descInfo->nxCells = nxy_cell;
	descInfo->nyCells = nxy_cell;
	descInfo->ntCells = nt_cell;
	descInfo->dim = descInfo->nBins*descInfo->nxCells*descInfo->nyCells;
	descInfo->blockHeight = size;
	descInfo->blockWidth = size;
}



/**
* This function release descriptor memory
* \param descMat Descriptor
*/
void ReleDescMat( DescMat* descMat)
{
	free(descMat->desc);
	free(descMat);
}


/**
* This funcion shows the user all the options that he/she could use as command line parameters
*/
void usage()
{
	fprintf(stderr, "Extract dense trajectories from a video\n\n");
	fprintf(stderr, "\nRequired:\n");
	fprintf(stderr, "[--video <videoname>] # Frames source \n");
	fprintf(stderr, "\n GPU options:\n");
	fprintf(stderr, "[--ofgpu]    # Using CUDA to calculate the Optical Flow\n");
	fprintf(stderr, "[--oftgpu]   # Using CUDA to calculate Median filters\n");
	fprintf(stderr, "[--ihgpu]    # Using CUDA to calculate integral histograms\n");
	fprintf(stderr, "\n Descriptors:\n");
	fprintf(stderr, "[--hog]      # Save HOG descriptors\n");	
	fprintf(stderr, "[--hof]      # Save HOF descriptors\n");	
	fprintf(stderr, "[--mbh]      # Save MBH descriptors\n");	
	fprintf(stderr, "[--curldiv]     # Save curldiv descriptors\n");
	fprintf(stderr, "[--shearC]      # Save shearC descriptors\n");
	fprintf(stderr, "[--shearD]      # Save shearD descriptors\n");	
	fprintf(stderr, "\n Others:\n");
	fprintf(stderr, "[--show]                  # Visualize the trajectories\n");	
	fprintf(stderr, "[--start <start frame>]   # The start frame to compute feature (default: 0 frame)\n");	
	fprintf(stderr, "[-end <end frame>]        # The end frame for feature computing (default: last frame)\n");
	fprintf(stderr, "[--C <DCS or others>]     # 0: traj hog hof/w-hof mbh | 1: traj curldiv shearC shearD (default 1)\n");
	fprintf(stderr, "[--L <trajectory length>] # The length of the trajectory (default: L=15 frames)\n");
	fprintf(stderr, "[--W <sampling stride>]   # The stride for dense sampling feature points (default: W=5 pixels)\n");
	fprintf(stderr, "[--N <neighborhood size>] # The neighborhood size for computing the descriptor (default: N=32 pixels)\n");
	fprintf(stderr, "[--s <spatial cells>]     # The number of cells in the nxy axis (default: nxy=2 cells)\n");
	fprintf(stderr, "[--t <temporal cells>]    # The number of cells in the nt axis (default: nt=3 cells)\n");
	fprintf(stderr, "[--help]                  # Display this message and exit\n");

	//These two options are disabled for this version
	//fprintf(stderr, "  -T [wFlow]		     0: track optical flow for traj, hog, hof, mbh | 1: track wflow for w-traj, w-hog, w-hof (hof+hofwflow), w-mbh (default 0)\n");
	//fprintf(stderr, "  -M [model files] 	     Set 1 to write Affine flow model files using Motion2D (default 0)\n");
}

/**
* This function checks if a string contains a integer or not.
* \param src String that will be checked
*/
void checkInteger(char * src){

	for(int i=0; i< (signed)strlen(src); i++){

		if(!isdigit(src[i])){
			cerr<<"Error in parameters"<<endl;
			usage();
			cout << "Press to end...";
			getchar();
			exit(EXIT_FAILURE);
		}
	}
}

/**
* This function checks if the total number of parameters from command line is equal to a number
* \param argc Total number of parameters
* \param i Integer that will be compared
*/
void checkErrors(int argc, int i){

	if(argc == i+1){
		cerr<<"Error in parameters"<<endl;
		usage();
		cout << "Press to end...";
		getchar();
		exit(EXIT_FAILURE);
	}
}

/**
* Thsi function checks if the user has introduced the right parameters in the command line
* \param argc Number of parameteres
* \param argv Array with the parameters
*/
void checkArgs(int argc, char** argv){

	//if there are no parameters
	if(argc == 1){
		usage();
		cout <<"Press to exit:"; 
		getchar(); 
		exit(EXIT_SUCCESS);
	}
	else{
	  //The first parameter must be the video
	  if(string(argv[1]) == "--video"){
			if(argc == 2){
				usage();
				cout << "Press to end...";
				getchar();
				exit(EXIT_FAILURE);
			}
		  video_name = std::string(argv[2]);
	  }
	  else{
		   usage();
		   cout << "Press to end...";
		   getchar();
		   exit(EXIT_FAILURE);
	  }
	}

	//The other parameters are checked
		for(int i=3; i< argc; i++){
			if(std::string(argv[i]) == "--help"){usage(); cout <<"Press to exit:"; getchar(); exit(EXIT_SUCCESS);}
			else if(std::string(argv[i]) == "--start"){checkErrors(argc,i); checkInteger(argv[i+1]); start_frame =  atoi(argv[i+1]);}
			else if(std::string(argv[i]) == "--end"){checkErrors(argc,i); checkInteger(argv[i+1]); end_frame =  atoi(argv[i+1]);}
			else if(std::string(argv[i]) == "--ofgpu")flow_gpu = true;
			else if(std::string(argv[i]) == "--oftgpu")optical_flow_tracker_gpu = true;
			else if(std::string(argv[i]) == "--ihgpu")integral_histograms_gpu = true;
			else if(std::string(argv[i]) == "--show")show_track = 1;
			else if(std::string(argv[i]) == "--L"){checkErrors(argc,i); checkInteger(argv[i+1]); track_length = atoi(argv[i+1]);}
			else if(std::string(argv[i]) == "--W"){checkErrors(argc,i); checkInteger(argv[i+1]); min_distance = atoi(argv[i+1]);}
			else if(std::string(argv[i]) == "--N"){checkErrors(argc,i); checkInteger(argv[i+1]); patch_size = atoi(argv[i+1]);}
			else if(std::string(argv[i]) == "--s"){checkErrors(argc,i); checkInteger(argv[i+1]); nxy_cell = atoi(argv[i+1]);}
			else if(std::string(argv[i]) == "--t"){checkErrors(argc,i); checkInteger(argv[i+1]); nt_cell = atoi(argv[i+1]);}
			else if(string(argv[i]) == "--C"){checkErrors(argc,i); checkInteger(argv[i+1]); feat_comb=atoi(argv[i+1]); 
			      if(feat_comb >1 || feat_comb < 0){usage(); cout << "Press to end...";getchar();exit(EXIT_FAILURE);}}
			else if(string(argv[i]) == "--hog")hogdesc = true;
			else if(string(argv[i]) == "--hof")hofdesc = true;
			else if(string(argv[i]) == "--mbh")mbhdesc = true;
			else if(string(argv[i]) == "--curldiv"){curldivdesc = true;}
			else if(string(argv[i]) == "--shearC") shearCdesc = true;
			else if(string(argv[i]) == "--shearD") shearDdesc = true;
		}
}

/**
* This function calculates the images range of a pyramid instance that should be
* use for the gpu version. The bigger are the images, the bigger is the range of images
* that will be used in the gpu methods.
*
*/
void calculateRangeScale(int width, int height, int scale){

	int w = width;
	int h = height;
	int total = w*h;

	flow_gpu_range = 0;

	for(int i=0; i<scale;i++){

		//The restriction for an image to belong to this range is to have
		//a bigger size than 10000 elements
		if(total >=10000)
			flow_gpu_range++;


		w = w / scale_stride;
		h = h / scale_stride;
		total = w *h;
	}
	//The optical flow range and the integral histogram range are the same
	histograms_gpu_range = flow_gpu_range;

	//The optical flow tracker range is one unit bigger than optical flow range
	optical_flow_tracker_gpu_range = flow_gpu_range +1;
}


#endif //CUWFLOW_INITIALIZE_H_
