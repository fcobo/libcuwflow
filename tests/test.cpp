#include "densetrack.h"

#include "descriptors.h"
#include "descriptorsgpu.h"

#include "initialize.h"
#include "initializegpu.h"



int main( int argc, char** argv )
{
	
	//Initialize CUDA graphic card
	cout << "Initializing CUDA graphic card..."<<endl;

	cudaError_t error;

	int ndevices = cv::gpu::getCudaEnabledDeviceCount();
    
    if(ndevices == 0){

        cerr << "No CUDA-capable devices were detected by the installed CUDA driver" << endl;
        cout << "Press to end";
        getchar();
        exit(EXIT_FAILURE);
    }

    if( cudaSuccess != (error = cudaSetDevice(0)) ){

        cerr << "There was a problem during GPU initializaction.";

        switch (error)
        {
        case 10:
            cerr << "The device which has been supplied by the user does not correspond to a valid CUDA device." 
                 << " Try to change cudaSetDevice() with another value."<<endl;
            break;
        default:
            break;
        }
    
        cout << "Press to end";
        getchar();
        exit(EXIT_FAILURE);
    }

	cv::gpu::printShortCudaDeviceInfo(0);

	//The parameters from the command line are checked
	checkArgs(argc,argv);

	int frameNum = 0;
	CvCapture* capture = 0;
	TrackerInfo tracker;
	DescInfo hogInfo;
	DescInfo hofInfo;
	DescInfo mbhInfo;
	DescInfo curldivInfo;
	DescInfo shearCDInfo;

	//To save the descriptors in memory
	ofstream descriptorsFile("descriptors.txt");


	IplImageWrapper image, prev_image, grey, prev_grey;
	IplImageWrapper realVideo;
	IplImagePyramid grey_pyramid, prev_grey_pyramid, eig_pyramid;

	//Initialize descriptors info
	InitTrackerInfo(&tracker, track_length, init_gap);
	InitDescInfo(&hogInfo, 8, 0, 1, patch_size, nxy_cell, nt_cell);
	InitDescInfo(&hofInfo, 9, 1, 1, patch_size, nxy_cell, nt_cell);
	InitDescInfo(&mbhInfo, 8, 0, 1, patch_size, nxy_cell, nt_cell);
	InitDescInfo(&curldivInfo, 8, 0, 1, patch_size, nxy_cell, nt_cell);
	InitDescInfo(&shearCDInfo, 8, 0, 0, patch_size, nxy_cell, nt_cell);

	//int n_feature_points = 0;


	//Check the video
	capture = cvCreateFileCapture(video_name.c_str());
	if(!capture ){ 
		printf( "Could not initialize capturing..\n" );
		usage();
		cout << "Press to end...";
		getchar();
		exit(EXIT_FAILURE);
	}
	else{
	 IplImage *ipl;
  	 ipl = cvQueryFrame(capture);
	 calculateRangeScale(ipl->width, ipl->height, scale_num);
	 capture = cvCreateFileCapture(video_name.c_str());
	}

	//A window is created to show the results
	if( show_track == 1 )
		cvNamedWindow("DenseTrack", 1);
	

	//Pointers that will be used to allocate pinned memory if necessary
	DescMat *descMatPinnedMemory_1;
	DescMat *descMatPinnedMemory_2;
	DescMat *descMatPinnedMemory_3;
	DescMat *descMatPinnedMemory_4;
	DescMat *descMatPinnedMemory_5;
	DescMat *descMatPinnedMemory_6;
	

	std::vector<std::list<Track> > xyScaleTracks;
	int init_counter = 0; // indicate when to detect new feature points

	//Parameters for the gpu version
	struct GpuMemoryOpticalFlow  memoryof;
	struct GpuMemoryIntegralHistogram  memoryih;
	cudaStream_t streamof;

	double *featurepoints;

	featurepoints = (double*)malloc(sizeof(double)*scale_num);

	for(int i=0; i< scale_num; i++)
		featurepoints[i] = 0;
	

	//Initialize gpu memory for the optical flow
	if(flow_gpu){

		GpuDescriptors::initializeFarnebackOpticalFlow(&memoryof.optical_flow);

		memoryof.d_data_x = GpuDescriptors::createPointers(scale_num);
		memoryof.d_data_y = GpuDescriptors::createPointers(scale_num);
		memoryof.h_optical_flow.resize(scale_num);

		for(int i=0; i< scale_num; i++){
		  memoryof.h_optical_flow[i].push_back(cv::Mat());
		  memoryof.h_optical_flow[i].push_back(cv::Mat());
		}
	}

	//Initialize gpu memory for integral histograms
	if(integral_histograms_gpu){

		cudaStreamCreate(&streamof);

		memoryih.d_flow_x_dx = GpuDescriptors::createPointers(scale_num);
		memoryih.d_flow_x_dy = GpuDescriptors::createPointers(scale_num);
		memoryih.d_flow_y_dx = GpuDescriptors::createPointers(scale_num);
		memoryih.d_flow_y_dy = GpuDescriptors::createPointers(scale_num);
		memoryih.h_flow_x_dx = new cv::gpu::CudaMem[scale_num];
		memoryih.h_flow_x_dy = new cv::gpu::CudaMem[scale_num];
		memoryih.h_flow_y_dx = new cv::gpu::CudaMem[scale_num];
		memoryih.h_flow_y_dy = new cv::gpu::CudaMem[scale_num];

		if(!feat_comb){

			memoryih.d_hog_x = GpuDescriptors::createPointers(scale_num);
			memoryih.d_hog_y = GpuDescriptors::createPointers(scale_num);

			memoryih.h_hog_x = new cv::gpu::CudaMem[scale_num];
			memoryih.h_hog_y = new cv::gpu::CudaMem[scale_num];
		}
	}

	//Initialize times
	struct TimesAlgorithm times;
	times.get_descriptors_time = 0;
	times.integral_histograms_time = 0;
	times.optical_flow_time = 0;
	times.optical_flow_tracker_time = 0;
	times.total_time = 0;

	startTime(&times,1);


	struct GpuMemoryOpticalFlowTracker memoryoftracker;

	if(optical_flow_tracker_gpu){
		cudaStreamCreate(&memoryoftracker.stream);
	}
	

	std::vector<vector<CvPoint2D32f> > points_in_vector(scale_num);
	std::vector<vector<CvPoint2D32f> > points_out_vector(scale_num);
	std::vector<vector<int> > status_vector(scale_num);
	IplImage **flow_vector;

	flow_vector = (IplImage **)malloc(scale_num* sizeof(IplImage*));

	while( true ) {

		if(frameNum < start_frame)
			cout << "Frame " <<frameNum << endl;
		else
			if(frameNum >= start_frame && frameNum <=end_frame)
		     cout << "Frame " <<frameNum << " (" << 100*(frameNum-start_frame)/(end_frame-start_frame) << "% completed)" <<endl;
			else
			 cout << "Frame " <<frameNum << endl;

		IplImage* frame = 0;
		int i, c;

		// get a new frame
		frame = cvQueryFrame(capture);


		if(!frame)break;

		 if(show_track == 1 && frameNum < start_frame)
			cvShowImage( "DenseTrack", frame);			
		

		if( frameNum >= start_frame && frameNum <= end_frame ){



		if(!image ){ //First frame

			// initailize all the buffers
			image = IplImageWrapper( cvGetSize(frame), 8, 3 );
			image->origin = frame->origin;
			prev_image= IplImageWrapper( cvGetSize(frame), 8, 3 );
			prev_image->origin = frame->origin;
			grey = IplImageWrapper( cvGetSize(frame), 8, 1 );
			grey_pyramid = IplImagePyramid( cvGetSize(frame), 8, 1, scale_stride );
			prev_grey = IplImageWrapper( cvGetSize(frame), 8, 1 );
			prev_grey_pyramid = IplImagePyramid( cvGetSize(frame), 8, 1, scale_stride );
			eig_pyramid = IplImagePyramid( cvGetSize(frame), 32, 1, scale_stride);

			cvCopy( frame, image, 0 );
			cvCvtColor( image, grey, CV_BGR2GRAY );
			grey_pyramid.rebuild( grey );

			// how many scale we can have
			scale_num = std::min<std::size_t>(scale_num, grey_pyramid.numOfLevels());
			fscales = (float*)cvAlloc(scale_num*sizeof(float));
			xyScaleTracks.resize(scale_num);

			for( int ixyScale = 0; ixyScale < scale_num; ++ixyScale ) {

				std::list<Track>& tracks = xyScaleTracks[ixyScale];
				fscales[ixyScale] = pow(scale_stride, ixyScale);

				// find good features at each scale separately
				IplImage *grey_temp = 0, *eig_temp = 0;
				std::size_t temp_level = (std::size_t)ixyScale;
				grey_temp = cvCloneImage(grey_pyramid.getImage(temp_level));


				eig_temp = cvCloneImage(eig_pyramid.getImage(temp_level));
				std::vector<CvPoint2D32f> points(0);
				Descriptors::cvDenseSample(grey_temp, eig_temp, points, quality, min_distance);

				//Pinned or Page locked memory is allocated in the first iteration of the algorithm
				if(ixyScale == 0){
					if(integral_histograms_gpu)
					descMatPinnedMemory_1 = InitDescMatPagedLocked(grey_temp->height,grey_temp->width,8);
					descMatPinnedMemory_2 = InitDescMatPagedLocked(grey_temp->height,grey_temp->width,8);
					descMatPinnedMemory_3 = InitDescMatPagedLocked(grey_temp->height,grey_temp->width,8);
					descMatPinnedMemory_4 = InitDescMatPagedLocked(grey_temp->height,grey_temp->width,8);
					descMatPinnedMemory_5 = InitDescMatPagedLocked(grey_temp->height,grey_temp->width,8);
					descMatPinnedMemory_6 = InitDescMatPagedLocked(grey_temp->height,grey_temp->width,8);
				}

				//Allocating memory in gpu for integral histograms
				if(integral_histograms_gpu){

					memoryih.d_flow_x_dx[ixyScale].create(grey_temp->height, grey_temp->width, CV_32F);
					memoryih.d_flow_x_dy[ixyScale].create(grey_temp->height, grey_temp->width, CV_32F);
					memoryih.d_flow_y_dx[ixyScale].create(grey_temp->height, grey_temp->width, CV_32F); 
					memoryih.d_flow_y_dy[ixyScale].create(grey_temp->height, grey_temp->width, CV_32F); 
					memoryih.h_flow_x_dx[ixyScale].create(grey_temp->height, grey_temp->width, CV_32F);
					memoryih.h_flow_x_dy[ixyScale].create(grey_temp->height, grey_temp->width, CV_32F);
					memoryih.h_flow_y_dx[ixyScale].create(grey_temp->height, grey_temp->width, CV_32F);
					memoryih.h_flow_y_dy[ixyScale].create(grey_temp->height, grey_temp->width, CV_32F);

					if(!feat_comb){

						 memoryih.d_hog_x[ixyScale].create(grey_temp->height, grey_temp->width, CV_32F);
						 memoryih.d_hog_y[ixyScale].create(grey_temp->height, grey_temp->width, CV_32F);
						 memoryih.h_hog_x[ixyScale].create(grey_temp->height, grey_temp->width, CV_32F);
						 memoryih.h_hog_y[ixyScale].create(grey_temp->height, grey_temp->width, CV_32F);
					}
				}

				// save the feature points
				for( i = 0; i < (signed)points.size(); i++ ) {
					Track track(tracker.trackLength);
					if(feat_comb) {
					    PointDesc point(curldivInfo, shearCDInfo, points[i]);
					    track.addPointDesc(point); }
					else {
						if (w_flow) {
						    PointDesc point(0, hogInfo, hofInfo, mbhInfo, points[i]); 
						    track.addPointDesc(point); }
						else {
						    PointDesc point(hogInfo, hofInfo, mbhInfo, points[i]);
						    track.addPointDesc(point); }
					}
					tracks.push_back(track);
				}

				cvReleaseImage( &grey_temp );
				cvReleaseImage( &eig_temp );
			}

		   //upload images to GPU if flowGpu is true
			if(flow_gpu)
			  GpuDescriptors::uploadImagesToGpu(grey_pyramid, scale_num,memoryof.d_data_x);
		}


		// build the image pyramid for the current frame
		cvCopy( frame, image, 0 );
		cvCvtColor( image, grey, CV_BGR2GRAY );
		grey_pyramid.rebuild(grey);


		/////////////
		//         // 
		//Main loop//
		//         //
		/////////////
		if(frameNum > start_frame){

		init_counter++;



		//////////////////
		//              // 
		//Feature Points//
		//              //
		//////////////////

		 //All the feature points are collected for all the levels 
		for( int ixyScale = 0; ixyScale < scale_num; ++ixyScale ){
	
			std::list<Track> &tracks = xyScaleTracks[ixyScale];
			for (std::list<Track>::iterator iTrack = tracks.begin(); iTrack != tracks.end(); ++iTrack) {
				CvPoint2D32f point = iTrack->pointDescs.back().point;
				points_in_vector[ixyScale].push_back(point);
				//n_feature_points++;
			}

		 int count = points_in_vector[ixyScale].size();
		 points_out_vector[ixyScale].resize(count);
		 status_vector[ixyScale].resize(count);

		 featurepoints[ixyScale] = count;
		}



		///////////////////////////////////////
		//                                   // 
		//Optical Flow + Optical Flow Tracker//
		//                                   //
		///////////////////////////////////////

		//The optical flow and the optical flow tracker are calculated for all the levels
		IplImage **prev_grey_temp_vector = (IplImage **)malloc(sizeof(IplImage *)*scale_num);
		for( int ixyScale = 0; ixyScale < scale_num; ++ixyScale ) {


			IplImage *grey_temp = 0;
			std::size_t temp_level = ixyScale;
			prev_grey_temp_vector[ixyScale] = cvCloneImage(prev_grey_pyramid.getImage(temp_level));
			grey_temp = cvCloneImage(grey_pyramid.getImage(temp_level));

			cv::Mat prev_grey_mat = cv::cvarrToMat(prev_grey_temp_vector[ixyScale]);
			cv::Mat grey_mat = cv::cvarrToMat(grey_temp);

			
			flow_vector[ixyScale] = cvCreateImage(cvGetSize(grey_temp), IPL_DEPTH_32F, 2);
			cv::Mat flow_mat = cv::cvarrToMat(flow_vector[ixyScale]);
			cv::Mat aux;



			//OPTICAL FLOW
			startTime(&times,0); 
			//To calculate the optical flow in the gpu, two requirements are needed
			// 1. flowGpu must be activated
			// 2. The level of the pyramid image must be lower than flowGpuRange value
					      

			if(flow_gpu){

				if(ixyScale < flow_gpu_range){ //GPU version
				 aux = GpuDescriptors::opticalFlowGpu(grey_mat,frameNum,ixyScale,&memoryof);	 
				 aux.copyTo(flow_mat);
				}
				else  //CPU version
				 cv::calcOpticalFlowFarneback( prev_grey_mat, grey_mat, flow_mat,sqrt(2)/2.0, 5, 10, 2, 7, 1.5, cv::OPTFLOW_FARNEBACK_GAUSSIAN );			
			}
			else //CPU version
			  cv::calcOpticalFlowFarneback( prev_grey_mat, grey_mat, flow_mat,sqrt(2)/2.0, 5, 10, 2, 7, 1.5, cv::OPTFLOW_FARNEBACK_GAUSSIAN );

			endTime(&times,0);
			times.optical_flow_time += calculateTime(&times,0);
			//OPTICAL FLOW
			


			//OPTICAL FLOW TRACKER
			//Track feature points by median filtering
			startTime(&times,0);

			//To calculate the optical flow tracker in the gpu, two requirements are needed
			//1. OFTrackerGpu must be activated
			//2. The level of the pyramid image must be lower than OFTrackerGpuRange value
			if(optical_flow_tracker_gpu){

				
				if(ixyScale < optical_flow_tracker_gpu_range)//GPU version
					GpuDescriptors::opticalFlowTrackerGpuSync(flow_vector[ixyScale], points_in_vector[ixyScale],points_out_vector[ixyScale], status_vector[ixyScale],&memoryoftracker);	
				else//CPU version
					Descriptors::opticalFlowTracker(flow_vector[ixyScale], points_in_vector[ixyScale], points_out_vector[ixyScale], status_vector[ixyScale]);
			}
			else{//CPU version
			  Descriptors::opticalFlowTracker(flow_vector[ixyScale], points_in_vector[ixyScale], points_out_vector[ixyScale], status_vector[ixyScale]);
			}
			
			endTime(&times,0);
			times.optical_flow_tracker_time += calculateTime(&times,0);
			
			//OPTICAL FLOW TRACKER
			cvReleaseImage( &grey_temp );
		}



		/////////////////////////////////////////
		//                                     // 
		//Integral Histograms + get Descriptors//
		//                                     //
		/////////////////////////////////////////


		
		int width;
	    int height;
		float *d_sum;
		vector<cv::Mat> channelsM(2);

		DescMat** curldivMat;
		DescMat** shearCMat;
		DescMat** shearDMat;
		DescMat** hofMat;
		DescMat **hogMat;
		DescMat **mbhMatX; 
		DescMat **mbhMatY; 

		curldivMat = (DescMat**)malloc(sizeof(DescMat*)*scale_num);
		shearCMat =  (DescMat**)malloc(sizeof(DescMat*)*scale_num);
		shearDMat =  (DescMat**)malloc(sizeof(DescMat*)*scale_num);
		hofMat =     (DescMat**)malloc(sizeof(DescMat*)*scale_num);
		hogMat =     (DescMat**)malloc(sizeof(DescMat*)*scale_num);
		mbhMatX =    (DescMat**)malloc(sizeof(DescMat*)*scale_num);
		mbhMatY =    (DescMat**)malloc(sizeof(DescMat*)*scale_num);

		//Compute the integral histogram and get the descriptors for all the levels

		//This loop is designed to calculate both processes simultaneously. On the one hand,
		//the integral histograms are calculated in the first scale_num iterations. On the other hand,
		//"get the descriptors" are calculated from the second iteration to the last one. In this way, GPU and
		//CPU will work at the same time after the first iteration due to the fact that the intregral histograms
		//are calculated in the gpu wihtout blocking the cpu

		for(int ixyScale=0; ixyScale< scale_num+1; ixyScale++){
			
		if( ixyScale < scale_num){ 
			//INTEGRAL HISTOGRAMS
			startTime(&times,0);
			
			width  = flow_vector[ixyScale]->width;
			height = flow_vector[ixyScale]->height;
			 
		  if(feat_comb){

			curldivMat[ixyScale] = InitDescMat(height, width, curldivInfo.nBins);
			shearCMat[ixyScale]  = InitDescMat(height, width, shearCDInfo.nBins);
			shearDMat[ixyScale]  = InitDescMat(height, width, shearCDInfo.nBins);

			//To calculate the integral histograms in the gpu, two requirements are needed
			// 1. integralHistogramsGpu must be activated
			// 2. The level of the pyramid image must be lower than histogramsGpuRange value
			if(ixyScale < histograms_gpu_range && integral_histograms_gpu){
				
				cudaMalloc((void **)&d_sum,sizeof(float)*8*height*width);

				float parameters[6];

				 parameters[0] = curldivInfo.fullOrientation ? 360 : 180;
				 parameters[1] = (float)(curldivInfo.flagThre ? curldivInfo.nBins-1 : curldivInfo.nBins);
				 parameters[2] = parameters[0]/parameters[1];

				 parameters[3] = shearCDInfo.fullOrientation ? 360 : 180;
				 parameters[4] = (float)(shearCDInfo.flagThre ? shearCDInfo.nBins-1 : shearCDInfo.nBins);
				 parameters[5] = parameters[3]/parameters[4];
				
				if(flow_gpu){//GPU VERSION

					if(ixyScale % 2 == 0){
					  GpuDescriptors::kinemCompGpuAsync(memoryof.h_optical_flow[ixyScale], streamof, curldivInfo.nBins, parameters, d_sum,
						 memoryih.d_flow_x_dx[ixyScale], memoryih.d_flow_x_dy[ixyScale], memoryih.d_flow_y_dx[ixyScale], memoryih.d_flow_y_dy[ixyScale],
						 memoryih.h_flow_x_dx[ixyScale], memoryih.h_flow_x_dy[ixyScale], memoryih.h_flow_y_dx[ixyScale],memoryih.h_flow_y_dy[ixyScale],
						  descMatPinnedMemory_1->desc, descMatPinnedMemory_2->desc, descMatPinnedMemory_3->desc);

					}
					else{
						GpuDescriptors::kinemCompGpuAsync(memoryof.h_optical_flow[ixyScale], streamof, curldivInfo.nBins,  parameters,d_sum,
						  memoryih.d_flow_x_dx[ixyScale], memoryih.d_flow_x_dy[ixyScale], memoryih.d_flow_y_dx[ixyScale], memoryih.d_flow_y_dy[ixyScale],
						memoryih.h_flow_x_dx[ixyScale], memoryih.h_flow_x_dy[ixyScale], memoryih.h_flow_y_dx[ixyScale],memoryih.h_flow_y_dy[ixyScale],
						 descMatPinnedMemory_4->desc, descMatPinnedMemory_5->desc, descMatPinnedMemory_6->desc);

					}
				}
				else{

					cv::split(flow_vector[ixyScale],channelsM);

					if(ixyScale % 2 == 0){
						GpuDescriptors::kinemCompGpuAsync(channelsM, streamof,  curldivInfo.nBins, parameters, d_sum,
						  memoryih.d_flow_x_dx[ixyScale], memoryih.d_flow_x_dy[ixyScale], memoryih.d_flow_y_dx[ixyScale], memoryih.d_flow_y_dy[ixyScale],
						 memoryih.h_flow_x_dx[ixyScale], memoryih.h_flow_x_dy[ixyScale], memoryih.h_flow_y_dx[ixyScale],memoryih.h_flow_y_dy[ixyScale],
						  descMatPinnedMemory_1->desc, descMatPinnedMemory_2->desc, descMatPinnedMemory_3->desc);

					}
					else{
						GpuDescriptors::kinemCompGpuAsync(channelsM, streamof, curldivInfo.nBins,parameters, d_sum,
						  memoryih.d_flow_x_dx[ixyScale], memoryih.d_flow_x_dy[ixyScale], memoryih.d_flow_y_dx[ixyScale], memoryih.d_flow_y_dy[ixyScale],
						 memoryih.h_flow_x_dx[ixyScale], memoryih.h_flow_x_dy[ixyScale], memoryih.h_flow_y_dx[ixyScale],memoryih.h_flow_y_dy[ixyScale],
						  descMatPinnedMemory_4->desc, descMatPinnedMemory_5->desc, descMatPinnedMemory_6->desc);

					}
				}
			}
			else{//CPU VERSION
			 Descriptors::kinemComp(flow_vector[ixyScale], curldivMat[ixyScale], shearCMat[ixyScale], shearDMat[ixyScale], curldivInfo, shearCDInfo);
			}
		  }else{

			 hogMat[ixyScale] = InitDescMat(height,width, hogInfo.nBins);
			 hofMat[ixyScale] = InitDescMat(height,width, hofInfo.nBins);
			 mbhMatX[ixyScale] = InitDescMat(height, width, mbhInfo.nBins);
			 mbhMatY[ixyScale] = InitDescMat(height, width, mbhInfo.nBins);

			 if(ixyScale < histograms_gpu_range && integral_histograms_gpu){

				 cudaMalloc((void **)&d_sum,sizeof(float)*8*height*width);

				 	float parameters[3];

					 parameters[0] = mbhInfo.fullOrientation ? 360 : 180;
					 parameters[1] = (float)(mbhInfo.flagThre ? mbhInfo.nBins-1 : mbhInfo.nBins);
					 parameters[2] = parameters[0]/parameters[1];

				 if(flow_gpu){//GPU VERSION
					if(ixyScale % 2 == 0){
						GpuDescriptors::hogMbhCompGpuAsync(memoryof.h_optical_flow[ixyScale],prev_grey_temp_vector[ixyScale],streamof,
						  mbhInfo.nBins, parameters, d_sum,
						  memoryih.d_flow_x_dx[ixyScale],  memoryih.d_flow_x_dy[ixyScale], memoryih.d_flow_y_dx[ixyScale], memoryih.d_flow_y_dy[ixyScale],
						  memoryih.d_hog_x[ixyScale], memoryih.d_hog_y[ixyScale],memoryih.h_flow_x_dx[ixyScale],memoryih.h_flow_x_dy[ixyScale],
					      memoryih.h_flow_y_dx[ixyScale], memoryih.h_flow_y_dy[ixyScale],  memoryih.h_hog_x[ixyScale],memoryih.h_hog_y[ixyScale],
						  descMatPinnedMemory_1->desc,descMatPinnedMemory_2->desc, descMatPinnedMemory_3->desc );
					}
					else{
						GpuDescriptors::hogMbhCompGpuAsync(memoryof.h_optical_flow[ixyScale],prev_grey_temp_vector[ixyScale],streamof,
						   mbhInfo.nBins, parameters, d_sum, 
						  memoryih.d_flow_x_dx[ixyScale],  memoryih.d_flow_x_dy[ixyScale], memoryih.d_flow_y_dx[ixyScale], memoryih.d_flow_y_dy[ixyScale],
						  memoryih.d_hog_x[ixyScale], memoryih.d_hog_y[ixyScale],memoryih.h_flow_x_dx[ixyScale],memoryih.h_flow_x_dy[ixyScale],
					      memoryih.h_flow_y_dx[ixyScale], memoryih.h_flow_y_dy[ixyScale],  memoryih.h_hog_x[ixyScale],memoryih.h_hog_y[ixyScale],
						  descMatPinnedMemory_4->desc,descMatPinnedMemory_5->desc, descMatPinnedMemory_6->desc);
					}
				}
			    else{

					cv::split(flow_vector[ixyScale],channelsM);

					if(ixyScale % 2 == 0){
						GpuDescriptors::hogMbhCompGpuAsync(channelsM,prev_grey_temp_vector[ixyScale],streamof,
						  mbhInfo.nBins, parameters, d_sum,  
						  memoryih.d_flow_x_dx[ixyScale],  memoryih.d_flow_x_dy[ixyScale], memoryih.d_flow_y_dx[ixyScale], memoryih.d_flow_y_dy[ixyScale],
						  memoryih.d_hog_x[ixyScale], memoryih.d_hog_y[ixyScale],memoryih.h_flow_x_dx[ixyScale],memoryih.h_flow_x_dy[ixyScale],
					      memoryih.h_flow_y_dx[ixyScale], memoryih.h_flow_y_dy[ixyScale],  memoryih.h_hog_x[ixyScale],memoryih.h_hog_y[ixyScale],
						  descMatPinnedMemory_1->desc,descMatPinnedMemory_2->desc, descMatPinnedMemory_3->desc);
					}
					else{
						GpuDescriptors::hogMbhCompGpuAsync(channelsM,prev_grey_temp_vector[ixyScale],streamof,
						  mbhInfo.nBins, parameters, d_sum, 
						  memoryih.d_flow_x_dx[ixyScale],  memoryih.d_flow_x_dy[ixyScale], memoryih.d_flow_y_dx[ixyScale], memoryih.d_flow_y_dy[ixyScale],
						  memoryih.d_hog_x[ixyScale], memoryih.d_hog_y[ixyScale],memoryih.h_flow_x_dx[ixyScale],memoryih.h_flow_x_dy[ixyScale],
					      memoryih.h_flow_y_dx[ixyScale], memoryih.h_flow_y_dy[ixyScale],  memoryih.h_hog_x[ixyScale],memoryih.h_hog_x[ixyScale],
						  descMatPinnedMemory_4->desc,descMatPinnedMemory_5->desc, descMatPinnedMemory_6->desc);
					}
				}
			 }
			 else{//CPU VERSION
			 Descriptors::hogComp(prev_grey_temp_vector[ixyScale], hogMat[ixyScale], hogInfo);
			 Descriptors::mbhComp(flow_vector[ixyScale], mbhMatX[ixyScale], mbhMatY[ixyScale], mbhInfo);
			}

		    Descriptors::hofComp(flow_vector[ixyScale], hofMat[ixyScale], hofInfo);

			 cvReleaseImage(&prev_grey_temp_vector[ixyScale]);
		  }
		 }
		//INTEGRAL HISTOGRAMS


		endTime(&times,0);
		times.integral_histograms_time += calculateTime(&times,0);

		//GET DESCRIPTORS
		 if(ixyScale > 0){
			startTime(&times,0);
			int index = ixyScale-1;
			std::list<Track>& tracks = xyScaleTracks[index];
			i = 0;
			for (std::list<Track>::iterator iTrack = tracks.begin(); iTrack != tracks.end(); ++i) {

			if( status_vector[index][i] == 1 ) { // if the feature point is successfully tracked

				PointDesc& pointDesc = iTrack->pointDescs.back();
				CvPoint2D32f prev_point = points_in_vector[index][i];
				// get the descriptors for the feature point
				CvScalar rect = Descriptors::getRect(prev_point, cvSize(width, height), hogInfo);
				
				if(feat_comb){

				 if(!integral_histograms_gpu){
				   pointDesc.curldiv = Descriptors::getDesc(curldivMat[index], rect, curldivInfo);
				   pointDesc.shearC = Descriptors::getDesc(shearCMat[index], rect, shearCDInfo);
				   pointDesc.shearD = Descriptors::getDesc(shearDMat[index], rect, shearCDInfo);
				 }
				 else{

				if(index >= flow_gpu_range){
				   pointDesc.curldiv = Descriptors::getDesc(curldivMat[index], rect, curldivInfo);
				   pointDesc.shearC = Descriptors::getDesc(shearCMat[index], rect, shearCDInfo);
				   pointDesc.shearD = Descriptors::getDesc(shearDMat[index], rect, shearCDInfo);
				}
				else
				{
				 if(index %2 == 0){
				   pointDesc.curldiv = Descriptors::getDesc(descMatPinnedMemory_1, rect, curldivInfo);
				   pointDesc.shearC = Descriptors::getDesc(descMatPinnedMemory_2, rect, shearCDInfo);
				   pointDesc.shearD = Descriptors::getDesc(descMatPinnedMemory_3, rect, shearCDInfo);
				 }
				 else{
				   pointDesc.curldiv = Descriptors::getDesc(descMatPinnedMemory_4, rect, curldivInfo);
				   pointDesc.shearC = Descriptors::getDesc(descMatPinnedMemory_5, rect, shearCDInfo);
				   pointDesc.shearD = Descriptors::getDesc(descMatPinnedMemory_6, rect, shearCDInfo);
				 }
				}
				 }

				PointDesc point(curldivInfo, shearCDInfo, points_out_vector[index][i]);
				iTrack->addPointDesc(point); 
				
				}else{

					if(!integral_histograms_gpu){
				    pointDesc.hog = Descriptors::getDesc(hogMat[index], rect, hogInfo);
					pointDesc.hof = Descriptors::getDesc(hofMat[index], rect, hofInfo);
					pointDesc.mbhX = Descriptors::getDesc(mbhMatX[index], rect, mbhInfo);
					pointDesc.mbhY = Descriptors::getDesc(mbhMatY[index], rect, mbhInfo);
				    }
					else{

					if(index >= flow_gpu_range){
					pointDesc.hog = Descriptors::getDesc(hogMat[index], rect, hogInfo);
					pointDesc.hof = Descriptors::getDesc(hofMat[index], rect, hofInfo);
					pointDesc.mbhX = Descriptors::getDesc(mbhMatX[index], rect, mbhInfo);
					pointDesc.mbhY = Descriptors::getDesc(mbhMatY[index], rect, mbhInfo);
					}
					else{
						if(index %2 == 0){
					pointDesc.hog = Descriptors::getDesc(descMatPinnedMemory_3, rect, hogInfo);
					pointDesc.hof = Descriptors::getDesc(hofMat[index], rect, hofInfo);
					pointDesc.mbhX = Descriptors::getDesc(descMatPinnedMemory_1, rect, mbhInfo);
					pointDesc.mbhY = Descriptors::getDesc(descMatPinnedMemory_2, rect, mbhInfo);
						}
						else{
					pointDesc.hog = Descriptors::getDesc(descMatPinnedMemory_6, rect, hogInfo);
					pointDesc.hof = Descriptors::getDesc(hofMat[index], rect, hofInfo);
					pointDesc.mbhX = Descriptors::getDesc(descMatPinnedMemory_4, rect, mbhInfo);
					pointDesc.mbhY = Descriptors::getDesc(descMatPinnedMemory_5, rect, mbhInfo);
						}
					}
					}

				PointDesc point(hogInfo, hofInfo, mbhInfo, points_out_vector[index][i]);
				iTrack->addPointDesc(point); 
				}


				// draw this track			
				if( show_track == 1 ) {

					std::list<PointDesc>& descs = iTrack->pointDescs;
					std::list<PointDesc>::iterator iDesc = descs.begin();
					float length = descs.size();
					CvPoint2D32f point0 = iDesc->point;
					point0.x *= fscales[index]; // map the point to first scale
					point0.y *= fscales[index];
				
					

					float j = 0;
					for (iDesc++; iDesc != descs.end(); ++iDesc, ++j) {
						CvPoint2D32f point1 = iDesc->point;
						point1.x *= fscales[index];
						point1.y *= fscales[index];

						cvLine(image, cvPointFrom32f(point0), cvPointFrom32f(point1),
							   CV_RGB(0,cvFloor(255.0*(j+1.0)/length),0), 2, 8,0);
						point0 = point1;
					}
					cvCircle(image, cvPointFrom32f(point0), 2, CV_RGB(255,0,0), -1, 8,0);
				}

				++iTrack;
			}
			else // remove the track, if we lose feature point
				iTrack = tracks.erase(iTrack);
			}

			endTime(&times,0);
			times.get_descriptors_time += calculateTime(&times,0);

		 }
		 //GET DESCRIPTORS

			// Release
		  if(ixyScale < flow_gpu_range && integral_histograms_gpu)
			 cudaFree(d_sum);

		}

		if(feat_comb){
		for(int i=0; i< scale_num; i++){
			ReleDescMat(curldivMat[i]);
			ReleDescMat(shearCMat[i]);
			ReleDescMat(shearDMat[i]);
			points_in_vector[i].clear();
			points_out_vector[i].clear();
			status_vector[i].clear();
			cvReleaseImage( &flow_vector[i]);
		}
		}else{
			for(int i=0; i< scale_num; i++){
			ReleDescMat(hofMat[i]);
			ReleDescMat(hogMat[i]);
			ReleDescMat(mbhMatX[i]);
			ReleDescMat(mbhMatY[i]);
			points_in_vector[i].clear();
			points_out_vector[i].clear();
			status_vector[i].clear();
			cvReleaseImage( &flow_vector[i]);
			}
		}

		for( int ixyScale = 0; ixyScale < scale_num; ++ixyScale ) {
		std::list<Track>& tracks = xyScaleTracks[ixyScale]; // output the features for each scale
		for( std::list<Track>::iterator iTrack = tracks.begin(); iTrack != tracks.end(); ) {
			if( (signed)iTrack->pointDescs.size() >= tracker.trackLength+1 ) { // if the trajectory achieves the length we want
				std::vector<CvPoint2D32f> trajectory(tracker.trackLength+1);
				std::list<PointDesc>& descs = iTrack->pointDescs;
				std::list<PointDesc>::iterator iDesc = descs.begin();

				for (int count = 0; count <= tracker.trackLength; ++iDesc, ++count) {
					trajectory[count].x = iDesc->point.x*fscales[ixyScale];
					trajectory[count].y = iDesc->point.y*fscales[ixyScale];
				}
				float mean_x(0), mean_y(0), var_x(0), var_y(0), length(0);

				if( Descriptors::isValid(trajectory, mean_x, mean_y, var_x, var_y, length) == 1 ) {

				 if(hogdesc || hofdesc || mbhdesc || curldivdesc || shearCdesc || shearDdesc){
					descriptorsFile << frameNum << "\t" << mean_x << "\t" << mean_y << "\t" << var_x << "\t"
						            << var_y << "\t" <<length << "\t" << fscales[ixyScale] << "\t";
				 }

					for (int count = 0; count < tracker.trackLength; ++count){
					   if(hogdesc || hofdesc || mbhdesc || curldivdesc || shearCdesc || shearDdesc)
						descriptorsFile << trajectory[count].x << "\t" << trajectory[count].y << "\t";
					}

					int t_stride;
					if(!feat_comb && hogdesc) {	// HOG
					    iDesc = descs.begin();
					    t_stride = cvFloor(tracker.trackLength/hogInfo.ntCells);	
					    for( int n = 0; n < hogInfo.ntCells; n++ ) {
						std::vector<float> vec(hogInfo.dim);
						for( int t = 0; t < t_stride; t++, iDesc++ )
							for( int m = 0; m < hogInfo.dim; m++ )
								vec[m] += iDesc->hog[m];
												
							for( int m = 0; m < hogInfo.dim; m++ ){
								descriptorsFile << vec[m]/float(t_stride) << "\t";
							}
						}
					}

					if(!feat_comb && hofdesc) {	// HOF
					    iDesc = descs.begin();
					    t_stride = cvFloor(tracker.trackLength/hofInfo.ntCells);
						std::vector<float> vec(hofInfo.dim);
					    for( int n = 0; n < hofInfo.ntCells; n++ ) {			
						for( int t = 0; t < t_stride; t++, iDesc++ )
							for( int m = 0; m < hofInfo.dim; m++ )
								vec[m] += iDesc->hof[m];
						for( int m = 0; m < hofInfo.dim; m++ ){
							descriptorsFile << vec[m]/float(t_stride) << "\t";					
						}
						}
					}

					if(!feat_comb && mbhdesc) {		// MBH
					    iDesc = descs.begin();
					    t_stride = cvFloor(tracker.trackLength/mbhInfo.ntCells);
					    for( int n = 0; n < mbhInfo.ntCells; n++ ) {
						std::vector<float> vec(mbhInfo.dim);

						for( int t = 0; t < t_stride; t++, iDesc++ )
							for( int m = 0; m < mbhInfo.dim; m++ )
								vec[m] += iDesc->mbhX[m];
						for( int m = 0; m < mbhInfo.dim; m++ ){
							descriptorsFile << vec[m]/float(t_stride) << "\t";
						}
						}
					    
					    iDesc = descs.begin();
					    t_stride = cvFloor(tracker.trackLength/mbhInfo.ntCells);
					    for( int n = 0; n < mbhInfo.ntCells; n++ ) {
						std::vector<float> vec(mbhInfo.dim);
						for( int t = 0; t < t_stride; t++, iDesc++ )
							for( int m = 0; m < mbhInfo.dim; m++ )
								vec[m] += iDesc->mbhY[m];
						for( int m = 0; m < mbhInfo.dim; m++ ){
							descriptorsFile << vec[m]/float(t_stride) << "\t";
						}
						}
					 }

					if(feat_comb){
						if(curldivdesc){
					    iDesc = descs.begin();
					    t_stride = cvFloor(tracker.trackLength/curldivInfo.ntCells);
						//string descName = "curldiv";
						//descName += to_string(frameNum);
						//descName += ".xml";
						//filestorage.open(descName,cv::FileStorage::WRITE);
					    for( int n = 0; n < curldivInfo.ntCells; n++ ) {
                                                std::vector<float> vec(curldivInfo.dim);
                                                for( int t = 0; t < t_stride; t++, iDesc++ )
                                                        for( int m = 0; m < curldivInfo.dim; m++ )
                                                                vec[m] += iDesc->curldiv[m];
                                                for( int m = 0; m < curldivInfo.dim; m++ ){
													descriptorsFile << vec[m]/float(t_stride) << "\t";
												}
											
						}
						}

						if(shearCdesc){
					    iDesc = descs.begin();
                                            t_stride = cvFloor(tracker.trackLength/shearCDInfo.ntCells);
                                            for( int n = 0; n < shearCDInfo.ntCells; n++ ) {
                                                std::vector<float> vec(shearCDInfo.dim);
                                                for( int t = 0; t < t_stride; t++, iDesc++ )
                                                        for( int m = 0; m < shearCDInfo.dim; m++ )
                                                                vec[m] += iDesc->shearC[m];
                                                for( int m = 0; m < shearCDInfo.dim; m++ ){
												 descriptorsFile << vec[m]/float(t_stride) << "\t";
												}
											}
						}

						if(shearDdesc){
					    iDesc = descs.begin();
                                            t_stride = cvFloor(tracker.trackLength/shearCDInfo.ntCells);
                                            for( int n = 0; n < shearCDInfo.ntCells; n++ ) {
                                                std::vector<float> vec(shearCDInfo.dim);
                                                for( int t = 0; t < t_stride; t++, iDesc++ )
                                                        for( int m = 0; m < shearCDInfo.dim; m++ )
                                                                vec[m] += iDesc->shearD[m];
                                                for( int m = 0; m < shearCDInfo.dim; m++ ){
													descriptorsFile << vec[m]/float(t_stride) << "\t";
												}
											}
						}
					}		
					descriptorsFile <<"\n";
				}
				iTrack = tracks.erase(iTrack);
			}
			else
				iTrack++;
		    }
		}

		if( init_counter == tracker.initGap ) { // detect new feature points every initGap frames
		init_counter = 0;
		for (int ixyScale = 0; ixyScale < scale_num; ++ixyScale) {
			std::list<Track>& tracks = xyScaleTracks[ixyScale];
			std::vector<CvPoint2D32f> points_in(0);
			std::vector<CvPoint2D32f> points_out(0);
			for(std::list<Track>::iterator iTrack = tracks.begin(); iTrack != tracks.end(); iTrack++, i++) {
				std::list<PointDesc>& descs = iTrack->pointDescs;
				CvPoint2D32f point = descs.back().point; // the last point in the track
				points_in.push_back(point);
			}

			IplImage *grey_temp = 0, *eig_temp = 0;
			std::size_t temp_level = (std::size_t)ixyScale;
			grey_temp = cvCloneImage(grey_pyramid.getImage(temp_level));
			eig_temp = cvCloneImage(eig_pyramid.getImage(temp_level));

			Descriptors::cvDenseSample(grey_temp, eig_temp, points_in, points_out, quality, min_distance);
			// save the new feature points
			for( i = 0; i < (signed)points_out.size(); i++) {
				Track track(tracker.trackLength);
				if(feat_comb) {
				    PointDesc point(curldivInfo, shearCDInfo, points_out[i]);
				    track.addPointDesc(point); }
				else {
				    if(w_flow) {
					PointDesc point(0, hogInfo, hofInfo, mbhInfo, points_out[i]);
					track.addPointDesc(point); }
				    else {
					PointDesc point(hogInfo, hofInfo, mbhInfo, points_out[i]);
					track.addPointDesc(point); }
				}
				tracks.push_back(track);
			}
			cvReleaseImage( &grey_temp );
			cvReleaseImage( &eig_temp );
		}
		}
		}

		cvCopy( frame, prev_image, 0 );
		cvCvtColor( prev_image, prev_grey, CV_BGR2GRAY );
		prev_grey_pyramid.rebuild(prev_grey);
		}

		if( show_track == 1 ) {
			cvShowImage( "DenseTrack", image);

			c = cvWaitKey(1);
		}

		// get the next frame
		frameNum++;
	}

	if( show_track == 1 ){
		cvDestroyWindow("DenseTrack");
	}

	if(optical_flow_tracker_gpu){
		cudaStreamDestroy(memoryoftracker.stream);
	}

	if(integral_histograms_gpu){
	cudaFreeHost(descMatPinnedMemory_1->desc);
    cudaFreeHost(descMatPinnedMemory_2->desc);
    cudaFreeHost(descMatPinnedMemory_3->desc);
	cudaFreeHost(descMatPinnedMemory_4->desc);
    cudaFreeHost(descMatPinnedMemory_5->desc);
    cudaFreeHost(descMatPinnedMemory_6->desc);
	}

	endTime(&times,1);

	float threeMainFunctionsTime = times.optical_flow_time+times.optical_flow_tracker_time+times.integral_histograms_time;
	cout << "----------------------------Results---------------------------------- " << endl;
	cout << "---------------------------------------------------------------------" << endl;
	cout << "Total algorithm time: " << calculateTime(&times, 1) << " ms" << endl;
	cout << "---------------------------------------------------------------------" << endl;	
	cout << "---------------------------------------------------------------------" << endl;
	cout << "Total optical flow time: " << times.optical_flow_time << " ms (" <<  times.optical_flow_time*100/calculateTime(&times, 1) << "%)" << endl;
	cout << "Total optical flow tracker time: " << times.optical_flow_tracker_time << " ms (" <<  times.optical_flow_tracker_time*100/calculateTime(&times, 1) << "%)" << endl;
	cout << "Total integral histograms time: " << times.integral_histograms_time << " ms (" <<  times.integral_histograms_time*100/calculateTime(&times, 1) << "%)" << endl; 
	cout << "Total time: " << threeMainFunctionsTime << " ms. Percentage: " 
		 << threeMainFunctionsTime*100.0/calculateTime(&times, 1) << "%" <<endl;
	cout << "---------------------------------------------------------------------" << endl;
	cout << "---------------------------------------------------------------------" << endl;
	cout << "Total get descriptors time: " << times.get_descriptors_time << " ms (" <<  times.get_descriptors_time*100/calculateTime(&times, 1) << "%)" << endl; 
	cout << "Total time: " << threeMainFunctionsTime + times.get_descriptors_time << " ms. Percentage: " 
		 << (threeMainFunctionsTime + times.get_descriptors_time)*100/calculateTime(&times, 1) << "%" <<endl;
	cout << "---------------------------------------------------------------------" << endl;
	cout << "---------------------------------------------------------------------" << endl;
	
	descriptorsFile.close();

	cout << "Press to end";
	getchar();

	return 0;
}
