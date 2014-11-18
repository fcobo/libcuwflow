===============================================================================
libcuwflow: CUDA library to compute "wFlow dense trajectories" on a GPU in C++.
===============================================================================
Fernando Cobo Aguilera and Manuel J. Marin-Jimenez


This software is a CUDA implementation of the "wFlow dense trajectories" described in Jain et al. [1]
--------------------------------------------------------------------------------

   Source-Code:   https://github.com/fcobo/libcuwflow

--------------------------------------------------------------------------------
Contents of the package:
--------------------------------------------------------------------------------
- data - contains test data
- include - contains all the software header files
- src - contains all the software source files
- tests - contains a program to test the library


--------------------------------------------------------------------------------
Requirements:
--------------------------------------------------------------------------------
This software has been tested on Windows 7 and Ubuntu 12.04 LTS (Precise Pangolin) with the following libraries:
- OpenCV - v2.4.8 (required)
- CUDA - v5.5 (required)
- Boost - v1.49.0 (required)


--------------------------------------------------------------------------------
Quick start:
--------------------------------------------------------------------------------
Let us assume that the root directory of libcuwflow is named ‘rootdir’.

Open a terminal, and type in the command line the following instructions:
```
1) cd <rootdir>
2) mkdir build
3) cd build
4) cmake ..
5) make
6) make install   (You might need to do sudo if your are in an Unix-like system)
```
If everything went well, both the library and test programs should have been
created into <rootdir>/build subdirectories.

Note for Windows users: make sure that the environment variables needed by CMake have 
been properly setup before running the `cmake' command. E.g. BOOST_ROOT and CUDA_PATH

You can run the test program by executing the following command:
```
cd <rootdir>
cudadensef --video .\data\tr01_cam00.avi --start 100 --end 200 --curldiv --shearC --shearD --ofgpu --oftgpu --ihgpu --show
```

--------------------------------------------------------------------------------
Citation:
--------------------------------------------------------------------------------
If you use this library for your publications, please, consider citing the 
following publications:<br>
@inproceedings{castro2014icpr,  
author = {Castro, F. M. and Marin-Jimenez, Manuel J. and Medina-Carnicer, Rafael},
 title  = {{Pyramidal Fisher Motion} for Multiview Gait Recognition},
 year = {2014},
 booktitle = {Intl. Conference on Pattern Recognition (ICPR)}
}

@misc{libcuwflow,  
author = {Cobo-Aguilera, Fernando and Marin-Jimenez, Manuel J.},
 title = {{LibCuWFlow}: A CUDA library for computing dense trajectories in {C++}},
 year = {2014},
 note =   {Software available at \url{https://github.com/fcobo/libcuwflow}}
}

--------------------------------------------------------------------------------
Contact the authors:
--------------------------------------------------------------------------------
Fernando Cobo Aguilera (developer) - i92coagf@uco.es / fcoboaguilera@gmail.com<br>
Manuel J. Marin-Jimenez (advisor) - mjmarin@uco.es


--------------------------------------------------------------------------------
References:
--------------------------------------------------------------------------------
[1] Jain, M.; Jegou, H.; Bouthemy , P.; (2013): "Better exploiting motion for better
action recognition". Computer Vision and Pattern Recognition (CVPR) 2013 IEEE
Conference on, On page(s): 2555 - 2562


--------------------------------------------------------------------------------
Version history:
--------------------------------------------------------------------------------

- v0.1: first release.
