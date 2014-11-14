==============================================================================
libcuwflow: CUDA library to calculate "wFlow dense trajectory" on a GPU in C++.
==============================================================================
Fernando Cobo Aguilera and Manuel J. Marin-Jimenez


This software is a CUDA implementation of  "wFlow dense trajectory" described in Jain et al. [1] 
--------------------------------------------------------------------------------

   Source-Code:   https://github.com/fcobo/libcuwflow

--------------------------------------------------------------------------------
Contents of the package:
--------------------------------------------------------------------------------
- include - contains all the software header files
- src - contains all the software source files
- tests - contains a program to test the library
- makefile - used to compile the library, the documentation and the test program


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

--------------------------------------------------------------------------------
Contact the authors:
--------------------------------------------------------------------------------
Fernando Cobo Aguilera (developer) - i92coagf@uco.es / fcoboaguilera@gmail.com
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