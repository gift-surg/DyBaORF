DyBa ORF

DyBa ORF is a Dymanically Balanced Online Random Forest, which addresses three problems in learning with Random Forests: 
1). Online learning. It can accept gradually arrived training data and update the model on-the-fly.
2). Data imbalance. It deals with imbalanced data of different classes, avoiding low correct rate of classification for minor class.
3). Changing imbalance ratio. The ratio (degree) of imbalance may dynamically change when new training data arrive sequentially. 
    Compared with traditionaly Online Random Forest (Saffari, A. 2009, Barinova, O. 2012) which assumes the imblance ratio do not change,
    DyBa ORF updates the model dynamically with the ability to be adapted to new imbalance ratio. 

DyBa ORF was developed as part of the GIFT-Surg project. The algorithm and software were developed by Guotai Wang at the Translational Imaging Group in the Centre for Medical Image Computing at University College London (UCL).

If you use this software, please cite this paper:
    Guotai Wang, Maria A. Zuluaga, Rosalind Pratt\, Michael Aertsen, Tom Doel, Maria Klusmann, Anna L. David, Jan Deprest, Tom Vercauteren, Sebastien Ourselin
    Dynamically Balanced Online Random Forests for Interactive Segmentation, MICCAI 2016. (accepted)


Software links
    Slic-Seg home page.
    GitHub mirror.

License

Copyright (c) 2014-2016, University College London.

DyBa ORF is available as free open-source software under a BSD 3-Clause License. Other licenses may apply for dependencies:


System requirements

The current version of DyBa ORF requries:

    A C++ compiler installed and configured to work with cmake.
    cmake required to generate the project from source code.

How to use

    Before attempting to use Slic-Seg, please ensure you have C++ and CUDA compilers installed and correctly configured to work with Matlab
        What you need to build mex files
        CUDA Installation guides
        Why can't MEX find a supported compiler in MATLAB R2015b after I upgraded to Xcode 7.0?

    GPU computing may lead to system instability and data loss. Please back up any valuable data before using the software.

    Switch to the matlab directory.

    To launch the user interface, run slicseg on the command window

    Alternatively, run the test script to illustrate use of the algorithm without the user interface. To do this:
        Run 'SlicSegAddPaths` to set up the paths for this session
        Type test in the command window

How to use the user interface

    Run slicseg to launch the user interface.
    The mex and cuda files will automatically compile if they have not already been compiled. This will fail if you have not installed and correctly set up your mex and cuda compilers to work with Matlab.
    Click Load to load Dicom or a series of png image from a directory you specify
    Choose your starting slice (usually a slice in the middle of the object)
    Draw scribbles (lines) over parts of the object you wish to segment. The left button selects the foreground (object) and the right button selects the background.
        The Background button makes the left button select background, while the Foreground button makes the left button select foreground
    Click Segment to segment the object on this slice, based on the scribbes you have entered
    Select the range (start and end slices) over which the segmentation will propagate
    Click Propagate to continue the segmentation over these slices
    Click Save to save the segmentation

Issues

    The most likely issues will be due to not having correctly set up your mex and cuda compilers.
    If you get compilation errors, please fix your mex and cuda compiler setup, then run CompileSlicSeg recompile on the command window to force re-compilation.
    OSX users, please check the supported versions of XCode. Note that NVIDIA CUDA may not support the latest versions of XCode.

    If you are using OSX and receive a No supported compiler or SDK was found error, and you have already installed XCode, please follows these instructions

    On linux, linking problems may occur due to Matlab adding an internal linking path before mex is called. If you get version problems when linking C++ files you can force Matlab to find specific library versions using LD_PRELOAD, for example: LD_PRELOAD=/path-to-desired-library/libstdc++.so.6 /path-to-matlab/bin/matlab

Funding

This work was supported through an Innovative Engineering for Health award by the Wellcome Trust [WT101957], the Engineering and Physical Sciences Research Council (EPSRC) [NS/A000027/1] and a National Institute for Health Research Biomedical Research Centre UCLH/UCL High Impact Initiative.
Supported Platforms

Slic-Seg is a cross-platform Matlab/C++ library. We have tested Slic-Seg on the following platforms:

    Linux
        Ubuntu Desktop 14.04.3 LTS 64-bit
        NVIDIA 12GB GTX TITAN X
        CUDA 7.5
        Matlab R2015b

    MacOS X
        OS X Yosemite 10.10.5
        NVIDIA GeForce GT 750M 1024 MB
        XCode 7.2.1
        CUDA 7.5

    Windows
        Not yet tested
