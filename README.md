DyBa ORF

    DyBa ORF is a Dymanically Balanced Online Random Forest, which addresses three problems in learning with Random Forests: 
    
    1). Online learning. It can accept gradually arrived training data and update the model on-the-fly.
    2). Data imbalance. It deals with imbalanced data of different classes, avoiding low correct rate of classification for minor class.
    3). Changing imbalance ratio. The ratio (degree) of imbalance may dynamically change when new training data arrive sequentially. 
    
    Compared with traditionaly Online Random Forest (Saffari, A. 2009, Barinova, O. 2012) which assumes the imblance ratio do not change,
    DyBa ORF updates the model dynamically with the ability to be adapted to new imbalance ratio.

Author Info

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

    Suppose the source code directory is $DyBaORF.
    Switch to the $DyBaORF directory.
    Open Cmake, set the source code directory as $DyBaORF, and set the build directory as $DyBaORF/build.
    Click Configure, select the c++ compile you use. (VS in Windows, Clang++ in Mac, gcc in Linux)
    Click Generate to generate the project, which can be found at $DyBaORF/build.
    Go to $DyBaORF/build to open the generated project, comile it and you can run it.


Funding

    This work was supported through an Innovative Engineering for Health award by the Wellcome Trust [WT101957], the Engineering and Physical Sciences Research Council (EPSRC) [NS/A000027/1] and a National Institute for Health Research Biomedical Research Centre UCLH/UCL High Impact Initiative, A UCL Overseas Research Scholarship and a UCL Graduate Research Scholarship.

Supported Platforms

    DyBa ORF is a cross-platform C++ library. It supports Windows, Mac and Linux.
