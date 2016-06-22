DyBa ORF
--------
DyBa ORF is a Dymanically Balanced Online Random Forest, which addresses three problems in learning with Random Forests:  
1). Online learning. It can accept gradually arrived training data and update the model on-the-fly.
2). Data imbalance. It deals with imbalanced data of different classes, avoiding low correct rate of classification for minor class.
3). Changing imbalance ratio. The ratio (degree) of imbalance may dynamically change when new training data arrive sequentially. Compared with traditionaly Online Random Forest (Saffari, A. 2009, Barinova, O. 2012) which assumes the imblance ratio do not change,
    DyBa ORF updates the model dynamically with the ability to be adapted to new imbalance ratio.

DyBa ORF was was developed as part of the [GIFT-Surg][giftsurg] project. The algorithm and software were developed by [Guotai Wang][guotai] at the [Translational Imaging Group][tig] in the [Centre for Medical Image Computing][cmic] at [University College London (UCL)][ucl].

How to cite
----------

If you use this software, please cite this paper:

Guotai Wang, Maria A. Zuluaga, Rosalind Pratt, Michael Aertsen, Tom Doel, Maria Klusmann, Anna L. David, Jan Deprest, Tom Vercauteren, Sebastien Ourselin
Dynamically Balanced Online Random Forests for Interactive Segmentation, MICCAI 2016. (accepted)

Software links
--------------

- [DyBa ORF home page][DyBaORFHome].
- [GitHub mirror][githubhome].

License
-----------

Copyright (c) 2014-2016, University College London.
DyBa ORF is available as free open-source software under a BSD 3-Clause License. Other licenses may apply for dependencies:


System requirements
-------------------

The current version of DyBa ORF requries:
* A C++ compiler installed and configured to work with cmake.
* Cmake to generate the project from source code.

How to use
-------------------

1, Suppose the source code directory is $DyBaORF, create a new folder named "build" under that.
2, Switch to the $DyBaORF directory.
3, Open Cmake, set the source code directory as $DyBaORF, and set the build directory as $DyBaORF/build.
4, Click Configure, select the c++ compile you use. (VS in Windows, Clang++ in Mac, gcc in Linux)
5, Click Generate to generate the project, which can be found at $DyBaORF/build.
6, Go to $DyBaORF/build to open the generated project, compile it and then you can run it.

Funding
-------------------

This work was supported through an Innovative Engineering for Health award by the [Wellcome Trust][wellcometrust] [WT101957], the [Engineering and Physical Sciences Research Council (EPSRC)][epsrc] [NS/A000027/1] and a [National Institute for Health Research][nihr] Biomedical Research Centre [UCLH][uclh]/UCL High Impact Initiative, a UCL Overseas Research Scholarship and a UCL Graduate Research Scholarship.

Supported Platforms
-------------------

DyBa ORF is a cross-platform C++ library. It supports Windows, Mac and Linux.

[tig]: http://cmictig.cs.ucl.ac.uk
[giftsurg]: http://www.gift-surg.ac.uk
[cmic]: http://cmic.cs.ucl.ac.uk
[ucl]: http://www.ucl.ac.uk
[nihr]: http://www.nihr.ac.uk/research
[uclh]: http://www.uclh.nhs.uk
[epsrc]: http://www.epsrc.ac.uk
[wellcometrust]: http://www.wellcome.ac.uk
[maxflow]: http://uk.mathworks.com/matlabcentral/fileexchange/21310-maxflow
[coremat]: http://github.com/tomdoel/coremat
[dicomat]: http://github.com/tomdoel/dicomat
[citation]: http://www.sciencedirect.com/science/article/pii/S1361841516300287
[DyBaORFHome]: https://cmiclab.cs.ucl.ac.uk/GIFT-Surg/DyBaORF
[githubhome]: https://github.com/gift-surg/DyBaORF
[guotai]: http://cmictig.cs.ucl.ac.uk/people/phd-students/guotai-wang