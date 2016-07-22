/*=========================================================================
 Program:   DyBa ORF
 Module:    main.cpp
 
 Created by Guotai Wang on 01/12/2015.
 Copyright (c) 2014-2016 University College London, United Kingdom. All rights reserved.
 http://cmictig.cs.ucl.ac.uk
 
 Reference:
 Dynamically Balanced Online Random Forests for Interactive Scribble-based Segmentation.
 Presented at: MICCAI 2016
 Guotai Wang, Maria A. Zuluaga, Rosalind Pratt, Michael Aertsen, Tom Doel,
 Maria Klusmann, Anna L. David, Jan Deprest, Tom Vercauteren, and Sebastien Ourselin.
 
 Distributed under the BSD-3 licence. Please see the file licence.txt
 =========================================================================*/

#include <fstream>
#include <iostream>
#include "RFTestExample.h"

int main(int argc, char ** argv)
{
    srand (time(NULL));
    RFTestExample testExample;
    if(testExample.LoadData(BIODEG)) //DataSetName: BIODEG, MUSK, CTG, WINE
    {
        testExample.SetTrainDataChunk(0.5,1.0, 0.05);
        testExample.Run(100);
        testExample.PrintPerformance();
    }
    return EXIT_SUCCESS;
}
