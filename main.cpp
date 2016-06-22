//
// main.cpp
//
//  Created on: 17 Mar 2015
//      Author: Guotai Wang
//
// Copyright (c) 2014-2016 University College London, United Kingdom. All rights reserved.
// http://cmictig.cs.ucl.ac.uk
//
// Distributed under the BSD-3 licence. Please see the file licence.txt
//

#include <fstream>
#include <iostream>
#include "Test/RFTestExample.h"

int main(int argc, char ** argv)
{
    srand (time(NULL));
    RFTestExample testExample;
    if(testExample.LoadData(BIODEG)) //DataSetName: BIODEG, MUSK, CTG, WINE
    {
        testExample.SetTrainDataChunk(0.5,1.0, 0.05);
        testExample.Run(4);
        testExample.PrintPerformance();
    }
    return EXIT_SUCCESS;
}
