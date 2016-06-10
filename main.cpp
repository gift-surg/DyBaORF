/*
 * main.cpp
 *
 *  Created on: 17 Mar 2015
 *      Author: guotaiwang
 */

#include <fstream>
#include <iostream>
#include "Test/RFTestExample.h"

using namespace std;

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
    return 0;
}
