/*
 * ORForest.cpp
 *
 *  Created on: 17 Mar 2015
 *      Author: guotaiwang
 */

#include <fstream>
#include <iostream>
#include "Test/TestExample.h"
#include "Test/TestExample1.h"
#include "Test/TestExampleBayes.h"
#include "Test/TestExampleSVM.h"
using namespace std;

//enum DataSetName{COVTYPE, IMAGE_SEG, CTG, ABALONE, CHESS, LETTER, PAGE ,WALL, WINE,YEAST,TRANSFUSION,CLIMATE,CMC,MESSIDOR,EEG,DERMATOLOGY,BANKNOTE,BREASTCANCER,CAR,KIDNEYDISEASE,HOUSEVOTES,CONNECT, CRX, SONAR, BAND, ECOLI,FERTILITY, LMPROVE, GESTURE, GLASS, HEPATITIS, LLPD, IONOSPHERE, MONK, MUSHROOM, MUSK, OPTDIGITS,PARKINSON, PHISHING, BIODEG};
void TestOnlineRandomForest();

int main(int argc, char ** argv)
{
    srand (time(NULL));
    TestExample1 myExample;
//    TestExample1 myExample;
//    TestExampleSVM myExample;
    time_t start=clock();
    if(myExample.LoadData(CTG))
    {
        myExample.SetTrainDataChunk(0.5,1.0, 0.05);
        //myExample.PrintDataInformation();
        myExample.Run(100);
        myExample.PrintPerformance();
    }
    double during=(double)(clock()-start)/CLOCKS_PER_SEC;
//    cout<<"time "<<during<<endl;
    return 0;
}
