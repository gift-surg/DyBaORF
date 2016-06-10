//
//  TestExampleSVM.h
//  ORF_test
//
//  Created by Guotai Wang on 23/12/2015.
//
//

#ifndef __TestExampleSVM__
#define __TestExampleSVM__

#include "TestExample.h"
#include "../LASVM/lasvmEnsemble.h"

class TestExampleSVM:public TestExample
{
public:
    TestExampleSVM();
    ~TestExampleSVM();
    virtual void Run(int MaxIter);
    virtual void PrintPerformance();
    void NormalizeData();
    void SaveData(string name, int n);
};

#endif /* defined(__TestExampleSVM__) */
