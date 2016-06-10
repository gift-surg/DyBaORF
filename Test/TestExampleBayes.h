//
//  TestExampleBayes.h
//
//
//  Created by Guotai Wang on 08/12/2015.
//
//

#ifndef __TestExampleBayes__
#define __TestExampleBayes__

#include "TestExample.h"
#include "../Bayes/BayesClassifier.h"
class TestExampleBayes:public TestExample
{
public:
    TestExampleBayes();
    ~TestExampleBayes();
    virtual void Run(int MaxIter);
    virtual void PrintPerformance();
};

#endif /* defined(__TestExampleBayes__) */
