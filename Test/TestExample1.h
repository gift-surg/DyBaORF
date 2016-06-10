//
//  TestExample1.h
//  ORF_test
//
//  Created by Guotai Wang on 07/12/2015.
//
//

#ifndef __ORF_test__TestExample1__
#define __ORF_test__TestExample1__

#define USE_PROPOSED_ORF 1
#define COMPARE_WITH_MPB_ORF 1
#define COMPARE_WITH_SPB_ORF 1
#define COMPARE_WITH_OFFLINE 1
#include "TestExample.h"

class TestExample1:public TestExample
{
public:
    TestExample1();
    ~TestExample1();
    virtual void Run(int MaxIter);
    virtual void PrintPerformance();

    
#ifdef COMPARE_WITH_SPB_ORF
    vector<vector<double> > compareSensitivity2;
    vector<vector<double> > compareSpecificity2;
    vector<vector<double> > compareGmean2;
    vector<vector<double> > compareTime2;
#endif
    
#ifdef COMPARE_WITH_OFFLINE
    vector<vector<double> > compareSensitivity3;
    vector<vector<double> > compareSpecificity3;
    vector<vector<double> > compareGmean3;
    vector<vector<double> > compareTime3;
#endif
    
};
#endif /* defined(__ORF_test__TestExample1__) */
