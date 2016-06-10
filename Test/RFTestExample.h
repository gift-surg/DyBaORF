//
//  RFTestExample.h
//  DyBaORF_test
//
//  Created by Guotai Wang on 07/12/2015.
//
// Copyright (c) 2014-2016 University College London, United Kingdom. All rights reserved.
// http://cmictig.cs.ucl.ac.uk
//
// Distributed under the BSD-3 licence. Please see the file licence.txt
//

#ifndef __ORF_test__RFTestExample__
#define __ORF_test__RFTestExample__

#define USE_PROPOSED_ORF 1
#define COMPARE_WITH_MPB_ORF 1
#define COMPARE_WITH_SPB_ORF 1
#define COMPARE_WITH_OFFLINE 1
#include "AbstractTestExample.h"

class RFTestExample:public AbstractTestExample
{
public:
    RFTestExample();
    ~RFTestExample();
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
#endif /* defined(__ORF_test__RFTestExample__) */
