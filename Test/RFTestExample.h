/*=========================================================================
 Program:   DyBa ORF
 Module:    RFTestExample.h
 
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

#pragma once

#define USE_PROPOSED_ORF 1
#define COMPARE_WITH_MPB_ORF 1
#define COMPARE_WITH_SPB_ORF 1
#define COMPARE_WITH_OFFLINE 1
#include "AbstractTestExample.h"

/** \brief class RFTestExample
 *
 * An test example to compare SP ORF, MP ORF, DyBa ORF and its offine counterpart
 */
class RFTestExample:public AbstractTestExample
{
public:
    /** Construction function */
    RFTestExample();
    /** Deconstruction function */
    ~RFTestExample();
    /** Run the forest training and testing 
     * @param[in] maxIter the maximal number of iteration
     */
    virtual void Run(int maxIter);
    
    /** Print performance */
    virtual void PrintPerformance();
    
private:
#ifdef COMPARE_WITH_SPB_ORF
    std::vector<std::vector<double> > compareSensitivity2;
    std::vector<std::vector<double> > compareSpecificity2;
    std::vector<std::vector<double> > compareGmean2;
    std::vector<std::vector<double> > compareTime2;
#endif
    
#ifdef COMPARE_WITH_OFFLINE
    std::vector<std::vector<double> > compareSensitivity3;
    std::vector<std::vector<double> > compareSpecificity3;
    std::vector<std::vector<double> > compareGmean3;
    std::vector<std::vector<double> > compareTime3;
#endif
    
};
