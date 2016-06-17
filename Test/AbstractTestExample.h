//
//  AbstractTestExample.h
//  DyBaORF_test
//
//  Created by Guotai Wang on 03/12/2015.
//
// Copyright (c) 2014-2016 University College London, United Kingdom. All rights reserved.
// http://cmictig.cs.ucl.ac.uk
//
// Distributed under the BSD-3 licence. Please see the file licence.txt
//
#ifndef __ORF_test__AbstractTestExample__
#define __ORF_test__AbstractTestExample__

#include "ORForest.h"
#include <iomanip>

void GetMeanAndStd(std::vector<std::vector<double> > i_array, std::vector<double> * o_mean, std::vector<double> * o_std);

enum DataSetName{BIODEG, MUSK, CTG, WINE};
enum FeatureTypeName{GAUSSIAN, BENOULLI, MULTINOMIAL};

class AbstractTestExample{
public:
    AbstractTestExample();
    ~AbstractTestExample();
    bool LoadData(DataSetName data);
    void UpdateDataInfo();
    void GenerateTrainAndTestData();
    void SetTrainDataChunk(double startPercent, double endPercent, double increasePercent);
    void SetTrainDataOnline(int startIndex, int endIndex);
    virtual void Run(int MaxIter);
    virtual void PrintPerformance();
    void PrintDataInformation();
    std::vector<double> GetImbalanceRatio();
    int GetTrainN(){return trainN;};
protected:

    bool LoadCTGDataSet();
    bool LoadWineDataSet();
    bool LoadMuskDataSet();
    bool LoadBiodegDataSet();
    
    int featureN;
    int instanceN;
    int positiveN;
    int negtiveN;
    int trainN;
    int testN;
    int posNtrain;
    int negNtrain;
    int posNtest;
    int negNtest;
    std::shared_ptr<std::vector<std::shared_ptr<std::vector<double> > > > originData;
    std::shared_ptr<std::vector<std::shared_ptr<std::vector<double> > > > trainData;
    std::shared_ptr<std::vector<std::shared_ptr<std::vector<double> > > > testData;
    std::vector<int> trainIndexEachUpdate;
    
    std::vector<std::vector<double> > imbalanceRatio;
    
    std::vector<std::vector<double> > Sensitivity;
    std::vector<std::vector<double> > Specificity;
    std::vector<std::vector<double> > Gmean;
    std::vector<std::vector<double> > Time;
    
    std::vector<std::vector<double> > compareSensitivity;
    std::vector<std::vector<double> > compareSpecificity;
    std::vector<std::vector<double> > compareGmean;
    std::vector<std::vector<double> > compareTime;
    


    std::vector<FeatureTypeName> featureTypeList;
};

#endif /* defined(__ORF_test__AbstractTestExample__) */
