//
//  ORForest.h
//
//  Created on: 17 Mar 2015
//      Author: Guotai Wang
//
// Copyright (c) 2014-2016 University College London, United Kingdom. All rights reserved.
// http://cmictig.cs.ucl.ac.uk
//
// Distributed under the BSD-3 licence. Please see the file licence.txt
//

#ifndef ORFOREST_H_
#define ORFOREST_H_
#include "ODTree.h"

namespace RandomForest {

template<typename T>
class ORForest
{
public:
	int treeNumber;
	int maxDepth;
	int leastNsample;
    ODTree<T> * trees;

    std::shared_ptr<std::vector<std::shared_ptr<std::vector<T> > > > trainData;
    std::shared_ptr<std::vector<std::shared_ptr<std::vector<T> > > > testData; // for training

    double testDataRatio;// how many percents of given data are used as test data (10% or 20%)
    BalanceType balanceType;
    SamplingType samplingType;
    bool onlineUpdate;
public:
    ORForest();
	~ ORForest();
	void Init(int Ntree,int treeDepth, int leastNsampleForSplit);
    void Clear(); // delte trees and training data
    void SetBalanceType(BalanceType type){balanceType=type;};
    BalanceType GetBalanceType(){return balanceType;};
    
    void SetSamplingType(SamplingType type){samplingType=type;};
    SamplingType GetSamplingType(){return samplingType;};
    void DisableOnlineUpdate(){onlineUpdate=false;};
    
    void Train(const std::shared_ptr<std::vector<std::shared_ptr<std::vector<T> > > > i_trainData);
    void Train(const T *i_trainData, int i_Ns, int i_Nfp1);
    void Predict(const std::shared_ptr<std::vector<std::shared_ptr<std::vector<T> > > > i_testData, std::vector<float> ** o_predict);
    
    
    int GetActureMaxTreeDepth();
    int GetActureMaxTreeNode();
    double GetAverageOOBE();
    double GetAverageBalancedOOBE();
    void GetRankedGiniImportance( std::shared_ptr<std::vector<int> > * featureIndexList,  std::shared_ptr<std::vector<double> > * giniImportanceList);
    void ConvertTreeToList(int * io_left, int * io_right, 
        int *io_splitFeature,double *io_splitValue,
        int maxNodeNumber);
protected:
    void GetTrainAndTestData(std::shared_ptr<std::vector<std::shared_ptr<std::vector<T> > > > i_trainData);
};
}

#endif
