/* ODTree.h
 *
 * Created by Guotai Wang on 01/12/2015.
 * Copyright (c) 2014-2016 University College London, United Kingdom. All rights reserved.
 * http://cmictig.cs.ucl.ac.uk
 *
 * Distributed under the BSD-3 licence. Please see the file licence.txt
 *
 * @file ODTree.h
 * @author Guotai Wang
 * @date  01/12/2015
 * @brief
 */

#ifndef ODTREE_H_
#define ODTREE_H_
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <vector>
#include <iostream>

namespace RandomForest {

    
enum BalanceType{SingleParameterBoostrap, MultipleParameterBoostrap, DynamicImbalanceAdaptableBootstrap};
enum SamplingType{DownSamplingMajority, UpSamplingMinority};

template<typename T>
class Node;

template<typename T>
    
/**
 * \brief ODTree
 *
 */
class ODTree
{
public:
	ODTree();
	~ODTree();
	
    void Train(const std::shared_ptr<std::vector<std::shared_ptr<std::vector<T> > > > i_trainData);
    void Predict(const std::shared_ptr<std::vector<std::shared_ptr<std::vector<T> > > > i_testData, std::vector<float> ** o_forecast);
    double GetOOBE(const std::shared_ptr<std::vector<std::shared_ptr<std::vector<T> > > > i_testData);
    double GetBalancedOOBE(const std::shared_ptr<std::vector<std::shared_ptr<std::vector<T> > > > i_testData);

    void ConvertTreeToList(int * io_left, int * io_right, 
        int *io_splitFeature, double *io_splitValue);
    
    Node<T> * Root(){return root;};
    void Reset();
    std::shared_ptr<std::vector<std::shared_ptr<std::vector<T> > > > TrainData(){return trainData;};
    std::shared_ptr<std::vector<double> > GiniImportance(){return giniImportance;};
    
    int GetActureTreeDepth(){return actureTreeDepth;};
    void SetActureTreeDepth(int d){actureTreeDepth=d;};
    
    int GetDepthUpperBound(){return depthUpperBound;};
    void SetDepthUpperBound(int d){depthUpperBound=d;};
    
    int GetActureTreeNode(){return actureTreeNode;};
    void SetActureTreeNode(int n){actureTreeNode=n;};
    
    double GetVarThreshold(){return varThreshold;};
    void SetVarTreshold(double t){varThreshold=t;};
    
    int GetSampleNumberThreshold(){return sampleNumberThreshold;};
    void SetSampleNumberThreshold(int n){sampleNumberThreshold=n;};
    
    void SetBalanceType(BalanceType type){balanceType=type;};
    BalanceType GetBalanceType(){return balanceType;};
    
    void SetSamplingType(SamplingType type){samplingType=type;};
    SamplingType GetSamplingType(){return samplingType;};
    
    void UpdateGiniImportance();
    
private:
    void UpdateTrainDataList(int oldNs);
    void GetPosNewAddedSampledList(int oldPosN,  double lambdaPos, std::shared_ptr<std::vector<int> > *o_list);
    void GetNegNewAddedSampledList(int oldNegN,  double lambdaNeg, std::shared_ptr<std::vector<int> > *o_list);
    void SingleParameterBoostrapSampling(int oldNs, std::shared_ptr<std::vector<int> > *o_list);
    void MultipleParameterBoostrapSampling(int oldN, std::shared_ptr<std::vector<int> > *o_list);
    void DynamicImbalanceAdaptiveBoostrapSampling(int oldNs, std::shared_ptr<std::vector<int> > * o_removeSampleList, std::shared_ptr<std::vector<int> > * o_addSampleList);

    /// Get a random number that belongs to the distribution of Pois(lambda)
    int GetPossionNumber(double lambda);
    
    /// Bootsttrap sampling Ns samples. each sample is sampled k times, where k belongs to Pois(possonLambda).
    void BoostrapSampling(double possionLambda, int Ns, double bagFactor, std::vector<int> *o_list);
    
    Node<T> *root;
    std::shared_ptr<std::vector<std::shared_ptr<std::vector<T> > > > trainData;
    
    int actureTreeDepth;
    int actureTreeNode;
    
    int depthUpperBound;
    double varThreshold;
    int sampleNumberThreshold;
    
    //0, no balanced bagging, just use a boostrap with same lambda for two classes
    //1, boostrap sampling with different lambda for two classes
    //2, sampling with remove list
    BalanceType balanceType;
    SamplingType samplingType;
    double subDataSetRatio;
    double oldPosLambda;
    double oldNegLambda;
    std::shared_ptr<std::vector<int> > posTrainDataList; // index list for all the postive training data
    std::shared_ptr<std::vector<int> > negTrainDataList; // index list for all the negtive training data
    std::shared_ptr<std::vector<int> > posSampledList;   // index list for all the sampled postive training data
    std::shared_ptr<std::vector<int> > negSampledList;   // index list for all the sampled negtive training data
    std::shared_ptr<std::vector<double> > giniImportance;
};
}

#endif
