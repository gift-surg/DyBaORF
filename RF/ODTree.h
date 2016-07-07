/*=========================================================================
 Program:   DyBa ORF
 Module:    ODTree.h
 
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
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <vector>
#include <iostream>

namespace RandomForest {

/** \brief enum BalanceType
 *
 * The type of method used to deal with imbalanced trainging data
 */
enum BalanceType{
    /** use a single parameter of Possion distribution to sample both positive and negative samples*/
    SingleParameterBoostrap,
    /** use multiple Possion distributions to sample both positive and negative samples*/
    MultipleParameterBoostrap,
    /** use multiple Possion distributions to sample both positive and negative samples
     *  and also consider the data imbalance ratio change
     */
    DynamicImbalanceAdaptableBootstrap};
    
/** \brief enum SamplingType
 *
 * The type of sampling method
 */
enum SamplingType{
    /** down dample the majority class */
    DownSamplingMajority,
    /** up sample the minority class */
    UpSamplingMinority};

template<typename T>
class Node;

template<typename T>
    
/**
 * \brief Class ODTree
 *
 * During training, the nodes of the tree are generated recucively.
 * During testing, the test samples are propagated from the root of the
 * tree to its leafs for inference. 
 * For online training, the tree may be growed and shrinked based on the new training data.
 */
class ODTree
{
public:
    /** Construction function */
	ODTree();
    
    /** Deconstruction function */
	~ODTree();
    
    /** Delete all the nodes of that tree */
	void Reset();
    
    /** Construct the tree based on training data
     * @param[in] i_trainData the input training data
     */
    void Train(const std::shared_ptr<std::vector<std::shared_ptr<std::vector<T> > > > i_trainData);
    
    /** Predict the probability of the foreground for a list of test data
     * @param[in] i_testData the input test data list
     * @param[out] o_forecast the predicted probability for each test sample
     */
    void Predict(const std::shared_ptr<std::vector<std::shared_ptr<std::vector<T> > > > i_testData,
                 std::vector<float> ** o_forecast);
    
    /** Get the out of bag error
     *
     * @param[in] i_testData the input test data with label to calculate the prediction correct rate.
     */
    double GetOOBE(const std::shared_ptr<std::vector<std::shared_ptr<std::vector<T> > > > i_testData) const;
    
    /** Get the training data set of this tree */
    std::shared_ptr<std::vector<std::shared_ptr<std::vector<T> > > > GetTrainData() const;
    
    /** Set the acture tree depth
     * @param[in] d the input tree depth
     */
    void SetActureTreeDepth(int d);
    
    /** Get the acture tree depth */
    int GetActureTreeDepth() const;
    
    /** Set the upper bound of tree depth
     * @param[in] d the input depth
     */
    void SetDepthUpperBound(int d);
    
    /** Get the upper bound of tree depth */
    int GetDepthUpperBound() const;
    
    /** Set acture number of tree nodes
     * @param[in] n the input number of tree nodes
     */
    void SetActureTreeNode(int n);
    
    /** Get the acture number of tree nodes */
    int GetActureTreeNode() const;
    
    /** Set the variance threshold used for split test.
     * If the variance change after a potential split is less than \p t, then the split test fails.
     * @param[in] t the input threshold
     */
    void SetVarTreshold(double t);
    
    /** Get the variance threshold used for split test */
    double GetVarThreshold() const;
    
    /** Set the sample number threshold for split test.
     * If the sample number of a node is less than \p n, then the split test fails.
     * param[in] t the input sample number.
     */
    void SetSampleNumberThreshold(int n);
    
    /** Get the sample number of threshold */
    int GetSampleNumberThreshold() const;
    
    /** Set the balance type
     * @param[in] type the input balance type
     */
    void SetBalanceType(BalanceType type);
    
    /** Get he balance type */
    BalanceType GetBalanceType() const;
    
    /** Set the sampling type 
     * @param[in] type the input sampling type 
     */
    void SetSamplingType(SamplingType type);
    
    /** Get the sampling type */
    SamplingType GetSamplingType() const;
    
    /** Update the gini importance of each feature */
    void UpdateGiniImportance();
    
    /** Get the gini importance of each feature 
     * @return a vector storing the gini importance of each feature
     */
    std::shared_ptr<std::vector<double> > GetGiniImportance() const;
    
private:
    /** Update positive and negative train data list.
     * @param[in] oldNs the previous number of samples
     */
    void UpdateTrainDataList(int oldNs);
    
    /** Get the list of newly added positive samples
     * @param[in] oldPosN the old number of positive samples
     * @param[in] lambdaPos the previous lambda used for sampling positive samples
     * @param[out] o_list the output vector that storing the newly added positive samples
     */
    void GetPosNewAddedSampledList(int oldPosN,  double lambdaPos,
                                   std::shared_ptr<std::vector<int> > *o_list);
    
    /** Get the list of newly added negative samples
     * @param[in] oldNegN the old number of negative samples
     * @param[in] lambdaNeg the previous lambda used for sampling negative samples
     * @param[out] o_list the output vector that storing the newly added negative samples
     */
    void GetNegNewAddedSampledList(int oldNegN,
                                   double lambdaNeg,
                                   std::shared_ptr<std::vector<int> > *o_list);
    
    /**  Use single parameter boostrap sampling to obtain the training samples for one tree
     * @param[in] onldNs the previous number of training samples
     * @param[out] o_list the output list storing the sampled training data
     */
    void SingleParameterBoostrapSampling(int oldNs, std::shared_ptr<std::vector<int> > *o_list);
    
    /**  Use multiple parameter boostrap sampling to obtain the training samples for one tree
     * @param[in] onldNs the previous number of training samples
     * @param[out] o_list the output list storing the sampled training data
     */
    void MultipleParameterBoostrapSampling(int oldN, std::shared_ptr<std::vector<int> > *o_list);
    
    /**  Use dynamically imbalance adaptive boostrap sampling to obtain the training samples for one tree
     * @param[in] onldNs the previous number of training samples
     * @param[out] o_removeSampleList the output list storing the training data that should be removed
     * @param[out] o_addSampleList the output list storing the training data that should be added
     */
    void DynamicImbalanceAdaptiveBoostrapSampling(int oldNs,
                                                  std::shared_ptr<std::vector<int> > * o_removeSampleList,
                                                  std::shared_ptr<std::vector<int> > * o_addSampleList);

    /** Get a random number that belongs to the distribution of Pois(lambda)
     * @param[in] lambda the parameter for Possion distribution
     * @return a integer number than belongs to Pos(lambda)
     */
    int GetPossionNumber(double lambda);
    
    /** Bootsttrap sampling Ns samples. each sample is sampled k times, where k belongs to Pois(possonLambda).
     * @param[in] possionLambda the parameter for Possion distribution
     * @param[in] Ns the number of input samples for Boostrap sampling
     * @param[in] bagFactor a float number between 0 and 1 to control the sampling rate. (default is 1）
     * @param[out] o_list an output list containing the sampled numbers
     */
    void BoostrapSampling(double possionLambda,
                          int Ns,
                          double bagFactor,
                          std::vector<int> *o_list);

private:
    
    std::shared_ptr<Node<T> > root;
    std::shared_ptr<std::vector<std::shared_ptr<std::vector<T> > > > trainData;
    
    int actureTreeDepth;
    int actureTreeNode;
    
    int depthUpperBound;
    double varThreshold;
    int sampleNumberThreshold;
    
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

