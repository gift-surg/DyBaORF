/*=========================================================================
 Program:   DyBa ORF
 Module:    ORForest.h
 
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
#include "ODTree.h"

namespace RandomForest {

/** \brief Class ORForest
 *
 * A random forest contains a number of trees and each tree is trained speratively.
 */
template<typename T>
class ORForest
{
public:
    /** Construction function */
    ORForest();
    
    /** Deconstruction fuction */
	~ ORForest();
    
    /** Init function 
     * @param[in] Ntree number of trees
     * @param[in] treeDepth the maximal depth allowded for each tree
     * @param[in] leastNsampleForSplit the least number of samples for split test
     */
	void Init(int Ntree,int treeDepth, int leastNsampleForSplit);
    
    /** Delete trees and training data */
    void Clear();
    
    /** Set balance type
     * @param[in] type the input balance type
     */
    void SetBalanceType(BalanceType type);
    
    /** Get balance type*/
    BalanceType GetBalanceType() const;

    /** Set sampling type 
     * @param[in] type the input sampling type
     */
    void SetSamplingType(SamplingType type);
    
    /** Get sampling type */
    SamplingType GetSamplingType() const;
    
    /** Disable online update. Train from scratch when new data arrives */
    void DisableOnlineUpdate();

    /** Get the data for train and validate 
     * @param[in] i_trainData the entire input dataset
     */
    void GetTrainAndTestData(std::shared_ptr<std::vector<std::shared_ptr<std::vector<T> > > > i_trainData);
    
    /** Train with the training data
     * @param[in] i_trainData the input training data set
     */
    void Train(const std::shared_ptr<std::vector<std::shared_ptr<std::vector<T> > > > i_trainData);
    
    /** Train with the training data, a variance of #Train(const std::shared_ptr<std::vector<std::shared_ptr<std::vector<T> > > > i_trainData)
     * @param[in] i_trainData a pointer pointing to the data
     * @param[in] i_Ns the number of samples
     * @param[in] i_Nfp1 the number of features plus 1
     */
    void Train(const T *i_trainData, int i_Ns, int i_Nfp1);
    
    /** Predict
     * @param[in] i_testData the input test data set
     * @param[out] o_predict the prediction result of the test data
     */
    void Predict(const std::shared_ptr<std::vector<std::shared_ptr<std::vector<T> > > >& i_testData, std::vector<float> ** o_predict);
    
    /** Get acture max tree deepth */
    int GetActureMaxTreeDepth() const;
    
    /** Get acture max tree nodes */
    int GetActureMaxTreeNode() const;
    
    /** Get average OOBE across all the trees */
    double GetAverageOOBE() const;
    
    /** Get racnked Gini importance of all the features 
     * @param[out] featureIndexList a list of feature index with the one with the highest importance going first
     * @param[out] giniImportanceList a list of gini importance with the highest value going first
     */
    void GetRankedGiniImportance(std::shared_ptr<std::vector<int> > * featureIndexList,
                                 std::shared_ptr<std::vector<double> > * giniImportanceList);

private:
    int treeNumber;
    int maxDepth;
    int leastNsample;
    BalanceType balanceType;
    SamplingType samplingType;
    bool onlineUpdate;
    double testDataRatio; /// how many percents of given data are used as test data (10% or 20%)
    
    typedef std::vector< ODTree<T> > TreeList;
    typedef typename TreeList::iterator TreeListIterator;
    typedef typename TreeList::const_iterator TreeListConstIterator;
    TreeList trees;
    std::shared_ptr<std::vector<std::shared_ptr<std::vector<T> > > > trainData;
    std::shared_ptr<std::vector<std::shared_ptr<std::vector<T> > > > testData; // for validating
};
}
