//
//  Node.h
//  ORF
//
//  Created by Guotai Wang on 01/12/2015.
//
// Copyright (c) 2014-2016 University College London, United Kingdom. All rights reserved.
// http://cmictig.cs.ucl.ac.uk
//
// Distributed under the BSD-3 licence. Please see the file licence.txt
//

#ifndef RF__Node__
#define RF__Node__

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <vector>
#include <iostream>
#include "ODTree.h"

namespace RandomForest {

template<typename T>
class ODTree;

template<typename T>
class Node
{
public:
    Node();
    ~Node();
    void binSplitDataSet(const std::shared_ptr<std::vector<int> > i_indexList, int feature,  T featureValue,
                         std::shared_ptr<std::vector<int> > o_indexList0, std::shared_ptr<std::vector<int> > o_indexList1);
    double meanLeaf();
    double impurityLeaf(const std::shared_ptr<std::vector<int> > i_sampleIndexList);
    double impurityLeafWithWeight(const std::shared_ptr<std::vector<int> > i_sampleIndexList, double * w1, double * w0);
    double impurityLeafWithWeight(const std::shared_ptr<std::vector<int> > i_sampleIndexList, double w1, double w0);

    void GetFeatureRange(int fIndex,T * min,T * max);
    void chooseBestSplit(int * o_bestFeatureIndex, T * o_bestFeatureValue, double * o_decreadedImpurity);
    void CreateTree();
    void UpdateTree(const std::shared_ptr<std::vector<int> > i_addSampleList);
    int UpdateTree(const std::shared_ptr<std::vector<int> > i_rmvSampleList, const std::shared_ptr<std::vector<int> > i_addSampleList);
    void GetSampleList(std::shared_ptr<std::vector<int> > o_posSampleList, std::shared_ptr<std::vector<int> > o_negSampleList);
        
    void SetLeft(Node<T> * l){left=l;};
    void SetRight(Node<T> *r ) {right=r;};
    void SetFeatureIndex(int idx){featureIndex=idx;};
    void SetSplitValue(double v){splitValue=v;};
    void SetDepth(int d){depth=d;};
    void SetTree(ODTree<T> * tr){tree=tr;};
    void SetSampleIndexList(std::shared_ptr<std::vector<int> > list){sampleIndexList=list;};
    
    Node<T> * GetLeft(){return left;};
    Node<T> * GetRight(){return right;};
    int GetFeatureIndex(){return featureIndex;};
    double GetSplitValue(){return splitValue;};
    int GetDepth(){return depth;};
    ODTree<T> * GetTree(){return tree;};
    std::shared_ptr<std::vector<int> > GetSampleIndexList(){return sampleIndexList;};
    
    void UpdateGiniImportance();
    double PredictOneSample(const std::shared_ptr<std::vector<T> > i_inData);
    void ConvertTreeToList(int * io_left, int * io_right,
                           int *io_splitFeature, double *io_splitValue,
                           int currentListIndex, int * io_globalListIndex);
private:
    Node<T> *left;
    Node<T> *right;
    int featureIndex;
    double splitValue;
    double decreasedImpurity;
    int depth;
    std::shared_ptr<std::vector<int> > sampleIndexList;
    ODTree<T> * tree;
};

}
#endif /* defined(RF__Node__) */
