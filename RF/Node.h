//
//  Node.h
//  ORF
//
//  Created by Guotai Wang on 01/12/2015.
//
//

#ifndef __RF__Node__
#define __RF__Node__

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <vector>
#include <iostream>
#include "ODTree.h"
using namespace std;


template<typename T>
class ODTree;

template<typename T>
class Node
{
public:
    Node();
    ~Node();
    void binSplitDataSet(const shared_ptr<vector<int> > i_indexList, int feature,  T featureValue,
                         shared_ptr<vector<int> > o_indexList0, shared_ptr<vector<int> > o_indexList1);
    double meanLeaf();
    double impurityLeaf(const shared_ptr<vector<int> > i_sampleIndexList);
    double impurityLeafWithWeight(const shared_ptr<vector<int> > i_sampleIndexList, double * w1, double * w0);
    double impurityLeafWithWeight(const shared_ptr<vector<int> > i_sampleIndexList, double w1, double w0);

    void GetFeatureRange(int fIndex,T * min,T * max);
    void chooseBestSplit(int * o_bestFeatureIndex, T * o_bestFeatureValue, double * o_decreadedImpurity);
    void CreateTree();
    void UpdateTree(const shared_ptr<vector<int> > i_addSampleList);
    int UpdateTree(const shared_ptr<vector<int> > i_rmvSampleList, const shared_ptr<vector<int> > i_addSampleList);
    void GetSampleList(shared_ptr<vector<int> > o_posSampleList, shared_ptr<vector<int> > o_negSampleList);
        
    void SetLeft(Node<T> * l){left=l;};
    void SetRight(Node<T> *r ) {right=r;};
    void SetFeatureIndex(int idx){featureIndex=idx;};
    void SetSplitValue(double v){splitValue=v;};
    void SetDepth(int d){depth=d;};
    void SetTree(ODTree<T> * tr){tree=tr;};
    void SetSampleIndexList(shared_ptr<vector<int> > list){sampleIndexList=list;};
    
    Node<T> * GetLeft(){return left;};
    Node<T> * GetRight(){return right;};
    int GetFeatureIndex(){return featureIndex;};
    double GetSplitValue(){return splitValue;};
    int GetDepth(){return depth;};
    ODTree<T> * GetTree(){return tree;};
    shared_ptr<vector<int> > GetSampleIndexList(){return sampleIndexList;};
    
    void UpdateGiniImportance();
    double PredictOneSample(const shared_ptr<vector<T> > i_inData);
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
    shared_ptr<vector<int> > sampleIndexList;
    ODTree<T> * tree;
};

#endif /* defined(__WeightedRF__Node__) */
