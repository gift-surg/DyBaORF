#ifndef ODTREE_H_
#define ODTREE_H_
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <vector>
#include <iostream>

using namespace std;


template<typename T>
class Node;

template<typename T>
class ODTree
{
public:
	ODTree();
	~ODTree();
	
    void Train(const shared_ptr<vector<shared_ptr<vector<T> > > > i_trainData);
    void Predict(const shared_ptr<vector<shared_ptr<vector<T> > > > i_testData, vector<float> ** o_forecast);
    double GetOOBE(const shared_ptr<vector<shared_ptr<vector<T> > > > i_testData);
    double GetBalancedOOBE(const shared_ptr<vector<shared_ptr<vector<T> > > > i_testData);

    void ConvertTreeToList(int * io_left, int * io_right, 
        int *io_splitFeature, double *io_splitValue);
    
    Node<T> * Root(){return root;};
    void Reset();
    shared_ptr<vector<shared_ptr<vector<T> > > > TrainData(){return trainData;};
    shared_ptr<vector<double> > GiniImportance(){return giniImportance;};
    
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
    
    void UseBalancedBagging(bool b){useBalancedBagging=b;};
    bool UseBalancedBagging(){return useBalancedBagging;};
    
    void UpdateGiniImportance();
    
private:
    void BalancedBagging(shared_ptr<vector<int> > o_indexList);
    void GetUpdateSampleList(int oldNs, shared_ptr<vector<int> > * o_removeSampleList, shared_ptr<vector<int> > * o_addSampleList);
    
    Node<T> *root;
    shared_ptr<vector<shared_ptr<vector<T> > > > trainData;
    
    int actureTreeDepth;
    int actureTreeNode;
    
    int depthUpperBound;
    double varThreshold;
    int sampleNumberThreshold;
    
    bool useBalancedBagging;
    double subDataSetRatio;
    shared_ptr<vector<int> > posTrainDataList; // index list for all the postive training data
    shared_ptr<vector<int> > negTrainDataList; // index list for all the negtive training data
    shared_ptr<vector<int> > posSampledList;   // index list for all the sampled postive training data
    shared_ptr<vector<int> > negSampledList;   // index list for all the sampled negtive training data
    shared_ptr<vector<double> > giniImportance;
};


#endif
