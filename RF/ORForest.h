#ifndef ORFOREST_H_
#define ORFOREST_H_
#include "ODTree.h"

template<typename T>
class ORForest
{
public:
	int treeNumber;
	int maxDepth;
	int leastNsample;
    ODTree<T> * trees;

    shared_ptr<vector<shared_ptr<vector<T> > > > trainData;
    shared_ptr<vector<shared_ptr<vector<T> > > > testData; // for training

    double testDataRatio;// how many percents of given data are used as test data (10% or 20%)
    bool useBalancedBagging;
    bool onlineUpdate;
public:
    ORForest();
	~ ORForest();
	void Init(int Ntree,int treeDepth, int leastNsampleForSplit);
    void Clear(); // delte trees and training data
    void UseBalancedBagging(bool b);
    void DisableOnlineUpdate(){onlineUpdate=false;};
    
    void Train(const shared_ptr<vector<shared_ptr<vector<T> > > > i_trainData);
    void Predict(const shared_ptr<vector<shared_ptr<vector<T> > > > i_testData, vector<float> ** o_predict);
    void PredictGPU(const T *i_testData, int i_Ns, int i_Nf, float *o_predict);
    
    int GetActureMaxTreeDepth();
    int GetActureMaxTreeNode();
    double GetAverageOOBE();
    double GetAverageBalancedOOBE();
    void GetRankedGiniImportance( shared_ptr<vector<int> > * featureIndexList,  shared_ptr<vector<double> > * giniImportanceList);
    void ConvertTreeToList(int * io_left, int * io_right, 
        int *io_splitFeature,double *io_splitValue,
        int maxNodeNumber);
protected:
    void GetTrainAndTestData(shared_ptr<vector<shared_ptr<vector<T> > > > i_trainData);
};


#endif
