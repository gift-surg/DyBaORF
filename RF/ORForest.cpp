/*
 * ORForest.cpp
 *
 *  Created on: 17 Mar 2015
 *      Author: guotaiwang
 */

#include "ORForest.h"
#include "Node.h"
#include <iostream>
#include <ctime>
using namespace std;

template<typename T>
ORForest<T>::ORForest()
{
	treeNumber=20;
	maxDepth=10;
	leastNsample=10;
	trees=NULL;
    trainData=nullptr;;
    testData=nullptr;
    testDataRatio=0.2;
    useBalancedBagging=false;
    onlineUpdate=true;
};

template<typename T>
ORForest<T>::~ORForest()
{
    Clear();
}

template<typename T>
void ORForest<T>::Init(int Ntree,int treeDepth, int leastNsampleForSplit)
{
	treeNumber=Ntree;
	maxDepth=treeDepth;
	leastNsample=leastNsampleForSplit;
}

template<typename T>
void ORForest<T>::UseBalancedBagging(bool b)
{
    useBalancedBagging=b;
}
template<typename T>
void ORForest<T>::Clear()
{
    if(trees)
    {
        delete [] trees;
        trees=NULL;
    }
    if(trainData)
    {
        trainData.reset();
        trainData=nullptr;;
    }
    if(testData)
    {
        testData.reset();
        trainData=nullptr;
    }
}



template<typename T>
void ORForest<T>::GetTrainAndTestData(const shared_ptr<vector<shared_ptr<vector<T> > > > i_trainData)
{
    int i_addNs=i_trainData->size();
    int posNadd=0, negNadd=0;
    for(int i=0;i<i_addNs;i++)
    {
        T tempLabel=i_trainData->at(i)->back();
        if(tempLabel==1)
        {
            posNadd++;
        }
        else
        {
            negNadd++;
        }
    }
    if(trainData==nullptr && (posNadd==0 || negNadd==0))
    {
        printf("addNs %d, posNadd %d, negNadd %d, the class number of training data shoule be 2!\n", i_addNs, posNadd, negNadd);
        return;
    }

    int addNTest=i_addNs*testDataRatio;
    int addNTrain=i_addNs-addNTest;
    
    // select train and test data randomly
    vector<bool> TestDataMask;
    for(int i=0;i<i_addNs;i++) TestDataMask.push_back(false);

    int add_nTest=0;
    while(add_nTest<addNTest)
    {
        double randf=(double)rand()/RAND_MAX;
        int tempIdx=i_addNs*randf;
        if(TestDataMask[tempIdx]==false)
        {
            TestDataMask[tempIdx]=true;
            add_nTest++;
        }
    }

    shared_ptr<vector<shared_ptr<vector<T> > > > addTrainData(new vector<shared_ptr<vector<T> > >);
    shared_ptr<vector<shared_ptr<vector<T> > > > addTestData(new vector<shared_ptr<vector<T> > >);
    addTrainData->reserve(addNTrain);
    addTestData->reserve(addNTest);
    for(int i=0;i<i_addNs;i++)
    {
        if(TestDataMask[i])
        {
            addTestData->push_back(i_trainData->at(i));
        }
        else{
            addTrainData->push_back(i_trainData->at(i));
        }
    }
    
    shared_ptr<vector<shared_ptr<vector<T> > > > newTrainData(new vector<shared_ptr<vector<T> > >);
    shared_ptr<vector<shared_ptr<vector<T> > > > newTestData(new vector<shared_ptr<vector<T> > >);
    if(trainData==nullptr)
    {
        newTrainData=addTrainData;
        newTestData=addTestData;
    }
    else{
        newTrainData->insert(newTrainData->end(), trainData->begin(), trainData->end());
        newTrainData->insert(newTrainData->end(),addTrainData->begin(), addTrainData->end());
        
        newTestData->insert(newTestData->end(), testData->begin(), testData->end());
        newTestData->insert(newTestData->end(), addTestData->begin(), addTestData->end());
    }
    trainData=newTrainData;
    testData=newTestData;
 }


template<typename T>
void ORForest<T>::Train(const shared_ptr<vector<shared_ptr<vector<T> > > > i_trainData)
{
	if(trees==nullptr || onlineUpdate==false)
	{
        GetTrainAndTestData(i_trainData);
//        int newNs=trainData->size();
        if(trees)
        {
            delete [] trees;
        }
        trees=new ODTree<T>[treeNumber];
		for(int i=0;i<treeNumber;i++)
		{
			trees[i].SetDepthUpperBound(maxDepth);
            trees[i].SetSampleNumberThreshold(leastNsample);
            trees[i].UseBalancedBagging(useBalancedBagging);
			trees[i].Train(trainData);
		}
	}
	else
	{
        // combine old training data and newly arrived training data
        int oldNs=trainData->size();
        GetTrainAndTestData(i_trainData);
        int newNs=trainData->size();
		for(int i=0;i<treeNumber;i++)
		{
			trees[i].Train(trainData);
			T oobe=trees[i].GetOOBE(testData);
			if(oobe>0.4)
			{
                trees[i].Reset();
				trees[i].Train(trainData);
			}
		}
	}
}

template<typename T>
void ORForest<T>::Predict(const shared_ptr<vector<shared_ptr<vector<T> > > > i_testData, vector<float>  ** o_predict)
{
    vector<float> *sumPredict=new vector<float>;
    sumPredict->resize(i_testData->size());
    for(int i=0;i<i_testData->size();i++) sumPredict->at(i)=0;
    for(int i=0;i<treeNumber;i++)
    {
        vector<float> * tempPredict;
        trees[i].Predict(i_testData, &tempPredict);
        for(int j=0;j<sumPredict->size();j++)
        {
            sumPredict->at(j)+=tempPredict->at(j);
        }
    }
    for(int i=0;i<sumPredict->size();i++)
    {
        sumPredict->at(i)=sumPredict->at(i)/treeNumber;
    }
    *o_predict=sumPredict;
}

template<typename T>
void ORForest<T>::PredictGPU(const T *i_testData, int i_Ns, int i_Nf, float *o_predict)
{
//    int d=GetActureMaxTreeDepth();
//    int nodeN=GetActureMaxTreeNode();
//    int *leftList=new int [nodeN*treeNumber];
//    int *rightList=new int [nodeN*treeNumber];
//    int *splitFeatureList=new int [nodeN*treeNumber];
//    double *splitValueList=new double [nodeN*treeNumber];
//    time_t start=clock();
//    ConvertTreeToList(leftList, rightList, splitFeatureList, splitValueList, nodeN);
//    double during=((double)(clock()-start))/CLOCKS_PER_SEC;
//    cout<<"tree converting time "<<during<<"s"<<endl;
//    ForestPredict<double>(leftList,
//                         rightList,
//                         splitFeatureList,
//                         splitValueList,
//                         treeNumber, nodeN,
//                         (double *)i_testData,
//                         i_Ns, i_Nf,
//                         o_predict);
}

template<typename T>
int ORForest<T>::GetActureMaxTreeDepth()
{
    int tempD=0;
    for(int i=0;i<treeNumber;i++)
    {
        if(tempD < trees[i].GetActureTreeDepth())
        {
            tempD = trees[i].GetActureTreeDepth();
        }
    }
    return tempD;
}

template<typename T>
double ORForest<T>::GetAverageOOBE()
{
    double oobe=0;
    for(int i=0;i<treeNumber;i++)
    {
        oobe+=trees[i].GetOOBE(testData);
    }
    return oobe/treeNumber;
}

template<typename T>
double ORForest<T>::GetAverageBalancedOOBE()
{
    double balancedOobe=0;
    for(int i=0;i<treeNumber;i++)
    {
        balancedOobe+=trees[i].GetBalancedOOBE(testData);
    }
    return balancedOobe/treeNumber;
}
template<typename T>
int ORForest<T>::GetActureMaxTreeNode()
{
    int n=0;
    for(int i=0;i<treeNumber;i++)
    {
        if(n<trees[i].GetActureTreeNode())
        {
            n=trees[i].GetActureTreeNode();
        }
    }
    return n;
}

template<typename T>
void ORForest<T>::GetRankedGiniImportance( shared_ptr<vector<int> > * o_featureIndexList,  shared_ptr<vector<double> > * o_giniImportanceList)
{
    shared_ptr<vector<int> > featureIndexList=* o_featureIndexList;
    shared_ptr<vector<double> > giniImportanceList=* o_giniImportanceList;
    featureIndexList->resize(trainData->at(0)->size()-1);
    giniImportanceList->resize(trainData->at(0)->size()-1);
    for(int i=0;i<featureIndexList->size();i++)
    {
        featureIndexList->at(i)=i;
        giniImportanceList->at(i)=0.0;
    }
    for(int i=0;i<treeNumber;i++)
    {
        trees[i].UpdateGiniImportance();
        for(int j=0;j<featureIndexList->size();j++)
        {
            giniImportanceList->at(j)=giniImportanceList->at(j)+trees[i].GiniImportance()->at(j);
        }
    }
    
    // sorting
    for(int i=0;i<featureIndexList->size()-1;i++)
    {
        for(int j=i+1;j<featureIndexList->size();j++)
        {
            double leftValue=giniImportanceList->at(i);
            double rightValue=giniImportanceList->at(j);
            if(leftValue<rightValue)
            {
                int leftIndex=featureIndexList->at(i);
                int rightIndex=featureIndexList->at(j);
                giniImportanceList->at(i)=rightValue;
                giniImportanceList->at(j)=leftValue;
                featureIndexList->at(i)=rightIndex;
                featureIndexList->at(j)=leftIndex;
            }
        }
    }
    
    *o_featureIndexList=featureIndexList;
    *o_giniImportanceList=giniImportanceList;
}

template<typename T>
void ORForest<T>::ConvertTreeToList(int * io_left, int * io_right,
        int *io_splitFeature, double *io_splitValue,
        int maxNodeNumber)
{
	for(int i=0;i<treeNumber;i++)
	{
        trees[i].ConvertTreeToList(io_left+i*maxNodeNumber, io_right+i*maxNodeNumber,
        	io_splitFeature+i*maxNodeNumber, io_splitValue+i*maxNodeNumber);
	}
}


template class ORForest<float>;
template class ORForest<double>;