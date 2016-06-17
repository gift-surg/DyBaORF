//
// ODTree.cpp
//
// Created by Guotai Wang on 01/12/2015.
//
// Copyright (c) 2014-2016 University College London, United Kingdom. All rights reserved.
// http://cmictig.cs.ucl.ac.uk
//
// Distributed under the BSD-3 licence. Please see the file licence.txt
//
#include "ODTree.h"
#include "Node.h"

template<typename T>
RandomForest::ODTree<T>::ODTree()
{
    root=nullptr;
    trainData=nullptr;
    depthUpperBound=10;
    varThreshold=0.02;
    sampleNumberThreshold=10;
    actureTreeDepth=0;
    actureTreeNode=0;
    balanceType=SingleParameterBoostrap;
    samplingType=DownSamplingMajority;
    subDataSetRatio=1.0;
    srand(time(0));
    Reset();
}

template<typename T>
RandomForest::ODTree<T>::~ODTree()
{
    if(root) delete root;
    if(posTrainDataList) posTrainDataList.reset();
    if(negSampledList) negTrainDataList.reset();
    if(posSampledList) posSampledList.reset();
    if(negSampledList) negSampledList.reset();
    if(giniImportance) giniImportance.reset();
}

template<typename T>
int RandomForest::ODTree<T>::GetPossionNumber(double lambda)
{
    double L=exp(-lambda);
    int k=0;
    double p=1;
    do{
        k=k+1;
        double u=(double)rand()/RAND_MAX;
        p=p*u;
    }
    while(p>L);
    k=k-1;
    return k;
}

template<typename T>
void RandomForest::ODTree<T>::BoostrapSampling(double possionLambda, int Ns, double bagFactor, vector<int> *o_list)
{
    o_list->reserve(Ns*bagFactor);
    for(int i=0;i<Ns;i++)
    {
        double randNumber=(double)rand()/RAND_MAX;
        if(randNumber>bagFactor)continue;
        int k=GetPossionNumber(possionLambda);
        for(int j=0;j<k;j++)
        {
            o_list->push_back(i);
        }
    }
}

template<typename T>
void RandomForest::ODTree<T>::Reset()
{
    if(root)
    {
        delete root;
        root=nullptr;
    }
    trainData=nullptr;
    posTrainDataList=make_shared<vector<int> >();
    negTrainDataList=make_shared<vector<int> >();
    posSampledList=make_shared<vector<int> >();
    negSampledList=make_shared<vector<int> >();
    giniImportance=make_shared<vector<double> >();
    oldPosLambda=1.0;
    oldNegLambda=1.0;
}

template<typename T>
void RandomForest::ODTree<T>::UpdateTrainDataList(int oldNs)
{
    shared_ptr<vector<int> > addPosTrainDataList(new vector<int>);
    shared_ptr<vector<int> > addNegTrainDataList(new vector<int>);
    addPosTrainDataList->reserve(trainData->size()-oldNs);
    addNegTrainDataList->reserve(trainData->size()-oldNs);
    for(int i=oldNs;i<trainData->size();i++)
    {
        T tempLabel=trainData->at(i)->back();
        if(tempLabel>0)
        {
            addPosTrainDataList->push_back(i);
        }
        else{
            addNegTrainDataList->push_back(i);
        }
    }
    posTrainDataList->insert(posTrainDataList->end(), addPosTrainDataList->begin(),addPosTrainDataList->end());
    negTrainDataList->insert(negTrainDataList->end(), addNegTrainDataList->begin(),addNegTrainDataList->end());
}

template<typename T>
void RandomForest::ODTree<T>::GetPosNewAddedSampledList(int oldPosN, double lambdaPos,shared_ptr<vector<int> > *o_list)
{
    shared_ptr<vector<int> > tempPosSampledList(new vector<int>);
    int newPosN=posTrainDataList->size();
    if(newPosN>oldPosN)
    {
        vector<int> posBootstrapList;
        BoostrapSampling(lambdaPos, newPosN-oldPosN, 1.0, &posBootstrapList);
        tempPosSampledList->resize(posBootstrapList.size());
        for(int i=0;i<posBootstrapList.size();i++)
        {
            tempPosSampledList->at(i)=posTrainDataList->at(oldPosN+posBootstrapList[i]);
        }
    }
    *o_list=tempPosSampledList;
}

template<typename T>
void RandomForest::ODTree<T>::GetNegNewAddedSampledList(int oldNegN, double lambdaNeg, shared_ptr<vector<int> > *o_list)
{
    shared_ptr<vector<int> > tempNegSampledList(new vector<int>);
    int newNegN=negTrainDataList->size();
    if(newNegN>oldNegN)
    {
        vector<int> negBootstrapList;
        BoostrapSampling(lambdaNeg, newNegN-oldNegN, 1.0, &negBootstrapList);
        tempNegSampledList->resize(negBootstrapList.size());
        for(int i=0;i<negBootstrapList.size();i++)
        {
            tempNegSampledList->at(i)=negTrainDataList->at(oldNegN+negBootstrapList[i]);
        }
    }
    *o_list=tempNegSampledList;
}

template<typename T>
void RandomForest::ODTree<T>::SingleParameterBoostrapSampling(int oldNs, shared_ptr<vector<int> > *o_list)
{
    int oldPosN=posTrainDataList->size();
    int oldNegN=negTrainDataList->size();
    UpdateTrainDataList(oldNs);
    int newPosN=posTrainDataList->size();
    int newNegN=negTrainDataList->size();
    
    shared_ptr<vector<int> > outputSampledList(new vector<int>);
    if(newPosN>oldPosN)
    {
        shared_ptr<vector<int> > tempPosSampledList;
        GetPosNewAddedSampledList(oldPosN, 1.0, &tempPosSampledList);
        posSampledList->insert(posSampledList->end(), tempPosSampledList->begin(), tempPosSampledList->end());
        outputSampledList->insert(outputSampledList->end(), tempPosSampledList->begin(), tempPosSampledList->end());
    }
    if(newNegN>oldNegN)
    {
        shared_ptr<vector<int> > tempNegSampledList;
        GetNegNewAddedSampledList(oldNegN, 1.0, &tempNegSampledList);
        negSampledList->insert(negSampledList->end(), tempNegSampledList->begin(), tempNegSampledList->end());
        outputSampledList->insert(outputSampledList->end(), tempNegSampledList->begin(), tempNegSampledList->end());
    }
    *o_list=outputSampledList;
}

template<typename T>
void RandomForest::ODTree<T>::MultipleParameterBoostrapSampling(int oldNs, shared_ptr<vector<int> > *o_list)
{
    int oldPosN=posTrainDataList->size();
    int oldNegN=negTrainDataList->size();
    UpdateTrainDataList(oldNs);
    int newPosN=posTrainDataList->size();
    int newNegN=negTrainDataList->size();

    double imbalanceRatio=(double)newNegN/newPosN;
    double posLambda, negLambda;
    if(samplingType==DownSamplingMajority)
    {
        posLambda=imbalanceRatio>=1.0 ? 1.0:imbalanceRatio;
        negLambda=imbalanceRatio>=1.0 ? 1.0/imbalanceRatio:1.0;
    }
    else{
        posLambda=imbalanceRatio>=1.0 ? imbalanceRatio:1.0;
        negLambda=imbalanceRatio>=1.0 ? 1.0:1.0/imbalanceRatio;
    }
    
    shared_ptr<vector<int> > outputSampledList(new vector<int>);
    if(newPosN>oldPosN)
    {
        shared_ptr<vector<int> > tempPosSampledList;
        GetPosNewAddedSampledList(oldPosN, posLambda, &tempPosSampledList);
        posSampledList->insert(posSampledList->end(), tempPosSampledList->begin(), tempPosSampledList->end());
        outputSampledList->insert(outputSampledList->end(), tempPosSampledList->begin(), tempPosSampledList->end());
    }
    if(newNegN>oldNegN)
    {
        shared_ptr<vector<int> > tempNegSampledList;
        GetNegNewAddedSampledList(oldNegN, negLambda, &tempNegSampledList);
        negSampledList->insert(negSampledList->end(), tempNegSampledList->begin(), tempNegSampledList->end());
        outputSampledList->insert(outputSampledList->end(), tempNegSampledList->begin(), tempNegSampledList->end());
    }
    *o_list=outputSampledList;
}

template<typename T>
void RandomForest::ODTree<T>::DynamicImbalanceAdaptiveBoostrapSampling(int oldNs, shared_ptr<vector<int> > * o_removeSampleList, shared_ptr<vector<int> > * o_addSampleList)
{
    int oldPosN=posTrainDataList->size();
    int oldNegN=negTrainDataList->size();
    UpdateTrainDataList(oldNs);
    int newPosN=posTrainDataList->size();
    int newNegN=negTrainDataList->size();
    
    
    double newImbalanceRatio=(double)newNegN/newPosN;
    double newPosLambda,newNegLambda;
    if(samplingType==DownSamplingMajority)
    {
        newPosLambda=newImbalanceRatio>=1.0 ? 1.0:newImbalanceRatio;
        newNegLambda=newImbalanceRatio>=1.0 ? 1.0/newImbalanceRatio:1.0;
    }
    else{
        newPosLambda=newImbalanceRatio>=1.0 ? newImbalanceRatio:1.0;
        newNegLambda=newImbalanceRatio>=1.0 ? 1.0:1.0/newImbalanceRatio;
    }
    
    shared_ptr<vector<int> > addSampledList(new vector<int>);
    shared_ptr<vector<int> > addSampledPosList(new vector<int>);
    shared_ptr<vector<int> > addSampledNegList(new vector<int>);
    shared_ptr<vector<int> > rmvSampledList(new vector<int>);
    shared_ptr<vector<int> > rmvSampledPosList(new vector<int>);
    shared_ptr<vector<int> > rmvSampledNegList(new vector<int>);
    
    // sample new arrived data with newPosLambda and newNegLambda
    if(newPosN>oldPosN)
    {
        shared_ptr<vector<int> > tempPosSampledList;
        GetPosNewAddedSampledList(oldPosN, newPosLambda, &tempPosSampledList);
        addSampledPosList->insert(addSampledPosList->end(), tempPosSampledList->begin(), tempPosSampledList->end());
    }
    if(newNegN>oldNegN)
    {
        shared_ptr<vector<int> > tempNegSampledList;
        GetNegNewAddedSampledList(oldNegN, newNegLambda, &tempNegSampledList);
        addSampledNegList->insert(addSampledNegList->end(), tempNegSampledList->begin(), tempNegSampledList->end());
    }
    
    // posLambda increased, shoud add samples from previous postive training data set.
    if(oldPosN>0 &&  newPosLambda>oldPosLambda)
    {
        shared_ptr<vector<int> > tempPosSampledList(new vector<int>);
        vector<int> posBootstrapList;
        BoostrapSampling(newPosLambda-oldPosLambda, oldPosN, 1.0, &posBootstrapList);
        tempPosSampledList->resize(posBootstrapList.size());
        for(int i=0;i<posBootstrapList.size();i++)
        {
            tempPosSampledList->at(i)=posTrainDataList->at(posBootstrapList[i]);
        }
        addSampledPosList->insert(addSampledPosList->end(), tempPosSampledList->begin(), tempPosSampledList->end());

    }
    // posLambda decreased, shoud remove samples from previous postive sampled training data set.
    else if (oldPosN>0 && newPosLambda<oldPosLambda)
    {
        double deltaLambdaN=(oldPosLambda-newPosLambda)*oldPosN;
        int deltaN=GetPossionNumber(deltaLambdaN);
        
        vector<int> posSampledListCopy;
        posSampledListCopy.insert(posSampledListCopy.end(), posSampledList->begin(),posSampledList->end());
        
        if(deltaN<posSampledListCopy.size())
        {
            for(int j=0;j<deltaN;j++)
            {
                int tempIdx=rand() % posSampledListCopy.size();
                rmvSampledPosList->push_back(posSampledListCopy.at(tempIdx));
                posSampledListCopy[tempIdx]=posSampledListCopy.back();
                posSampledListCopy.pop_back();
            }
        }
        else{
            rmvSampledPosList->insert(rmvSampledPosList->end(), posSampledList->begin(),posSampledList->end());
        }
    }
    
    // negLambda increased, shoud add samples from previous negative training data set.
    if(oldNegN>0 &&  newNegLambda>oldNegLambda)
    {
        shared_ptr<vector<int> > tempNegSampledList(new vector<int>);
        vector<int> negBootstrapList;
        BoostrapSampling(newNegLambda-oldNegLambda, oldNegN, 1.0, &negBootstrapList);
        tempNegSampledList->resize(negBootstrapList.size());
        for(int i=0;i<negBootstrapList.size();i++)
        {
            tempNegSampledList->at(i)=negTrainDataList->at(negBootstrapList[i]);
        }
        addSampledNegList->insert(addSampledNegList->end(), tempNegSampledList->begin(), tempNegSampledList->end());
    }
    // negLambda decreased, shoud remove samples from previous negative sampled training data set.
    else if (oldNegN>0 && newNegLambda<oldNegLambda)
    {
        double deltaLambdaN=(oldNegLambda-newNegLambda)*oldNegN;
        int deltaN=GetPossionNumber(deltaLambdaN);
        
        vector<int> negSampledListCopy;
        negSampledListCopy.insert(negSampledListCopy.end(), negSampledList->begin(),negSampledList->end());
        
        if(deltaN<negSampledListCopy.size())
        {
            for(int j=0;j<deltaN;j++)
            {
                int tempIdx=rand() % negSampledListCopy.size();
                rmvSampledNegList->push_back(negSampledListCopy.at(tempIdx));
                negSampledListCopy[tempIdx]=negSampledListCopy.back();
                negSampledListCopy.pop_back();
            }
        }
        else{
            rmvSampledNegList->insert(rmvSampledNegList->end(), negSampledList->begin(),negSampledList->end());
        }
    }
    addSampledList->insert(addSampledList->end(), addSampledPosList->begin(), addSampledPosList->end());
    addSampledList->insert(addSampledList->end(), addSampledNegList->begin(), addSampledNegList->end());
    rmvSampledList->insert(rmvSampledList->end(), rmvSampledPosList->begin(), rmvSampledPosList->end());
    rmvSampledList->insert(rmvSampledList->end(), rmvSampledNegList->begin(), rmvSampledNegList->end());
    
    *o_addSampleList=addSampledList;
    *o_removeSampleList=rmvSampledList;
    
    posSampledList->insert(posSampledList->end(), addSampledPosList->begin(), addSampledPosList->end());
    negSampledList->insert(negSampledList->end(), addSampledNegList->begin(), addSampledNegList->end());
    oldPosLambda=newPosLambda;
    oldNegLambda=newNegLambda;
}

template<typename T>
void RandomForest::ODTree<T>::Train(const shared_ptr<vector<shared_ptr<vector<T> > > > i_trainData)
{
    int oldNs=0;
    if(trainData)
    {
        oldNs=trainData->size();
    }
    trainData=i_trainData;
    shared_ptr<vector<int> > addSampleIndexList;
    shared_ptr<vector<int> > rmvSampleIndexList;
    
	if(root==nullptr)//create tree
	{
		//online bagging
        while (posSampledList->size()==0 || negSampledList->size()==0)
        {
            if(balanceType==SingleParameterBoostrap){
                SingleParameterBoostrapSampling(oldNs,&addSampleIndexList);
            }
            else {//if (balanceType==MultipleParameterBoostrap || balanceType==DynamicImbalanceAdaptableBootstrap){
                DynamicImbalanceAdaptiveBoostrapSampling(oldNs, &rmvSampleIndexList, &addSampleIndexList);
            }
        }
        root=new Node<T>;
		root->SetTree(this);
		root->SetSampleIndexList(addSampleIndexList);
		root->CreateTree();
	}
	else //update tree, now training data is the expanded data set
	{
        if(balanceType==SingleParameterBoostrap){
            SingleParameterBoostrapSampling(oldNs,&addSampleIndexList);
            root->UpdateTree(addSampleIndexList);
        }
        else if (balanceType==MultipleParameterBoostrap){
            MultipleParameterBoostrapSampling(oldNs, &addSampleIndexList);
            root->UpdateTree(addSampleIndexList);
        }
        else//balanceType==DynamicImbalanceAdaptableBootstrap
        {
            DynamicImbalanceAdaptiveBoostrapSampling(oldNs,&rmvSampleIndexList, &addSampleIndexList);
            root->UpdateTree(rmvSampleIndexList, addSampleIndexList);
            shared_ptr<vector<int> > newPosSampleList(new vector<int>);
            shared_ptr<vector<int> > newNegSampleList(new vector<int>);
            root->GetSampleList(newPosSampleList, newNegSampleList);
            posSampledList=newPosSampleList;
            negSampledList=newNegSampleList;
        }
	}
}

template<typename T>
void RandomForest::ODTree<T>::Predict(const shared_ptr<vector<shared_ptr<vector<T> > > > i_testData, vector<float> ** o_forecast)
{
    vector<float> *tempPredict=new vector<float>;
    tempPredict->resize(i_testData->size());
    for(int i=0;i<i_testData->size();i++)
    {
        tempPredict->at(i)=root->PredictOneSample(i_testData->at(i));
    }
    *o_forecast=tempPredict;
}

template<typename T>
void RandomForest::ODTree<T>::UpdateGiniImportance()
{
    giniImportance->resize(trainData->at(0)->size()-1);
    for(int i=0;i<giniImportance->size();i++) giniImportance->at(i)=0.0;
    root->UpdateGiniImportance();
}

template<typename T>
double RandomForest::ODTree<T>::GetOOBE(const shared_ptr<vector<shared_ptr<vector<T> > > > i_testData)
{
    double incorrectPrediction=0;
    for(int i=0;i<i_testData->size();i++)
    {
        double prediction=root->PredictOneSample(i_testData->at(i));
        double trueLabel=i_testData->at(i)->at(i_testData->at(i)->size()-1);
        if((prediction>=0.5 && trueLabel<0.5) || (prediction<0.5 && trueLabel>=0.5))
       {
           incorrectPrediction++;
       }
    }
    double oobe=-1;
    
    oobe=incorrectPrediction/i_testData->size();
    return oobe;
}

template<typename T>
double RandomForest::ODTree<T>::GetBalancedOOBE(const shared_ptr<vector<shared_ptr<vector<T> > > > i_testData)
{
    double incorPredPos=0;
    double incorPredNeg=0;
    double totalPredPos=0;
    double totalPredNeg=0;
    for(int i=0;i<i_testData->size();i++)
    {
        T tempLabel=i_testData->at(i)->back();
        if(tempLabel<0.5)
        {
            if(root->PredictOneSample(i_testData->at(i))>=0.5)
            {
                incorPredNeg++;
            }
            totalPredNeg++;
        }
        else if(tempLabel>=0.5)
        {
            if(root->PredictOneSample(i_testData->at(i))<0.5)
            {
                incorPredPos++;
            }
            totalPredPos++;
        }
    }
    double errorRatePos=0;
    double errorRateNeg=0;
    
    if(totalPredPos>0)errorRatePos=incorPredPos/totalPredPos;
    if(totalPredNeg>0)errorRateNeg=incorPredNeg/totalPredNeg;
    return (errorRatePos+errorRateNeg)/2;
}

template<typename T>
void RandomForest::ODTree<T>::ConvertTreeToList(int * io_left, int * io_right,
        int *io_splitFeature, double *io_splitValue)
{
    int currentListIndex=0;
    int globalListIndex=0;
    root->ConvertTreeToList(io_left, io_right, 
        io_splitFeature, io_splitValue,
        currentListIndex,&globalListIndex);
}



template class RandomForest::ODTree<double>;
template class RandomForest::ODTree<float>;
