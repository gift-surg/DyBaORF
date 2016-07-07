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
    Reset();
}

template<typename T>
RandomForest::ODTree<T>::~ODTree()
{
    if(posTrainDataList) posTrainDataList.reset();
    if(negSampledList) negTrainDataList.reset();
    if(posSampledList) posSampledList.reset();
    if(negSampledList) negSampledList.reset();
    if(giniImportance) giniImportance.reset();
}

template<typename T>
void RandomForest::ODTree<T>::Reset()
{
    root=nullptr;
    trainData=nullptr;
    posTrainDataList=std::make_shared<std::vector<int> >();
    negTrainDataList=std::make_shared<std::vector<int> >();
    posSampledList=std::make_shared<std::vector<int> >();
    negSampledList=std::make_shared<std::vector<int> >();
    giniImportance=std::make_shared<std::vector<double> >();
    oldPosLambda=1.0;
    oldNegLambda=1.0;
}



template<typename T>
void RandomForest::ODTree<T>::Train(const std::shared_ptr<std::vector<std::shared_ptr<std::vector<T> > > > i_trainData)
{
    int oldNs=0;
    if(trainData)
    {
        oldNs=trainData->size();
    }
    trainData=i_trainData;
    std::shared_ptr<std::vector<int> > addSampleIndexList;
    std::shared_ptr<std::vector<int> > rmvSampleIndexList;
    
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
        std::shared_ptr<Node<T> > tempRoot(new Node<T>(this));
        root=tempRoot;
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
            std::shared_ptr<std::vector<int> > newPosSampleList(new std::vector<int>);
            std::shared_ptr<std::vector<int> > newNegSampleList(new std::vector<int>);
            root->GetSampleList(newPosSampleList, newNegSampleList);
            posSampledList=newPosSampleList;
            negSampledList=newNegSampleList;
        }
    }
}

template<typename T>
void RandomForest::ODTree<T>::Predict(const std::shared_ptr<std::vector<std::shared_ptr<std::vector<T> > > > i_testData, std::vector<float> ** o_forecast)
{
    std::vector<float> *tempPredict=new std::vector<float>;
    tempPredict->resize(i_testData->size());
    for(int i=0;i<i_testData->size();i++)
    {
        tempPredict->at(i)=root->PredictOneSample(i_testData->at(i));
    }
    *o_forecast=tempPredict;
}


template<typename T>
double RandomForest::ODTree<T>::GetOOBE(const std::shared_ptr<std::vector<std::shared_ptr<std::vector<T> > > > i_testData) const
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
std::shared_ptr<std::vector<std::shared_ptr<std::vector<T> > > >
RandomForest::ODTree<T>::GetTrainData() const
{
    return trainData;
};

template<typename T>
void RandomForest::ODTree<T>::SetActureTreeDepth(int d)
{
    actureTreeDepth=d;
};

template<typename T>
int RandomForest::ODTree<T>::GetActureTreeDepth() const
{
    return actureTreeDepth;
};

template<typename T>
void RandomForest::ODTree<T>::SetDepthUpperBound(int d)
{
    depthUpperBound=d;
};

template<typename T>
int RandomForest::ODTree<T>::GetDepthUpperBound() const
{
    return depthUpperBound;
};

template<typename T>
void RandomForest::ODTree<T>::SetActureTreeNode(int n)
{
    actureTreeNode=n;
};

template<typename T>
int RandomForest::ODTree<T>::GetActureTreeNode() const
{
    return actureTreeNode;
};

template<typename T>
void RandomForest::ODTree<T>::SetVarTreshold(double t)
{
    varThreshold=t;
};

template<typename T>
double RandomForest::ODTree<T>::GetVarThreshold() const
{
    return varThreshold;
};

template<typename T>
void RandomForest::ODTree<T>::SetSampleNumberThreshold(int n)
{
    sampleNumberThreshold=n;
};

template<typename T>
int RandomForest::ODTree<T>::GetSampleNumberThreshold() const
{
    return sampleNumberThreshold;
};

template<typename T>
void RandomForest::ODTree<T>::SetBalanceType(BalanceType type)
{
    balanceType=type;
};

template<typename T>
RandomForest::BalanceType RandomForest::ODTree<T>::GetBalanceType() const
{
    return balanceType;
};

template<typename T>
void RandomForest::ODTree<T>::SetSamplingType(SamplingType type)
{
    samplingType=type;
};

template<typename T>
RandomForest::SamplingType RandomForest::ODTree<T>::GetSamplingType() const
{
    return samplingType;
};


template<typename T>
void RandomForest::ODTree<T>::UpdateGiniImportance()
{
    giniImportance->resize(trainData->at(0)->size()-1);
    for(int i=0;i<giniImportance->size();i++) giniImportance->at(i)=0.0;
    root->UpdateGiniImportance();
}

template<typename T>
std::shared_ptr<std::vector<double> > RandomForest::ODTree<T>::GetGiniImportance() const
{
    return giniImportance;
};


////private functions

template<typename T>
void RandomForest::ODTree<T>::UpdateTrainDataList(int oldNs)
{
    std::shared_ptr<std::vector<int> > addPosTrainDataList(new std::vector<int>);
    std::shared_ptr<std::vector<int> > addNegTrainDataList(new std::vector<int>);
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
void RandomForest::ODTree<T>::GetPosNewAddedSampledList(int oldPosN, double lambdaPos,std::shared_ptr<std::vector<int> > *o_list)
{
    std::shared_ptr<std::vector<int> > tempPosSampledList(new std::vector<int>);
    int newPosN=posTrainDataList->size();
    if(newPosN>oldPosN)
    {
        std::vector<int> posBootstrapList;
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
void RandomForest::ODTree<T>::GetNegNewAddedSampledList(int oldNegN, double lambdaNeg, std::shared_ptr<std::vector<int> > *o_list)
{
    std::shared_ptr<std::vector<int> > tempNegSampledList(new std::vector<int>);
    int newNegN=negTrainDataList->size();
    if(newNegN>oldNegN)
    {
        std::vector<int> negBootstrapList;
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
void RandomForest::ODTree<T>::SingleParameterBoostrapSampling(int oldNs, std::shared_ptr<std::vector<int> > *o_list)
{
    int oldPosN=posTrainDataList->size();
    int oldNegN=negTrainDataList->size();
    UpdateTrainDataList(oldNs);
    int newPosN=posTrainDataList->size();
    int newNegN=negTrainDataList->size();
    
    std::shared_ptr<std::vector<int> > outputSampledList(new std::vector<int>);
    if(newPosN>oldPosN)
    {
        std::shared_ptr<std::vector<int> > tempPosSampledList;
        GetPosNewAddedSampledList(oldPosN, 1.0, &tempPosSampledList);
        posSampledList->insert(posSampledList->end(), tempPosSampledList->begin(), tempPosSampledList->end());
        outputSampledList->insert(outputSampledList->end(), tempPosSampledList->begin(), tempPosSampledList->end());
    }
    if(newNegN>oldNegN)
    {
        std::shared_ptr<std::vector<int> > tempNegSampledList;
        GetNegNewAddedSampledList(oldNegN, 1.0, &tempNegSampledList);
        negSampledList->insert(negSampledList->end(), tempNegSampledList->begin(), tempNegSampledList->end());
        outputSampledList->insert(outputSampledList->end(), tempNegSampledList->begin(), tempNegSampledList->end());
    }
    *o_list=outputSampledList;
}

template<typename T>
void RandomForest::ODTree<T>::MultipleParameterBoostrapSampling(int oldNs, std::shared_ptr<std::vector<int> > *o_list)
{
    int oldPosN=posTrainDataList->size();
    int oldNegN=negTrainDataList->size();
    UpdateTrainDataList(oldNs);
    int newPosN=posTrainDataList->size();
    int newNegN=negTrainDataList->size();

    double imbalanceRatio=static_cast<double>(newNegN)/newPosN;
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
    
    std::shared_ptr<std::vector<int> > outputSampledList(new std::vector<int>);
    if(newPosN>oldPosN)
    {
        std::shared_ptr<std::vector<int> > tempPosSampledList;
        GetPosNewAddedSampledList(oldPosN, posLambda, &tempPosSampledList);
        posSampledList->insert(posSampledList->end(), tempPosSampledList->begin(), tempPosSampledList->end());
        outputSampledList->insert(outputSampledList->end(), tempPosSampledList->begin(), tempPosSampledList->end());
    }
    if(newNegN>oldNegN)
    {
        std::shared_ptr<std::vector<int> > tempNegSampledList;
        GetNegNewAddedSampledList(oldNegN, negLambda, &tempNegSampledList);
        negSampledList->insert(negSampledList->end(), tempNegSampledList->begin(), tempNegSampledList->end());
        outputSampledList->insert(outputSampledList->end(), tempNegSampledList->begin(), tempNegSampledList->end());
    }
    *o_list=outputSampledList;
}

template<typename T>
void RandomForest::ODTree<T>::DynamicImbalanceAdaptiveBoostrapSampling(int oldNs, std::shared_ptr<std::vector<int> > * o_removeSampleList, std::shared_ptr<std::vector<int> > * o_addSampleList)
{
    int oldPosN=posTrainDataList->size();
    int oldNegN=negTrainDataList->size();
    UpdateTrainDataList(oldNs);
    int newPosN=posTrainDataList->size();
    int newNegN=negTrainDataList->size();
    
    
    double newImbalanceRatio=static_cast<double>(newNegN)/newPosN;
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
    
    std::shared_ptr<std::vector<int> > addSampledList(new std::vector<int>);
    std::shared_ptr<std::vector<int> > addSampledPosList(new std::vector<int>);
    std::shared_ptr<std::vector<int> > addSampledNegList(new std::vector<int>);
    std::shared_ptr<std::vector<int> > rmvSampledList(new std::vector<int>);
    std::shared_ptr<std::vector<int> > rmvSampledPosList(new std::vector<int>);
    std::shared_ptr<std::vector<int> > rmvSampledNegList(new std::vector<int>);
    
    // sample new arrived data with newPosLambda and newNegLambda
    if(newPosN>oldPosN)
    {
        std::shared_ptr<std::vector<int> > tempPosSampledList;
        GetPosNewAddedSampledList(oldPosN, newPosLambda, &tempPosSampledList);
        addSampledPosList->insert(addSampledPosList->end(), tempPosSampledList->begin(), tempPosSampledList->end());
    }
    if(newNegN>oldNegN)
    {
        std::shared_ptr<std::vector<int> > tempNegSampledList;
        GetNegNewAddedSampledList(oldNegN, newNegLambda, &tempNegSampledList);
        addSampledNegList->insert(addSampledNegList->end(), tempNegSampledList->begin(), tempNegSampledList->end());
    }
    
    // posLambda increased, shoud add samples from previous postive training data set.
    if(oldPosN>0 &&  newPosLambda>oldPosLambda)
    {
        std::shared_ptr<std::vector<int> > tempPosSampledList(new std::vector<int>);
        std::vector<int> posBootstrapList;
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
        
        std::vector<int> posSampledListCopy;
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
        std::shared_ptr<std::vector<int> > tempNegSampledList(new std::vector<int>);
        std::vector<int> negBootstrapList;
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
        
        std::vector<int> negSampledListCopy;
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
int RandomForest::ODTree<T>::GetPossionNumber(double lambda)
{
    double L=exp(-lambda);
    int k=0;
    double p=1;
    do{
        k=k+1;
        double u=static_cast<double>(rand())/RAND_MAX;
        p=p*u;
    }
    while(p>L);
    k=k-1;
    return k;
}


template<typename T>
void RandomForest::ODTree<T>::BoostrapSampling(double possionLambda, int Ns, double bagFactor, std::vector<int> *o_list)
{
    o_list->reserve(Ns*bagFactor);
    for(int i=0;i<Ns;i++)
    {
        double randNumber=static_cast<double>(rand())/RAND_MAX;
        if(randNumber>bagFactor)continue;
        int k=GetPossionNumber(possionLambda);
        for(int j=0;j<k;j++)
        {
            o_list->push_back(i);
        }
    }
}

template class RandomForest::ODTree<double>;
template class RandomForest::ODTree<float>;
