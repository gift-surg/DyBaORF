//
//  Node.cpp
//  ORF
//
// Created by Guotai Wang on 01/12/2015.
//
// Copyright (c) 2014-2016 University College London, United Kingdom. All rights reserved.
// http://cmictig.cs.ucl.ac.uk
//
// Distributed under the BSD-3 licence. Please see the file licence.txt
//


#include "Node.h"
/////////////////////////////
//common functions
////////////////////////////

template<typename T>
void getFeatureRange(const std::shared_ptr<std::vector<std::shared_ptr<std::vector<T> > > > i_dataSet, int featureIndx,
                     T *o_min,T *o_max)
{
    T min=100000;
    T max=-100000;
    for(int i=0;i<i_dataSet->size();i++)
    {
        T value=i_dataSet->at(i)->at(featureIndx);
        if(value>max)max=value;
        if(value<min)min=value;
    }
    *o_min=min;
    *o_max=max;
}

/////////////////////////////
//Node
////////////////////////////
template<typename T>
RandomForest::Node<T>::Node()
{
    left=nullptr;
    right=nullptr;
    featureIndex=-1;
    splitValue=0;
    depth=0;
    sampleIndexList=nullptr;
    tree=NULL;
}

template<typename T>
RandomForest::Node<T>::~Node()
{
    if(left)  delete left;
    if(right) delete right;
}

template<typename T>
void RandomForest::Node<T>::binSplitDataSet(const std::shared_ptr<std::vector<int> > i_indexList, int featureIdx,  T featureValue,
                              std::shared_ptr<std::vector<int> > o_indexList0, std::shared_ptr<std::vector<int> > o_indexList1)
{
    //cout<<"binSplitDataSet started"<< std::endl;
    if(tree->TrainData()==nullptr)return;
    o_indexList0->reserve(i_indexList->size());
    o_indexList1->reserve(i_indexList->size());
    for(int i=0;i<i_indexList->size();i++)
    {
        int sampleIndex=i_indexList->at(i);
        T value=tree->TrainData()->at(sampleIndex)->at(featureIdx);
        if(value>featureValue)
        {
            o_indexList0->push_back(sampleIndex);
        }
        else
        {
            o_indexList1->push_back(sampleIndex);
        }
    }
}

template<typename T>
double RandomForest::Node<T>::meanLeaf()
{
    if(sampleIndexList==nullptr || sampleIndexList->size()==0)return -1;
    T sum=0;
    for (int i=0;i<sampleIndexList->size();i++)
    {
        int sampleIndex=sampleIndexList->at(i);
        T label=tree->TrainData()->at(sampleIndex)->back();
        sum=sum+label;
    }
    double mean=(double)sum/sampleIndexList->size();
    return mean;
}

template<typename T>
void RandomForest::Node<T>::GetFeatureRange(int fIndex,T * min,T * max)
{
   	T tempMin=100000;
    T tempMax=-100000;
    for (int i=0;i<sampleIndexList->size();i++)
    {
        int sampleIndex=sampleIndexList->at(i);
        T value=tree->TrainData()->at(sampleIndex)->at(fIndex);
        if(value>tempMax)tempMax=value;
        if(value<tempMin)tempMin=value;
    }
    
    *min=tempMin;
    *max=tempMax;
}

template<typename T>
double RandomForest::Node<T>::impurityLeaf(const std::shared_ptr<std::vector<int> > i_sampleIndexList)
{
    //	double sum=0;
    //	double sq_sum=0;
    //	for (int i=0;i<sampleNumber;i++)
    //	{
    //		int sampleIndex=i_sampleIndexList[i];
    //		double label=getDataValue(i_dataSet,Ns,Nfp1,sampleIndex,Nfp1-1);
    //		sum=sum+label;
    //		sq_sum=sq_sum+label*label;
    //	}
    //	double mean=sum/sampleNumber;
    //	double var=sqrt(sq_sum/sampleNumber-mean*mean);
    //	return var*sampleNumber;
    
    // Gini index
    if(i_sampleIndexList==nullptr)return -1;
    double N0=0;
    double N1=0;
    int sampleNumber=i_sampleIndexList->size();
    for (int i=0;i<sampleNumber;i++)
    {
        int sampleIndex=i_sampleIndexList->at(i);
        T label=tree->TrainData()->at(sampleIndex)->back();
        if(label>=1)
        {
            N1++;
        }
        else
        {
            N0++;
        }
    }
    return (1.0-(N0/sampleNumber)*(N0/sampleNumber)-(N1/sampleNumber)*(N1/sampleNumber))*sampleNumber;
}
template<typename T>
double RandomForest::Node<T>::impurityLeafWithWeight(const std::shared_ptr<std::vector<int> > i_sampleIndexList, double * w1, double * w0)
{
   	if(i_sampleIndexList==nullptr)return -1;
    double N0=0;
    double N1=0;
    int sampleNumber=i_sampleIndexList->size();
    for (int i=0;i<sampleNumber;i++)
    {
        int sampleIndex=i_sampleIndexList->at(i);
        T label=tree->TrainData()->at(sampleIndex)->back();
        if(label>=1)
        {
            N1++;
        }
        else
        {
            N0++;
        }
    }
    double tempW1=N1>N0?1.0:N0/N1;
    double tempW0=N1>N0?N1/N0:1.0;
    double tempN=tempW1*N1+tempW0*N0;
    double p1=tempW1*N1/tempN;
    double p0=1.0-p1;
    *w1=tempW1;
    *w0=tempW0;
    return (1.0-p1*p1-p0*p0)*tempN;
}
template<typename T>
double RandomForest::Node<T>::impurityLeafWithWeight(const std::shared_ptr<std::vector<int> > i_sampleIndexList, double w1, double w0)
{
   	if(i_sampleIndexList==nullptr)return -1;
    double N0=0;
    double N1=0;
    int sampleNumber=i_sampleIndexList->size();
    for (int i=0;i<sampleNumber;i++)
    {
        int sampleIndex=i_sampleIndexList->at(i);
        T label=tree->TrainData()->at(sampleIndex)->back();
        if(label>=1)
        {
            N1++;
        }
        else
        {
            N0++;
        }
    }
    double tempN=w1*N1+w0*N0;
    double p1=w1*N1/tempN;
    double p0=1.0-p1;
    return (1.0-p1*p1-p0*p0)*tempN;
}

template<typename T>
void RandomForest::Node<T>::chooseBestSplit(int * o_bestFeatureIndex, T * o_bestFeatureValue, double * o_decreadedImpurity)
{
    bool singleCls=true;
    T label0=tree->TrainData()->at(sampleIndexList->at(0))->back();
    int sampleNumber=sampleIndexList->size();
    for(int i=0;i<sampleNumber;i++)
    {
        int sampleIndex=sampleIndexList->at(i);
        T label=tree->TrainData()->at(sampleIndex)->back();
        if(label!=label0)
        {
            singleCls=false;
            break;
        }
    }
    if(singleCls)
    {
        *o_bestFeatureIndex=-1;
        *o_bestFeatureValue=meanLeaf();
        return;
    }
    double S,bestS;
    S=impurityLeaf(sampleIndexList);
    bestS=S;
    
    // randomly selected features for split
    int Nfsq=sqrt(tree->TrainData()->at(0)->size()-1)+1;
    for(int i=0;i<Nfsq;i++)
    {
        double randf=(double)rand()/RAND_MAX;
        int fIndex=(tree->TrainData()->at(0)->size()-1)*randf;
        T max=0;
        T min=0;
        GetFeatureRange(fIndex, &min, &max);
        for(int j=0;j<10;j++)
        {
            double splitValue=min+(double)(max-min)*rand()/RAND_MAX;
            std::shared_ptr<std::vector<int> > indexList0(new std::vector<int>);
            std::shared_ptr<std::vector<int> > indexList1(new std::vector<int>);
            binSplitDataSet(sampleIndexList,fIndex,  splitValue,indexList0,indexList1);
            if(indexList0->size()<tree->GetSampleNumberThreshold() || indexList1->size()<tree->GetSampleNumberThreshold()) continue;
            
            double newS;
            newS=impurityLeaf(indexList0)+impurityLeaf(indexList1);
            
            if(newS<bestS)
            {
                *o_bestFeatureIndex=fIndex;
                *o_bestFeatureValue=splitValue;
                *o_decreadedImpurity=(S-newS)/sampleIndexList->size();
                bestS=newS;
            }
        }
    }
    
    if((S-bestS)/sampleNumber < tree->GetVarThreshold())
    {
        *o_bestFeatureIndex=-1;
        *o_bestFeatureValue=meanLeaf();
    }
}

template<typename T>
void RandomForest::Node<T>::CreateTree()
{
    tree->SetActureTreeNode(tree->GetActureTreeNode()+1);
    int bestFeatureIndex;
    T bestFeatureValue;
    if(depth==tree->GetDepthUpperBound())
    {
        bestFeatureIndex=-1;
        bestFeatureValue=meanLeaf();
        featureIndex=bestFeatureIndex;
        splitValue=bestFeatureValue;
        return;
    }
    chooseBestSplit(&bestFeatureIndex, &bestFeatureValue, &decreasedImpurity);
    featureIndex=bestFeatureIndex;
    splitValue=bestFeatureValue;
    if(bestFeatureIndex==-1)
    {
        return;
    }
    
    std::shared_ptr<std::vector<int> > indexList0(new std::vector<int>);
    std::shared_ptr<std::vector<int> > indexList1(new std::vector<int>);
    binSplitDataSet(sampleIndexList, bestFeatureIndex,  bestFeatureValue,
                    indexList0,indexList1);
    
//    std::shared_ptr<Node<T> > leftchild(new Node<T>);
    Node<T> *leftchild=new Node<T>;
    leftchild->SetDepth(depth+1);
    if(leftchild->GetDepth() > tree->GetActureTreeDepth())
    {
        tree->SetActureTreeDepth(leftchild->GetDepth());
    }
    leftchild->SetSampleIndexList(indexList0);
    leftchild->SetTree(tree);
    leftchild->CreateTree();
    left=leftchild;
    
//    std::shared_ptr<Node<T> >  rightchild(new Node<T>);
    Node<T> *rightchild=new Node<T>;
    rightchild->SetDepth(depth+1);
    if(rightchild->GetDepth() > tree->GetActureTreeDepth())
    {
        tree->SetActureTreeDepth(rightchild->GetDepth());
    }
    rightchild->SetSampleIndexList(indexList1);
    rightchild->SetTree(tree);
    rightchild->CreateTree();
    right=rightchild;
}

template<typename T>
void RandomForest::Node<T>::UpdateGiniImportance()
{
    if(featureIndex==-1)
    {
        return;
    }
    else
    {
        double oldImportance=tree->GiniImportance()->at(featureIndex);
        tree->GiniImportance()->at(featureIndex)=oldImportance+ decreasedImpurity;
        left->UpdateGiniImportance();
        right->UpdateGiniImportance();
    }
}

template<typename T>
void RandomForest::Node<T>::UpdateTree(const std::shared_ptr<std::vector<int> > i_addSampleList)
{
    if(i_addSampleList->size()==0)return;
    sampleIndexList->insert(sampleIndexList->end(), i_addSampleList->begin(),i_addSampleList->end());
    if(featureIndex==-1)
    {
        
        CreateTree();
        return;
    }
    std::shared_ptr<std::vector<int> > indexList0(new std::vector<int> );
    std::shared_ptr<std::vector<int> > indexList1(new std::vector<int> );
    binSplitDataSet(i_addSampleList, featureIndex,  splitValue,
                    indexList0,indexList1);
    left->UpdateTree(indexList0);
    right->UpdateTree(indexList1);
}

//the returned value is the number of samples at current node
template<typename T>
int RandomForest::Node<T>::UpdateTree(const std::shared_ptr<std::vector<int> > i_rmvSampleList, const std::shared_ptr<std::vector<int> > i_addSampleList)
{
    if(i_rmvSampleList->size()==0 && i_addSampleList->size()==0)return sampleIndexList->size();
    if(featureIndex==-1)
    {
        for(int i=0;i<i_rmvSampleList->size();i++)
        {
            int tempIndex=i_rmvSampleList->at(i);
            std::vector<int>::iterator it;
            it = find (sampleIndexList->begin(), sampleIndexList->end(), tempIndex);
            if(it!=sampleIndexList->end())
            {
                sampleIndexList->erase(it);
            }
        }
        
        sampleIndexList->insert(sampleIndexList->end(), i_addSampleList->begin(), i_addSampleList->end());
        if(sampleIndexList->size()>0) CreateTree();
        return sampleIndexList->size();
    }
    
    std::shared_ptr<std::vector<int> > rmvSampleList0(new std::vector<int>);
    std::shared_ptr<std::vector<int> > rmvSampleList1(new std::vector<int>);
    std::shared_ptr<std::vector<int> > addSampleList0(new std::vector<int>);
    std::shared_ptr<std::vector<int> > addSampleList1(new std::vector<int>);

    binSplitDataSet(i_rmvSampleList, featureIndex, splitValue, rmvSampleList0, rmvSampleList1);
    binSplitDataSet(i_addSampleList, featureIndex, splitValue, addSampleList0, addSampleList1);
    int leftSize=left->UpdateTree(rmvSampleList0, addSampleList0);
    int rightSize=right->UpdateTree(rmvSampleList1, addSampleList1);
//    if(leftSize==0 || rightSize==0)
    if(i_rmvSampleList->size()>0 || i_addSampleList->size()>0)
    {
//        if(leftSize==0 || rightSize==0 || left->GetFeatureIndex()==-1 || right->GetFeatureIndex()==-1)
        if(leftSize==0 || rightSize==0 )
        {
            sampleIndexList->clear();
            sampleIndexList->insert(sampleIndexList->end(), left->GetSampleIndexList()->begin(),left->GetSampleIndexList()->end());
            sampleIndexList->insert(sampleIndexList->end(), right->GetSampleIndexList()->begin(),right->GetSampleIndexList()->end());
            tree->SetActureTreeNode(tree->GetActureTreeNode()-3);
            if(leftSize+rightSize>0) CreateTree();
        }
    }
    return leftSize+rightSize;
}

template<typename T>
void RandomForest::Node<T>::GetSampleList(std::shared_ptr<std::vector<int> > o_posSampleList, std::shared_ptr<std::vector<int> > o_negSampleList)
{
    if(featureIndex==-1)
    {
        for(int i=0;i<sampleIndexList->size();i++)
        {
            int tempIndex=sampleIndexList->at(i);
            T label=tree->TrainData()->at(tempIndex)->back();
            if(label>0)
            {
                o_posSampleList->push_back(tempIndex);
            }
            else{
                o_negSampleList->push_back(tempIndex);
            }
        }
        return;
    }
    if(left)  left->GetSampleList(o_posSampleList,o_negSampleList);
    if(right)right->GetSampleList(o_posSampleList,o_negSampleList);
}

template<typename T>
RandomForest::Node<T> * RandomForest::Node<T>::GetLeft() const {
    return left;
}

template<typename T>
RandomForest::Node<T> * RandomForest::Node<T>::GetRight() const {
    return right;
}

template<typename T>
double RandomForest::Node<T>::PredictOneSample(const std::shared_ptr<std::vector<T> > i_inData)
{
    if(featureIndex==-1)
    {
        return splitValue;
    }
    if(i_inData->at(featureIndex)>splitValue)
    {
        return left->PredictOneSample(i_inData);
    }
    else
    {
        return right->PredictOneSample(i_inData);
    }
}

template<typename T>
void RandomForest::Node<T>::ConvertTreeToList(int * io_left, int * io_right,
                                int *io_splitFeature, double *io_splitValue,
                                int currentListIndex, int * io_globalListIndex)
{
    io_splitFeature[currentListIndex]=featureIndex;
    io_splitValue[currentListIndex]=splitValue;
    if(featureIndex!=-1)
    {
        *io_globalListIndex=*io_globalListIndex+1;
        int leftListIndex=*io_globalListIndex;
        io_left[currentListIndex]=leftListIndex;
        left->ConvertTreeToList(io_left, io_right, 
                                io_splitFeature, io_splitValue, 
                                leftListIndex, io_globalListIndex);
        
        *io_globalListIndex=*io_globalListIndex+1;
        int rightListIndex=*io_globalListIndex;
        io_right[currentListIndex]=rightListIndex;
        right->ConvertTreeToList(io_left, io_right, 
                                 io_splitFeature, io_splitValue, 
                                 rightListIndex, io_globalListIndex);
    }
}

template class RandomForest::Node<double>;
template class RandomForest::Node<float>;
