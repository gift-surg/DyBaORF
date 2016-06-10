#include "ODTree.h"
#include "Node.h"

vector<int> GenerateRandomSequenceNumber(int N, int n)
{
    vector<bool> mask(N);
    if(n<=N/2)
    {
        for(int i=0;i<N;i++) mask[i]=false;
        int selectN=0;
        while(selectN<n)
        {
            double randf=(double)rand()/RAND_MAX;
            int tempI=N*randf;
            if(mask[tempI]==false)
            {
                mask[tempI]=true;
                selectN++;
            }
        }
    }
    else{
        for(int i=0;i<N;i++) mask[i]=true;
        int unselectN=0;
        while(unselectN<N-n)
        {
            double randf=(double)rand()/RAND_MAX;
            int tempI=N*randf;
            if(mask[tempI]==true)
            {
                mask[tempI]=false;
                unselectN++;
            }
        }
    }
    vector<int> list;
    list.reserve(n);
    for(int i=0;i<N;i++)
    {
//        double randf=(double)rand()/RAND_MAX;
//        int tempI=N*randf;
//        list.push_back(tempI);
        if(mask[i])list.push_back(i);
    }
    return list;
}

// For each iteration in random forest, draw a bootstrap sample from the minority class.
// Randomly draw the same number of cases, with replacement, from the majority class.
void BootstrapSampling(int possionLambda, int Ns, double bagFactor, vector<int> *o_list)
{
    o_list->reserve(Ns*bagFactor);
    for(int i=0;i<Ns;i++)
    {
        double randNumber=(double)rand()/RAND_MAX;
        if(randNumber>bagFactor)continue;
        double L=exp(-possionLambda);
        int k=0;
        double p=1;
        do
        {
            k=k+1;
            double u=(double)rand()/RAND_MAX;
            p=p*u;
        }
        while(p>L);
        k=k-1;
        for(int j=0;j<k;j++)
        {
            o_list->push_back(i);
        }
    }
}



template<typename T>
ODTree<T>::ODTree()
{
	root=nullptr;
	trainData=nullptr;
	depthUpperBound=10;
	varThreshold=0.01;
	sampleNumberThreshold=10;
    actureTreeDepth=0;
    actureTreeNode=0;
    useBalancedBagging=false;
    subDataSetRatio=1.0;
	srand(time(0));
    Reset();
}

template<typename T>
ODTree<T>::~ODTree()
{
//    if(root)root.reset();
    if(root) delete root;
    if(posTrainDataList) posTrainDataList.reset();
    if(negSampledList) negTrainDataList.reset();
    if(posSampledList) posSampledList.reset();
    if(negSampledList) negSampledList.reset();
    if(giniImportance) giniImportance.reset();
}

template<typename T>
void ODTree<T>::Reset()
{
    if(root)
    {
        delete root;
        root=nullptr;
    }
    posTrainDataList=make_shared<vector<int> >();
    negTrainDataList=make_shared<vector<int> >();
    posSampledList=make_shared<vector<int> >();
    negSampledList=make_shared<vector<int> >();
    giniImportance=make_shared<vector<double> >();
}
template<typename T>
void ODTree<T>::BalancedBagging(shared_ptr<vector<int> > o_indexList)
{
    posTrainDataList->reserve(trainData->size());
    negTrainDataList->reserve(trainData->size());
    for(int i=0;i<trainData->size();i++)
    {
        T tempLabel=trainData->at(i)->back();
        if(tempLabel>0)
        {
            posTrainDataList->push_back(i);
        }
        else{
            negTrainDataList->push_back(i);
        }
    }
    
    shared_ptr<vector<int> > shortList, longList;
    if(negTrainDataList->size()<posTrainDataList->size())
    {
        shortList=negTrainDataList;
        longList=posTrainDataList;
    }
    else{
        shortList=posTrainDataList;
        longList=negTrainDataList;
    }

    shared_ptr<vector<int> > sampleShortIndexList(new vector<int>);
    if(subDataSetRatio==1.0)
    {
        sampleShortIndexList=shortList;
    }
    else{
        int Nsample_short=shortList->size()*subDataSetRatio;
        sampleShortIndexList->reserve(Nsample_short);
        vector<int> tempIndexList=GenerateRandomSequenceNumber(shortList->size(),Nsample_short);
        for(int i=0;i<tempIndexList.size();i++)
        {
            sampleShortIndexList->push_back(shortList->at(tempIndexList.at(i)));
        }
    }
    
    vector<int> bagIndexSampleList0;
    shared_ptr<vector<int> > minoritySampleList(new vector<int>);
    shared_ptr<vector<int> > majoritySampleList(new vector<int>);
    BootstrapSampling(1, sampleShortIndexList->size(),1.0, &bagIndexSampleList0);
    // ensure that resulted positive sample set is not empty during the first training time
    if(bagIndexSampleList0.size()==0)
    {
        bagIndexSampleList0=GenerateRandomSequenceNumber(sampleShortIndexList->size(), sampleShortIndexList->size());
    }

    minoritySampleList->resize(bagIndexSampleList0.size());
    for(int i=0;i<bagIndexSampleList0.size();i++)
    {
        minoritySampleList->at(i)=sampleShortIndexList->at(bagIndexSampleList0.at(i));
    }
    vector<int> bagIndexSampleList1=GenerateRandomSequenceNumber(longList->size(), minoritySampleList->size());
    majoritySampleList->resize(bagIndexSampleList1.size());
    for(int i=0;i<bagIndexSampleList1.size();i++)
    {
        majoritySampleList->at(i)=longList->at(bagIndexSampleList1.at(i));
    }
    
    o_indexList->insert(o_indexList->end(), minoritySampleList->begin(), minoritySampleList->end());
    o_indexList->insert(o_indexList->end(), majoritySampleList->begin(), majoritySampleList->end());
    posSampledList=minoritySampleList;
    negSampledList=majoritySampleList;
}


template<typename T>
void ODTree<T>::GetUpdateSampleList(int oldNs, shared_ptr<vector<int> > * o_removeSampleList, shared_ptr<vector<int> > * o_addSampleList)
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
    int oldNp=posTrainDataList->size();
    int oldNn=negTrainDataList->size();
    int addNp=addPosTrainDataList->size();
    int addNn=addNegTrainDataList->size();
    int Np=oldNp+addNp;
    int Nn=oldNn+addNn;
    
    shared_ptr<vector<int> > addSampledList(new vector<int>);
    shared_ptr<vector<int> > addSampledPosList(new vector<int>);
    shared_ptr<vector<int> > addSampledNegList(new vector<int>);
    shared_ptr<vector<int> > rmvSampledNegList(new vector<int>);

    if(addNp>0)
    {
        shared_ptr<vector<int> > subIndexList1(new vector<int>);
        if(subDataSetRatio==1.0)
        {
            subIndexList1=addPosTrainDataList;
        }
        else{
            int addNsp=addNp*subDataSetRatio;
            vector<int> tempIndexList1=GenerateRandomSequenceNumber(addNp, addNsp);
            for(int i=0;i<tempIndexList1.size();i++)
            {
                subIndexList1->push_back(addPosTrainDataList->at(tempIndexList1.at(i)));
            }
        }
        
        vector<int> bagSampleSequence;
        BootstrapSampling(1, subIndexList1->size(),1.0, &bagSampleSequence);

        addSampledPosList->resize(bagSampleSequence.size());
        for(int i=0;i<bagSampleSequence.size();i++)
        {
            addSampledPosList->at(i)=subIndexList1->at(bagSampleSequence.at(i));
        }
    }
    
    int oldN_sampled_pos=posSampledList->size();
    int addN_sampled_pos=addSampledPosList->size();
    int newN_sampled_pos=oldN_sampled_pos+addN_sampled_pos;
    
    int oldN_sampled_neg=negSampledList->size();
    int newN_sampled_neg=newN_sampled_pos;
    int newN_sampled_neg_old_data=oldNn*((double)newN_sampled_neg/Nn);
    int newN_sampled_neg_add_data=newN_sampled_neg-newN_sampled_neg_old_data;
    int deltaN_sampled_neg_old_data=newN_sampled_neg_old_data-oldN_sampled_neg;

    
    // more sample should be obtained form previous neg training dataset
    if(deltaN_sampled_neg_old_data>0)
    {
        vector<int> tempIndexList;
        tempIndexList=GenerateRandomSequenceNumber(oldNn, deltaN_sampled_neg_old_data);
        addSampledNegList->resize(tempIndexList.size());
        for(int i=0;i<tempIndexList.size();i++)
        {
            addSampledNegList->at(i)=negTrainDataList->at(tempIndexList.at(i));
        }
    }
    // should remove extra samples from old negtive sample index list
    else if(deltaN_sampled_neg_old_data<0)
    {
        vector<int> tempIndexList;
        tempIndexList=GenerateRandomSequenceNumber(negSampledList->size(), -deltaN_sampled_neg_old_data);
        rmvSampledNegList->resize(tempIndexList.size());
        for(int i=0;i<tempIndexList.size();i++)
        {
            rmvSampledNegList->at(i)=negSampledList->at(tempIndexList.at(i));
        }
    }
    
    if(addNn>0)
    {
        // add new negtive samples from newly arrived negtive training data set
        vector<int> tempIndexList;
        tempIndexList=GenerateRandomSequenceNumber(addNn, newN_sampled_neg_add_data);
        addSampledNegList->reserve(addSampledNegList->size()+tempIndexList.size());
        for(int i=0;i<tempIndexList.size();i++)
        {
            addSampledNegList->push_back(addNegTrainDataList->at(tempIndexList.at(i)));
        }
    }
    addSampledList->insert(addSampledList->end(), addSampledPosList->begin(), addSampledPosList->end());
    addSampledList->insert(addSampledList->end(), addSampledNegList->begin(), addSampledNegList->end());
    *o_removeSampleList=rmvSampledNegList;
    *o_addSampleList=addSampledList;
    
    posTrainDataList->insert(posTrainDataList->end(), addPosTrainDataList->begin(),addPosTrainDataList->end());
    negTrainDataList->insert(negTrainDataList->end(), addNegTrainDataList->begin(),addNegTrainDataList->end());
}

template<typename T>
void ODTree<T>::Train(const shared_ptr<vector<shared_ptr<vector<T> > > > i_trainData)
{
    int oldNs=0;
    if(trainData)
    {
        oldNs=trainData->size();
    }
    trainData=i_trainData;
    int Ns=trainData->size();

	if(root==nullptr)//create tree
	{
		//online bagging
		shared_ptr<vector<int> > sampleIndexList(new vector<int>);
        if(useBalancedBagging)
        {
            BalancedBagging(sampleIndexList);
        }
        else{
            BootstrapSampling(1, Ns,subDataSetRatio ,sampleIndexList.get());
        }
        root=new Node<T>;
		root->SetTree(this);
		root->SetSampleIndexList(sampleIndexList);
		root->CreateTree();
	}
	else //update tree, now training data is the expanded data set
	{
        shared_ptr<vector<int> > addSampleIndexList;
        shared_ptr<vector<int> > removeSampleIndexList;
        if(useBalancedBagging)
        {
            GetUpdateSampleList(oldNs,&removeSampleIndexList, &addSampleIndexList);
            root->UpdateTree(removeSampleIndexList, addSampleIndexList);
            shared_ptr<vector<int> > newPosSampleList(new vector<int>);
            shared_ptr<vector<int> > newNegSampleList(new vector<int>);
            root->GetSampleList(newPosSampleList, newNegSampleList);
            posSampledList=newPosSampleList;
            negSampledList=newNegSampleList;
        }
        else{
            addSampleIndexList=make_shared<vector<int> >();
            BootstrapSampling(1, Ns-oldNs, subDataSetRatio ,addSampleIndexList.get());
            for(int i=0;i<addSampleIndexList->size();i++)
            {
                addSampleIndexList->at(i)+=oldNs;
            }
            root->UpdateTree(addSampleIndexList);
        }
	}
}

template<typename T>
void ODTree<T>::Predict(const shared_ptr<vector<shared_ptr<vector<T> > > > i_testData, vector<float> ** o_forecast)
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
void ODTree<T>::UpdateGiniImportance()
{
    giniImportance->resize(trainData->at(0)->size()-1);
    for(int i=0;i<giniImportance->size();i++) giniImportance->at(i)=0.0;
    root->UpdateGiniImportance();
}

template<typename T>
double ODTree<T>::GetOOBE(const shared_ptr<vector<shared_ptr<vector<T> > > > i_testData)
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
double ODTree<T>::GetBalancedOOBE(const shared_ptr<vector<shared_ptr<vector<T> > > > i_testData)
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
void ODTree<T>::ConvertTreeToList(int * io_left, int * io_right,
        int *io_splitFeature, double *io_splitValue)
{
    int currentListIndex=0;
    int globalListIndex=0;
    root->ConvertTreeToList(io_left, io_right, 
        io_splitFeature, io_splitValue,
        currentListIndex,&globalListIndex);
}



template class ODTree<double>;
template class ODTree<float>;
