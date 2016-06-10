//
//  AbstractTestExample.cpp
//  DyBaORF_test
//
//  Created by Guotai Wang on 03/12/2015.
//
// Copyright (c) 2014-2016 University College London, United Kingdom. All rights reserved.
// http://cmictig.cs.ucl.ac.uk
//
// Distributed under the BSD-3 licence. Please see the file licence.txt
//

#include "AbstractTestExample.h"
#include <fstream>
#include <iostream>

double String2Double(string str)
{
    double sum=0;
    for (int i=0;i<str.length();i++)
    {
        sum+=str.at(i);
    }
    return sum;
}

void GetMeanAndStd(vector<vector<double> > i_array, vector<double> * o_mean, vector<double> * o_std)
{
    int rows=i_array.size();
    int column=i_array.at(0).size();
    vector<double> sum(column);
    vector<double> sumSq(column);
    vector<double> mean(column);
    vector<double> std(column);
    for(int i=0;i<column;i++)
    {
        sum[i]=0;
        sumSq[i]=0;
    }
    for(int i=0;i<rows;i++)
    {
        for(int j=0;j<column;j++)
        {
            double tempV=i_array.at(i).at(j);
            sum[j]+=tempV;
            sumSq[j]+=tempV*tempV;
        }
    }
    
    for(int i=0;i<column;i++)
    {
        mean[i]=sum[i]/rows;
        std[i]=sqrt(sumSq[i]/rows-mean[i]*mean[i]);
        o_mean->push_back(mean[i]);
        o_std->push_back(std[i]);
    }
}

AbstractTestExample::AbstractTestExample()
{
    featureN=0;
    instanceN=0;
    positiveN=0;
    negtiveN=0;
    trainN=0;
    testN=0;
    posNtest=0;
    negNtest=0;
    originData=nullptr;
    trainData=nullptr;
    testData=nullptr;
}

AbstractTestExample::~AbstractTestExample()
{
    
}

bool AbstractTestExample::LoadData(DataSetName data)
{
    switch (data) {
        case CTG:
            return LoadCTGDataSet();
        case WINE:
            return LoadWineDataSet();
        case MUSK:
            return LoadMuskDataSet();
        case BIODEG:
            return LoadBiodegDataSet();
        default:
            break;
    }
}

bool AbstractTestExample::LoadCTGDataSet()
{
    string fileName="../../data/CTG.txt";
    ifstream fileLoader(fileName.c_str());
    if(!fileLoader.is_open())
    {
        cout<<"open file failed"<<endl;
        return false;
    }
    featureN=21;
    instanceN=2126;
    positiveN=0;
    negtiveN=0;
    
    featureTypeList.resize(featureN);
    for(int i=0;i<featureN;i++)
    {
        featureTypeList[i]= (i==featureN-1) ? MULTINOMIAL : GAUSSIAN;
    }
    
    double label;
    double posLabel=8;//4,7
    shared_ptr<vector<shared_ptr<vector<double> > > > readData(new vector<shared_ptr<vector<double> > >);
    readData->reserve(instanceN);
    for(int i=0;i<instanceN;i++)
    {
        shared_ptr<vector<double> > tempSample(new vector<double>);
        tempSample->reserve(featureN+1);
        for(int j=0;j<featureN;j++)
        {
            double tempV;
            fileLoader>>tempV;
            tempSample->push_back(tempV);
        }
        
        fileLoader>>label;
        if(label==posLabel)
        {
            tempSample->push_back(1);
            positiveN++;
        }
        else{
            tempSample->push_back(0);
            negtiveN++;
        }
        readData->push_back(tempSample);
    }
    originData=readData;
    cout<<"DataSet    Feature Value"<<endl;
    cout<<"CTG        "<<posLabel<<endl;
    UpdateDataInfo();
    //GenerateTrainAndTestData();

    return true;
}


bool AbstractTestExample::LoadWineDataSet()
{
    string fileName="../../data/winequality.data";
    ifstream fileLoader(fileName.c_str());
    if(!fileLoader.is_open())
    {
        cout<<"open file failed"<<endl;
        return false;
    }
    featureN=11;
    instanceN=6497;
    positiveN=0;
    negtiveN=0;
    
    featureTypeList.resize(featureN);
    for(int i=0;i<featureN;i++)
    {
        featureTypeList[i]= GAUSSIAN;
    }
    
    int label;
    int posLabel=8;//5,8
    shared_ptr<vector<shared_ptr<vector<double> > > > readData(new vector<shared_ptr<vector<double> > >);
    readData->reserve(instanceN);
    for(int i=0;i<instanceN;i++)
    {
        shared_ptr<vector<double> > tempSample(new vector<double>);
        tempSample->reserve(featureN+1);
        
        for(int j=0;j<featureN;j++)
        {
            double tempV;
            fileLoader>>tempV;
            tempSample->push_back(tempV);
            char comma;
            fileLoader>>comma;
        }
        
        fileLoader>>label;
        if(label==posLabel)
        {
            tempSample->push_back(1);
            positiveN++;
        }
        else{
            tempSample->push_back(0);
            negtiveN++;
        }
        readData->push_back(tempSample);
    }
    originData=readData;
    cout<<"DataSet    Feature Value"<<endl;
    cout<<"Wine       "<<posLabel<<endl;
    UpdateDataInfo();
    
    
    //GenerateTrainAndTestData();

    return true;
}


bool AbstractTestExample::LoadMuskDataSet()
{
    string fileName="../../data/musk1.data";
    ifstream fileLoader(fileName.c_str());
    if(!fileLoader.is_open())
    {
        cout<<"open file failed"<<endl;
        return false;
    }
    featureN=166;
    instanceN=476;//6598;//476
    positiveN=0;
    negtiveN=0;
    
    featureTypeList.resize(featureN);
    for(int i=0;i<featureN;i++)
    {
        featureTypeList[i]= GAUSSIAN;
    }
    
    shared_ptr<vector<shared_ptr<vector<double> > > > readData(new vector<shared_ptr<vector<double> > >);
    readData->reserve(instanceN*2);
    
    int label;
    int posLabel=1;
    for(int i=0;i<instanceN;i++)
    {
        shared_ptr<vector<double> > tempSample(new vector<double>);
        tempSample->reserve(featureN+1);
        
        string str1,str2;
        fileLoader>>str1>>str2;
        for(int j=0;j<featureN;j++)
        {
            double tempV;
            fileLoader>>tempV;
            tempSample->push_back(tempV);
        }
        
        fileLoader>>label;
        if(label==posLabel)
        {
            tempSample->push_back(1);
            positiveN++;
        }
        else{
            tempSample->push_back(0);
            negtiveN++;
        }
        readData->push_back(tempSample);
    }
    originData=readData;
    cout<<"DataSet   Feature Value"<<endl;
    cout<<"Musk  "<<posLabel<<endl;
    UpdateDataInfo();
    //GenerateTrainAndTestData();
    return true;
}

bool AbstractTestExample::LoadBiodegDataSet()
{
    string fileName="../../data/biodeg.csv";
    ifstream fileLoader(fileName.c_str());
    if(!fileLoader.is_open())
    {
        cout<<"open file failed"<<endl;
        return false;
    }
    featureN=41;
    instanceN=1055;
    positiveN=0;
    negtiveN=0;
    
    featureTypeList.resize(featureN);
    for(int i=0;i<featureN;i++)
    {
        featureTypeList[i]= GAUSSIAN;
    }
    
    shared_ptr<vector<shared_ptr<vector<double> > > > readData(new vector<shared_ptr<vector<double> > >);
    readData->reserve(instanceN*2);
    
    string label;
    string posLabel="RB";
    for(int i=0;i<instanceN;i++)
    {
        shared_ptr<vector<double> > tempSample(new vector<double>);
        tempSample->reserve(featureN+1);
        
        for(int j=0;j<featureN;j++)
        {
            double tempV;
            fileLoader>>tempV;
            tempSample->push_back(tempV);
            char comma;
            fileLoader>>comma;
        }
        
        fileLoader>>label;
        if(label==posLabel)
        {
            tempSample->push_back(1);
            positiveN++;
        }
        else{
            tempSample->push_back(0);
            negtiveN++;
        }
        readData->push_back(tempSample);
    }
    originData=readData;
    cout<<"DataSet   Feature Value"<<endl;
    cout<<"Biodeg  "<<posLabel<<endl;
    UpdateDataInfo();
    //GenerateTrainAndTestData();
    return true;
}


void AbstractTestExample::GenerateTrainAndTestData()
{
    shared_ptr<vector<int> > posIndex(new vector<int>);
    shared_ptr<vector<int> > negIndex(new vector<int>);
    
    shared_ptr<vector<shared_ptr<vector<double> > > > tempTrainData(new vector<shared_ptr<vector<double> > >);
    shared_ptr<vector<shared_ptr<vector<double> > > > tempTestData(new vector<shared_ptr<vector<double> > >);
    tempTrainData->reserve(trainN);
    tempTestData->reserve(testN);
    
    shared_ptr<vector<bool> > testMask(new vector<bool>);
    testMask->resize(instanceN);
    for(int i=0;i<instanceN;i++) testMask->at(i)=false;
    
    for(int i=0;i<instanceN;i++)
    {
        double tempL=originData->at(i)->back();
        if(tempL==1.0)
            posIndex->push_back(i);
        else
            negIndex->push_back(i);
    }
    
    for(int i=0;i<posNtest;i++)
    {
        int randN=rand() % posIndex->size();
        int tempIdx=posIndex->at(randN);
        posIndex->at(randN)=posIndex->back();
        posIndex->pop_back();
        tempTestData->push_back(originData->at(tempIdx));
        testMask->at(tempIdx)=true;
    }
    
    for(int i=0;i<negNtest;i++)
    {
        int randN=rand() % negIndex->size();
        int tempIdx=negIndex->at(randN);
        negIndex->at(randN)=negIndex->back();
        negIndex->pop_back();
        tempTestData->push_back(originData->at(tempIdx));
        testMask->at(tempIdx)=true;
    }
    
    for(int i=0;i<originData->size();i++)
    {
        if(testMask->at(i)==false)
        {
            tempTrainData->push_back(originData->at(i));
        }
    }
    trainData=tempTrainData;
    testData=tempTestData;
}

void AbstractTestExample::SetTrainDataChunk(double startPercent, double endPercent, double increasePercent)
{
    int startIndex=startPercent*trainN;
    int endIndex=endPercent*trainN;
    int steps=ceil((endPercent-startPercent)/increasePercent);
    trainIndexEachUpdate.resize(steps+2);
    trainIndexEachUpdate.at(0)=0;
    for(int i=0;i<=steps;i++)
    {
        trainIndexEachUpdate[1+i]=(i==steps)?endIndex:(startIndex+i*increasePercent*trainN);
    }
}

void AbstractTestExample::UpdateDataInfo()
{
    cout<<"postive   "<<positiveN<<endl;
    cout<<"negative  "<<negtiveN<<endl;
    cout<<"imbalance "<<std::setprecision(4)<<(double)negtiveN/positiveN<<endl;
    
    double testRate=0.2;
    testN=instanceN*testRate;
    trainN=instanceN-testN;
    posNtest=positiveN*testRate;
    negNtest=testN-posNtest;
    posNtrain=positiveN-posNtest;
    negNtrain=negtiveN-negNtest;
}
void AbstractTestExample::PrintDataInformation()
{
    // print imbalance ratio after one data chunk arrives
    int posN=0;
    int negN=0;
    double imbalanceRatio=0;
    double sum=0;
    double sumsq=0;
    int steps=1;
    for(int i=0; i<trainN;i++)
    {
        double tempL=trainData->at(i)->back();
        if(tempL==1.0)posN++;
        else negN++;
        
        if(i==trainIndexEachUpdate[steps]-1)
        {
            imbalanceRatio=(double)negN/posN;
            sum+=imbalanceRatio;
            sumsq+=imbalanceRatio*imbalanceRatio;
            steps++;
            double percent=(double)i/trainN;
            cout<<std::setw(4)<< std::fixed<< std::setprecision(2)<<percent<<"   "
                <<std::setw(4)<< std::fixed<< std::setprecision(2)<<imbalanceRatio<<endl;
        }
    }
    
    double meanImbalance=sum/(steps-1);
    double std=sqrt(sumsq/(steps-1)-meanImbalance*meanImbalance);
    double normalizedStd=std/meanImbalance;
    cout<<"meanImbalance "<<meanImbalance<<endl;
    cout<<"std "<<std<<endl;
    cout<<"normalizedStd "<<normalizedStd<<endl;
}

vector<double> AbstractTestExample::GetImbalanceRatio()
{
    vector<double> imbalanceRatioList;
    int posN=0;
    int negN=0;
    double imbalanceRatio=0;
    int steps=1;
    for(int i=0; i<trainN;i++)
    {
        double tempL=trainData->at(i)->back();
        if(tempL==1.0)posN++;
        else negN++;
        
        if(i==trainIndexEachUpdate[steps]-1)
        {
            imbalanceRatio=(double)negN/posN;
            imbalanceRatioList.push_back(imbalanceRatio);
            steps++;
        }
    }
    return imbalanceRatioList;
}
void AbstractTestExample::SetTrainDataOnline(int startIndex, int endIndex)
{
    trainIndexEachUpdate.resize(endIndex-startIndex+2);
    trainIndexEachUpdate.at(0)=0;
    for(int i=0;i<=endIndex-startIndex;i++)
    {
        trainIndexEachUpdate[1+i]=startIndex+i;
    }
}

void AbstractTestExample::Run(int MaxIter)
{
    Sensitivity.clear();
    Specificity.clear();
    Gmean.clear();
    Time.clear();
    
    compareSensitivity.clear();
    compareSpecificity.clear();
    compareGmean.clear();
    compareTime.clear();

    for(int idx=0;idx<MaxIter;idx++)
    {
        for(int i=0;i<instanceN/2;i++)
        {
            int idx1=instanceN;
            while(idx1==instanceN) idx1=(double)instanceN*rand()/RAND_MAX;
            int idx2=instanceN;
            while(idx2==instanceN) idx2=(double)instanceN*rand()/RAND_MAX;
            if(idx1!=idx2)
            {
                shared_ptr<vector<double> > tempSample=originData->at(idx1);
                originData->at(idx1)=originData->at(idx2);
                originData->at(idx2)=tempSample;
            }
        }
        

        
        int posItrain=0;
        int negItrain=0;
        int testI=0;
        int trainI=0;
        for(int i=0;i<instanceN;i++)
        {
            double tempL=originData->at(i)->back();
            if(tempL==1.0)
            {
                if(posItrain<posNtrain)
                {
                    trainData->push_back(originData->at(i));
                    posItrain++;
                }
                else{
                    testData->push_back(originData->at(i));
                    testI++;
                }
            }
            else
            {
                if(negItrain<negNtrain)
                {
                    trainData->push_back(originData->at(i));
                    negItrain++;
                }
                else{
                    testData->push_back(originData->at(i));
                    testI++;
                }
            }
        }
        
//        // ensure the first training data trunck includes two classes
//        bool singleClass=true;
//        while(singleClass)
//        {
//            double firstL=trainData->at(0)->back();
//            for(int i=1;i<trainIndexEachUpdate[1];i++)
//            {
//                double anotherL=trainData->at(i)->back();
//                if(firstL!=anotherL)
//                {
//                    singleClass=false;
//                    break;
//                }
//            }
//            
//            if(singleClass)
//            {
//                for(int i=0;i<trainIndexEachUpdate[1]/2;i++)
//                {
//                    int idx1=(double)trainN*rand()/RAND_MAX;
//                    int idx2=(double)trainN*rand()/RAND_MAX;
//                    if(idx1!=idx2)
//                    {
//                        shared_ptr<vector<double> > tempSample=trainData->at(idx1);
//                        trainData->at(idx1)=trainData->at(idx2);
//                        trainData->at(idx2)=tempSample;
//                    }
//                }
//            }
//        }
        
        bool useBalance=true;
        cout<<"iteration "<<idx<<", data prepared"<<endl;
        RandomForest::ORForest<double> rf;
        rf.Init(20, 20,10);// tree number, depth, sample number in node
        rf.SetSamplingType(RandomForest::DownSamplingMajority);
        rf.SetBalanceType(RandomForest::DynamicImbalanceAdaptableBootstrap);
    
        vector<double>  Sensitivity0;
        vector<double>  Specificity0;
        vector<double>  Gmean0;
        vector<double>  Time0;
        
        vector<double>  compareSensitivity0;
        vector<double>  compareSpecificity0;
        vector<double>  compareGmean0;
        vector<double>  compareTime0;
        
        for(int it=0;it<trainIndexEachUpdate.size()-1;it++)
        {
            
            
            int addTrainStart=trainIndexEachUpdate[it];
            int addtrainEnd=trainIndexEachUpdate[it+1];
            int addTrainN=trainIndexEachUpdate[it+1]-trainIndexEachUpdate[it];
            
            vector<float> * predict_on;
            vector<float> * predict_off;
            
            ///get online training data
            shared_ptr<vector<shared_ptr<vector<double> > > > tempOnlineTrainData(new vector<shared_ptr<vector<double> > >);
            tempOnlineTrainData->reserve(addTrainN);
            for(int i=addTrainStart;i<addtrainEnd;i++)
            {
                tempOnlineTrainData->push_back(trainData->at(i));
            }
            
            time_t start0=clock();
            rf.Train(tempOnlineTrainData);
            double during0=(double)(clock()-start0)/CLOCKS_PER_SEC;
            Time0.push_back(during0);
            
            double oobe=rf.GetAverageOOBE();
            double balancedoobe=rf.GetAverageBalancedOOBE();
            rf.Predict(testData,&predict_on);
            
            /// get offline training data
            shared_ptr<vector<shared_ptr<vector<double> > > > tempOfflineTrainData(new vector<shared_ptr<vector<double> > >);
            tempOfflineTrainData->reserve(addtrainEnd);
            for(int i=0;i<addtrainEnd;i++)
            {
                tempOfflineTrainData->push_back(trainData->at(i));
            }
            
            RandomForest::ORForest<double> offrf;
            offrf.Init(20, 20, 10);
            offrf.SetSamplingType(RandomForest::DownSamplingMajority);
            offrf.SetBalanceType(RandomForest::SingleParameterBoostrap);
            offrf.DisableOnlineUpdate();
            time_t start1=clock();
            offrf.Train(tempOfflineTrainData);
            double during1=(double)(clock()-start1)/CLOCKS_PER_SEC;
            compareTime0.push_back(during1);
            
            offrf.Predict(testData, &predict_off);

            int PosN=0;
            int NegN=0;
            int correctPosPredict_on=0;
            int correctNegPredict_on=0;
            int correctPosPredict_off=0;
            int correctNegPredict_off=0;
            
            int posPredict_on=0;
            int TP_on=0;
            int negPredict_on=0;
            int TN_on=0;
            int posPredict_off=0;
            int TP_off=0;
            int negPredict_off=0;
            int TN_off=0;
            for(int i=0;i<testN;i++)
            {
                
                double realLabel=testData->at(i)->back();
                double predictLabel_on=predict_on->at(i);
                double predictLabel_off=predict_off->at(i);
                
                if(realLabel>=0.5)
                {
                    PosN++;
                    if(predictLabel_on>=0.5 )correctPosPredict_on++;
                    if(predictLabel_off>=0.5)correctPosPredict_off++;
                }
                else{
                    NegN++;
                    if(predictLabel_on<0.5)correctNegPredict_on++;
                    if(predictLabel_off<0.5)correctNegPredict_off++;
                }
                
                if(predictLabel_on>=0.5)
                {
                    posPredict_on++;
                    if(realLabel>=0.5) TP_on++;
                }
                else{
                    negPredict_on++;
                    if(realLabel<0.5) TN_on++;
                }
                
                if(predictLabel_off>=0.5)
                {
                    posPredict_off++;
                    if(realLabel>=0.5) TP_off++;
                }
                else{
                    negPredict_off++;
                    if(realLabel<0.5) TN_off++;
                }
            }
            
            double sensitivity_on=(double)correctPosPredict_on/PosN;
            double specificity_on=(double)correctNegPredict_on/NegN;
            double gMean_on=sqrt(sensitivity_on*specificity_on);
            
            double sensitivity_off=(double)correctPosPredict_off/PosN;
            double specificity_off=(double)correctNegPredict_off/NegN;
            double gMean_off=sqrt(sensitivity_off*specificity_off);
            
            Sensitivity0.push_back(sensitivity_on);
            Specificity0.push_back(specificity_on);
            Gmean0.push_back(gMean_on);
            
            compareSensitivity0.push_back(sensitivity_off);
            compareSpecificity0.push_back(specificity_off);
            compareGmean0.push_back(gMean_off);
        }
        Sensitivity.push_back(Sensitivity0);
        Specificity.push_back(Specificity0);
        Gmean.push_back(Gmean0);

        compareSensitivity.push_back(compareSensitivity0);
        compareSpecificity.push_back(compareSpecificity0);
        compareGmean.push_back(compareGmean0);
    }
}

void AbstractTestExample::PrintPerformance()
{
    cout<<"comparison between online bagging and offline bagging"<<endl;
    cout<<"sampleNumber ";
    cout<<"SensitivityMean std SpecificityMean std GmeanMean std ";
    cout<<"compareSensitivityMean std compareSpecificityMean std  compareGmeanMean std"<<endl;

    vector<double> SensitivityMean;
    vector<double> SensitivityStd;
    vector<double> SpecificityMean;
    vector<double> SpecificityStd;
    vector<double> GmeanMean;
    vector<double> GmeanStd;
    
    
    vector<double> compareSensitivityMean;
    vector<double> compareSensitivityStd;
    vector<double> compareSpecificityMean;
    vector<double> compareSpecificityStd;
    vector<double> compareGmeanMean;
    vector<double> compareGmeanStd;
    
    GetMeanAndStd(Sensitivity, &SensitivityMean, &SensitivityStd);
    GetMeanAndStd(Specificity, &SpecificityMean, &SpecificityStd);
    GetMeanAndStd(Gmean, &GmeanMean, &GmeanStd);
    
    GetMeanAndStd(compareSensitivity, &compareSensitivityMean, &compareSensitivityStd);
    GetMeanAndStd(compareSpecificity, &compareSpecificityMean, &compareSpecificityStd);
    GetMeanAndStd(compareGmean, &compareGmeanMean, &compareGmeanStd);

    for(int i=0;i<SensitivityMean.size();i++)
    {
        cout<< std::setw(4)<< std::fixed<< std::setprecision(2)<< (double)trainIndexEachUpdate.at(i+1)/trainN<<" ";
        cout<< std::setw(6)<< std::fixed<< std::setprecision(4)<< SensitivityMean[i]<<" ";
        cout<< std::setw(6)<< std::fixed<< std::setprecision(4)<< SensitivityStd[i]<<" ";
        cout<< std::setw(6)<< std::fixed<< std::setprecision(4)<< SpecificityMean[i]<<" ";
        cout<< std::setw(6)<< std::fixed<< std::setprecision(4)<< SpecificityStd[i]<<" ";
        cout<< std::setw(6)<< std::fixed<< std::setprecision(4)<< GmeanMean[i]<<" ";
        cout<< std::setw(6)<< std::fixed<< std::setprecision(4)<< GmeanStd[i]<<"     ";
        
        cout<< std::setw(6)<< std::fixed<< std::setprecision(4)<< compareSensitivityMean[i]<<" ";
        cout<< std::setw(6)<< std::fixed<< std::setprecision(4)<< compareSensitivityStd[i]<<" ";
        cout<< std::setw(6)<< std::fixed<< std::setprecision(4)<< compareSpecificityMean[i]<<" ";
        cout<< std::setw(6)<< std::fixed<< std::setprecision(4)<< compareSpecificityStd[i]<<" ";
        cout<< std::setw(6)<< std::fixed<< std::setprecision(4)<< compareGmeanMean[i]<<" ";
        cout<< std::setw(6)<< std::fixed<< std::setprecision(4)<< compareGmeanStd[i]<<" "<<endl;
    }
}
