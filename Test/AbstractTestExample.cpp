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

void AbstractTestExample::GenerateTrainAndTestData()
{
    std::shared_ptr<std::vector<int> > posIndex(new std::vector<int>);
    std::shared_ptr<std::vector<int> > negIndex(new std::vector<int>);
    
    std::shared_ptr<std::vector<std::shared_ptr<std::vector<double> > > > tempTrainData(new std::vector<std::shared_ptr<std::vector<double> > >);
    std::shared_ptr<std::vector<std::shared_ptr<std::vector<double> > > > tempTestData(new std::vector<std::shared_ptr<std::vector<double> > >);
    tempTrainData->reserve(trainN);
    tempTestData->reserve(testN);
    
    std::shared_ptr<std::vector<bool> > testMask(new std::vector<bool>);
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

void AbstractTestExample::SetTrainDataOnline(int startIndex, int endIndex)
{
    trainIndexEachUpdate.resize(endIndex-startIndex+2);
    trainIndexEachUpdate.at(0)=0;
    for(int i=0;i<=endIndex-startIndex;i++)
    {
        trainIndexEachUpdate[1+i]=startIndex+i;
    }
}

void AbstractTestExample::Run(int maxIter)
{
    Sensitivity.clear();
    Specificity.clear();
    Gmean.clear();
    Time.clear();
    
    compareSensitivity.clear();
    compareSpecificity.clear();
    compareGmean.clear();
    compareTime.clear();
    
    for(int idx=0;idx<maxIter;idx++)
    {
        for(int i=0;i<instanceN/2;i++)
        {
            int idx1=instanceN;
            while(idx1==instanceN) idx1=(double)instanceN*rand()/RAND_MAX;
            int idx2=instanceN;
            while(idx2==instanceN) idx2=(double)instanceN*rand()/RAND_MAX;
            if(idx1!=idx2)
            {
                std::shared_ptr<std::vector<double> > tempSample=originData->at(idx1);
                originData->at(idx1)=originData->at(idx2);
                originData->at(idx2)=tempSample;
            }
        }
        
        int posItrain=0;
        int negItrain=0;
        int testI=0;
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
        
        std::cout<<"iteration "<<idx<<", data prepared"<< std::endl;
        RandomForest::ORForest<double> rf;
        rf.Init(50, 20,10);// tree number, depth, sample number in node
        rf.SetSamplingType(RandomForest::DownSamplingMajority);
        rf.SetBalanceType(RandomForest::DynamicImbalanceAdaptableBootstrap);
        
        std::vector<double>  Sensitivity0;
        std::vector<double>  Specificity0;
        std::vector<double>  Gmean0;
        std::vector<double>  Time0;
        
        std::vector<double>  compareSensitivity0;
        std::vector<double>  compareSpecificity0;
        std::vector<double>  compareGmean0;
        std::vector<double>  compareTime0;
        
        for(int it=0;it<trainIndexEachUpdate.size()-1;it++)
        {
            
            
            int addTrainStart=trainIndexEachUpdate[it];
            int addtrainEnd=trainIndexEachUpdate[it+1];
            int addTrainN=trainIndexEachUpdate[it+1]-trainIndexEachUpdate[it];
            
            std::vector<float> * predict_on;
            std::vector<float> * predict_off;
            
            ///get online training data
            std::shared_ptr<std::vector<std::shared_ptr<std::vector<double> > > > tempOnlineTrainData(new std::vector<std::shared_ptr<std::vector<double> > >);
            tempOnlineTrainData->reserve(addTrainN);
            for(int i=addTrainStart;i<addtrainEnd;i++)
            {
                tempOnlineTrainData->push_back(trainData->at(i));
            }
            
            time_t start0=clock();
            rf.Train(tempOnlineTrainData);
            double during0=(double)(clock()-start0)/CLOCKS_PER_SEC;
            Time0.push_back(during0);
            
            rf.Predict(testData,&predict_on);
            
            /// get offline training data
            std::shared_ptr<std::vector<std::shared_ptr<std::vector<double> > > > tempOfflineTrainData(new std::vector<std::shared_ptr<std::vector<double> > >);
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
    std::cout<<"comparison between online bagging and offline bagging"<< std::endl;
    std::cout<<"sampleNumber ";
    std::cout<<"SensitivityMean std SpecificityMean std GmeanMean std ";
    std::cout<<"compareSensitivityMean std compareSpecificityMean std  compareGmeanMean std"<< std::endl;
    
    std::vector<double> SensitivityMean;
    std::vector<double> SensitivityStd;
    std::vector<double> SpecificityMean;
    std::vector<double> SpecificityStd;
    std::vector<double> GmeanMean;
    std::vector<double> GmeanStd;
    
    
    std::vector<double> compareSensitivityMean;
    std::vector<double> compareSensitivityStd;
    std::vector<double> compareSpecificityMean;
    std::vector<double> compareSpecificityStd;
    std::vector<double> compareGmeanMean;
    std::vector<double> compareGmeanStd;
    
    GetMeanAndStd(Sensitivity, &SensitivityMean, &SensitivityStd);
    GetMeanAndStd(Specificity, &SpecificityMean, &SpecificityStd);
    GetMeanAndStd(Gmean, &GmeanMean, &GmeanStd);
    
    GetMeanAndStd(compareSensitivity, &compareSensitivityMean, &compareSensitivityStd);
    GetMeanAndStd(compareSpecificity, &compareSpecificityMean, &compareSpecificityStd);
    GetMeanAndStd(compareGmean, &compareGmeanMean, &compareGmeanStd);
    
    for(int i=0;i<SensitivityMean.size();i++)
    {
        std::cout<< std::setw(4)<< std::fixed<< std::setprecision(2)<< (double)trainIndexEachUpdate.at(i+1)/trainN<<" ";
        std::cout<< std::setw(6)<< std::fixed<< std::setprecision(4)<< SensitivityMean[i]<<" ";
        std::cout<< std::setw(6)<< std::fixed<< std::setprecision(4)<< SensitivityStd[i]<<" ";
        std::cout<< std::setw(6)<< std::fixed<< std::setprecision(4)<< SpecificityMean[i]<<" ";
        std::cout<< std::setw(6)<< std::fixed<< std::setprecision(4)<< SpecificityStd[i]<<" ";
        std::cout<< std::setw(6)<< std::fixed<< std::setprecision(4)<< GmeanMean[i]<<" ";
        std::cout<< std::setw(6)<< std::fixed<< std::setprecision(4)<< GmeanStd[i]<<"     ";
        
        std::cout<< std::setw(6)<< std::fixed<< std::setprecision(4)<< compareSensitivityMean[i]<<" ";
        std::cout<< std::setw(6)<< std::fixed<< std::setprecision(4)<< compareSensitivityStd[i]<<" ";
        std::cout<< std::setw(6)<< std::fixed<< std::setprecision(4)<< compareSpecificityMean[i]<<" ";
        std::cout<< std::setw(6)<< std::fixed<< std::setprecision(4)<< compareSpecificityStd[i]<<" ";
        std::cout<< std::setw(6)<< std::fixed<< std::setprecision(4)<< compareGmeanMean[i]<<" ";
        std::cout<< std::setw(6)<< std::fixed<< std::setprecision(4)<< compareGmeanStd[i]<<" "<< std::endl;
    }
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
            std::cout<<std::setw(4)<< std::fixed<< std::setprecision(2)<<percent<<"   "
                <<std::setw(4)<< std::fixed<< std::setprecision(2)<<imbalanceRatio<<std::endl;
        }
    }
    
    double meanImbalance=sum/(steps-1);
    double std=sqrt(sumsq/(steps-1)-meanImbalance*meanImbalance);
    double normalizedStd=std/meanImbalance;
    std::cout<<"meanImbalance "<<meanImbalance<< std::endl;
    std::cout<<"std "<<std<< std::endl;
    std::cout<<"normalizedStd "<<normalizedStd<< std::endl;
}

std::vector<double> AbstractTestExample::GetImbalanceRatio()
{
    std::vector<double> imbalanceRatioList;
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

int AbstractTestExample::GetTrainN() const
{
    return trainN;
};

void AbstractTestExample::GetMeanAndStd(std::vector<std::vector<double> > i_array, std::vector<double> * o_mean, std::vector<double> * o_std)
{
    int rows=i_array.size();
    int column=i_array.at(0).size();
    std::vector<double> sum(column);
    std::vector<double> sumSq(column);
    std::vector<double> mean(column);
    std::vector<double> std(column);
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

bool AbstractTestExample::LoadCTGDataSet()
{
    std::string fileName="../../data/CTG.txt";
    std::ifstream fileLoader(fileName.c_str());
    if(!fileLoader.is_open())
    {
        std::cout<<"open file failed. "<<fileName<<" not found"<<std::endl;
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
    std::shared_ptr<std::vector<std::shared_ptr<std::vector<double> > > > readData(new std::vector<std::shared_ptr<std::vector<double> > >);
    readData->reserve(instanceN);
    for(int i=0;i<instanceN;i++)
    {
        std::shared_ptr<std::vector<double> > tempSample(new std::vector<double>);
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
    std::cout<<"DataSet    Feature Value"<<std::endl;
    std::cout<<"CTG        "<<posLabel<<std::endl;
    UpdateDataInfo();
    return true;
}


bool AbstractTestExample::LoadWineDataSet()
{
    std::string fileName="../../data/winequality.data";
    std::ifstream fileLoader(fileName.c_str());
    if(!fileLoader.is_open())
    {
        std::cout<<"open file failed. "<<fileName<<" not found"<<std::endl;
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
    std::shared_ptr<std::vector<std::shared_ptr<std::vector<double> > > > readData(new std::vector<std::shared_ptr<std::vector<double> > >);
    readData->reserve(instanceN);
    for(int i=0;i<instanceN;i++)
    {
        std::shared_ptr<std::vector<double> > tempSample(new std::vector<double>);
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
    std::cout<<"DataSet    Feature Value"<<std::endl;
    std::cout<<"Wine       "<<posLabel<<std::endl;
    UpdateDataInfo();
    
    return true;
}


bool AbstractTestExample::LoadMuskDataSet()
{
    std::string fileName="../../data/musk1.data";
    std::ifstream fileLoader(fileName.c_str());
    if(!fileLoader.is_open())
    {
        std::cout<<"open file failed. "<<fileName<<" not found"<<std::endl;
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
    
    std::shared_ptr<std::vector<std::shared_ptr<std::vector<double> > > > readData(new std::vector<std::shared_ptr<std::vector<double> > >);
    readData->reserve(instanceN*2);
    
    int label;
    int posLabel=1;
    for(int i=0;i<instanceN;i++)
    {
        std::shared_ptr<std::vector<double> > tempSample(new std::vector<double>);
        tempSample->reserve(featureN+1);
        
        std::string str1,str2;
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
    std::cout<<"DataSet   Feature Value"<<std::endl;
    std::cout<<"Musk  "<<posLabel<<std::endl;
    UpdateDataInfo();
    return true;
}

bool AbstractTestExample::LoadBiodegDataSet()
{
    std::string fileName="../../data/biodeg.csv";
    std::ifstream fileLoader(fileName.c_str());
    if(!fileLoader.is_open())
    {
        std::cout<<"open file failed. "<<fileName<<" not found"<<std::endl;
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
    
    std::shared_ptr<std::vector<std::shared_ptr<std::vector<double> > > > readData(new std::vector<std::shared_ptr<std::vector<double> > >);
    readData->reserve(instanceN*2);
    
    std::string label;
    std::string posLabel="RB";
    for(int i=0;i<instanceN;i++)
    {
        std::shared_ptr<std::vector<double> > tempSample(new std::vector<double>);
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
    std::cout<<"DataSet   Feature Value"<<std::endl;
    std::cout<<"Biodeg  "<<posLabel<<std::endl;
    UpdateDataInfo();
    return true;
}

void AbstractTestExample::UpdateDataInfo()
{
    std::cout<<"postive   "<<positiveN<<std::endl;
    std::cout<<"negative  "<<negtiveN<<std::endl;
    std::cout<<"imbalance "<<std::setprecision(4)<<(double)negtiveN/positiveN<<std::endl;
    
    double testRate=0.2;
    testN=instanceN*testRate;
    trainN=instanceN-testN;
    posNtest=positiveN*testRate;
    negNtest=testN-posNtest;
    posNtrain=positiveN-posNtest;
    negNtrain=negtiveN-negNtest;
}

