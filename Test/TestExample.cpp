//
//  TestExample.cpp
//  ORF_test
//
//  Created by Guotai Wang on 03/12/2015.
//
//

#include "TestExample.h"
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

TestExample::TestExample()
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

TestExample::~TestExample()
{
    
}

bool TestExample::LoadData(DataSetName data)
{
    switch (data) {
        case CTG:
            return LoadCTGDataSet();
        case ABALONE:
            return LoadAbaloneDataSet();
        case CHESS:
            return LoadChessDataSet();
        case LETTER:
            return LoadLetterDataSet();
        case COVTYPE:
            return LoadCovtypeDataSet();
        case PAGE:
            return LoadPageDataSet();
        case WALL:
            return LoadWallDataSet();
        case WINE:
            return LoadWineDataSet();
        case YEAST:
            return LoadYeastDataSet();
        case IMAGE_SEG:
            return LoadImageSegDataSet();
        case TRANSFUSION:
            return LoadTransfusionDataSet();
        case CLIMATE:
            return LoadClimateDataSet();
        case CMC:
            return LoadCMCDataSet();
        case MESSIDOR:
            return LoadMessidorDataSet();
        case EEG:
            return LoadEEGDataSet();
        case DERMATOLOGY:
            return LoadDermatologyDataSet();
        case BANKNOTE:
            return LoadBanknoteDataSet();
        case BREASTCANCER:
            return LoadBreastcancerDataSet();
        case CAR:
            return LoadCarDataSet();
        case KIDNEYDISEASE:
            return LoadKidneyDisease();
        case HOUSEVOTES:
            return LoadHouseVotesDataSet();
        case CONNECT:
            return LoadConnectDataSet();
        case CRX:
            return LoadCRXDataSet();
        case SONAR:
            return LoadSonarDataSet();
        case BAND:
            return LoadBandDataSet();
        case ECOLI:
            return LoadEcoliDataSet();
        case FERTILITY:
            return LoadFertilityDataSet();
        case LMPROVE:
            return LoadLMProveDataSet();
        case GESTURE:
            return LoadGestureDataSet();
        case GLASS:
            return LoadGlassDataSet();
        case HEPATITIS:
            return LoadHepatitisDataSet();
        case LLPD:
            return LoadLLPDDataSet();
        case IONOSPHERE:
            return LoadIonosphereDataSet();
        case MONK:
            return LoadMonkDataSet();
        case MUSHROOM:
            return LoadMushroomDataSet();
        case MUSK:
            return LoadMuskDataSet();
        case OPTDIGITS:
            return LoadOptdigtsDataSet();
        case PARKINSON:
            return LoadParkinsonDataSet();
        case PHISHING:
            return LoadPhishingDataSet();
        case BIODEG:
            return LoadBiodegDataSet();
        default:
            break;
    }
}

bool TestExample::LoadCTGDataSet()
{
    string fileName="/Users/guotaiwang/Documents/workspace/wgtRandomForest/dataset/CTG.txt";
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

bool TestExample::LoadAbaloneDataSet()
{
    string fileName="/Users/guotaiwang/Documents/workspace/wgtRandomForest/dataset/abalone.data";
    ifstream fileLoader(fileName.c_str());
    if(!fileLoader.is_open())
    {
        cout<<"open file failed"<<endl;
        return false;
    }
    featureN=7;
    instanceN=4177;
    positiveN=0;
    negtiveN=0;
    
    featureTypeList.resize(featureN);
    for(int i=0;i<featureN;i++)
    {
        featureTypeList[i]= GAUSSIAN;
    }
    
    double label;
    double posLabel=19;// 20,15,19
    shared_ptr<vector<shared_ptr<vector<double> > > > readData(new vector<shared_ptr<vector<double> > >);
    readData->reserve(instanceN);
    for(int i=0;i<instanceN;i++)
    {
        shared_ptr<vector<double> > tempSample(new vector<double>);
        tempSample->reserve(featureN+1);
        char gender;
        fileLoader>>gender;
        for(int j=0;j<featureN;j++)
        {
            char comma;
            fileLoader>>comma;
            double tempV;
            fileLoader>>tempV;
            tempSample->push_back(tempV);
        }
        char comma;
        fileLoader>>comma;
        
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
    cout<<"Abalon     "<<posLabel<<endl;
    UpdateDataInfo();
    //GenerateTrainAndTestData();

    return true;
}

bool TestExample::LoadChessDataSet()
{
    string fileName="/Users/guotaiwang/Documents/workspace/wgtRandomForest/dataset/krkopt.data";
    ifstream fileLoader(fileName.c_str());
    if(!fileLoader.is_open())
    {
        cout<<"open file failed"<<endl;
        return false;
    }
    featureN=6;
    instanceN=28056;//28056;
    positiveN=0;
    negtiveN=0;
    
    featureTypeList.resize(featureN);
    for(int i=0;i<featureN;i++)
    {
        featureTypeList[i]= MULTINOMIAL;
    }
    
    string label;
    string posLabel="four";// seven, twelve
    shared_ptr<vector<shared_ptr<vector<double> > > > readData(new vector<shared_ptr<vector<double> > >);
    readData->reserve(instanceN);
    for(int i=0;i<instanceN;i++)
    {
        shared_ptr<vector<double> > tempSample(new vector<double>);
        tempSample->reserve(featureN+1);
    
        for(int j=0;j<featureN;j++)
        {
            char tempV;
            fileLoader>>tempV;

            char comma;
            fileLoader>>comma;
            tempSample->push_back((double)tempV);
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
    cout<<"Chess     "<<posLabel<<endl;
    UpdateDataInfo();
    //GenerateTrainAndTestData();

    return true;
}

bool TestExample::LoadLetterDataSet()
{
    string fileName="/Users/guotaiwang/Documents/workspace/wgtRandomForest/dataset/letter.data";
    ifstream fileLoader(fileName.c_str());
    if(!fileLoader.is_open())
    {
        cout<<"open file failed"<<endl;
        return false;
    }
    featureN=16;
    instanceN=20000;
    positiveN=0;
    negtiveN=0;
    
    featureTypeList.resize(featureN);
    for(int i=0;i<featureN;i++)
    {
        featureTypeList[i]= MULTINOMIAL;
    }
    
    char label;
    char posLabel='A';//A,C
    shared_ptr<vector<shared_ptr<vector<double> > > > readData(new vector<shared_ptr<vector<double> > >);
    readData->reserve(instanceN);
    for(int i=0;i<instanceN;i++)
    {
        shared_ptr<vector<double> > tempSample(new vector<double>);
        tempSample->reserve(featureN+1);
        
        fileLoader>>label;
        for(int j=0;j<featureN;j++)
        {
            char comma;
            fileLoader>>comma;
            
            int tempV;
            fileLoader>>tempV;

            tempSample->push_back((double)tempV);
        }

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
    cout<<"Letter     "<<posLabel<<endl;
    UpdateDataInfo();
    //GenerateTrainAndTestData();

    return true;
}

bool TestExample::LoadCovtypeDataSet()
{
    string fileName="/Users/guotaiwang/Documents/workspace/wgtRandomForest/dataset/covtype.data";
    ifstream fileLoader(fileName.c_str());
    if(!fileLoader.is_open())
    {
        cout<<"open file failed"<<endl;
        return false;
    }
    featureN=54;
    instanceN=581012;
    positiveN=0;
    negtiveN=0;
    
    featureTypeList.resize(featureN);
    for(int i=0;i<featureN;i++)
    {
        featureTypeList[i]= (i<10) ? GAUSSIAN : BENOULLI;
    }
    
    int label;
    int posLabel=7;// || label==5 || label==6)
    shared_ptr<vector<shared_ptr<vector<double> > > > readData(new vector<shared_ptr<vector<double> > >);
    readData->reserve(instanceN);
    for(int i=0;i<instanceN;i++)
    {
        shared_ptr<vector<double> > tempSample(new vector<double>);
        tempSample->reserve(featureN+1);
        
        for(int j=0;j<featureN;j++)
        {
            int tempV;
            fileLoader>>tempV;
            
            char comma;
            fileLoader>>comma;
            tempSample->push_back((double)tempV);
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
    cout<<"Covtype    "<<posLabel<<endl;
    UpdateDataInfo();
    //GenerateTrainAndTestData();

    return true;
}

bool TestExample::LoadPageDataSet()
{
    string fileName="/Users/guotaiwang/Documents/workspace/wgtRandomForest/dataset/page-blocks.data";
    ifstream fileLoader(fileName.c_str());
    if(!fileLoader.is_open())
    {
        cout<<"open file failed"<<endl;
        return false;
    }
    featureN=10;
    instanceN=5473;
    positiveN=0;
    negtiveN=0;
    
    featureTypeList.resize(featureN);
    for(int i=0;i<featureN;i++)
    {
        featureTypeList[i]= GAUSSIAN;
    }
    
    int label;
    int posLabel=2;//2,4
    shared_ptr<vector<shared_ptr<vector<double> > > > readData(new vector<shared_ptr<vector<double> > >);
    readData->reserve(instanceN);
    for(int i=0;i<instanceN;i++)
    {
        shared_ptr<vector<double> > tempSample(new vector<double>);
        tempSample->reserve(featureN+1);
        
        for(int j=0;j<featureN;j++)
        {
            if(j>=3 && j<=6)
            {
                double tempV;
                fileLoader>>tempV;
                tempSample->push_back(tempV);
            }
            else{
                int tempV;
                fileLoader>>tempV;
                tempSample->push_back((double)tempV);
            }
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
    cout<<"Page       "<<posLabel<<endl;
    UpdateDataInfo();
    //GenerateTrainAndTestData();

    return true;
}

bool TestExample::LoadWallDataSet()
{
    string fileName="/Users/guotaiwang/Documents/workspace/wgtRandomForest/dataset/wall_24.data";
    ifstream fileLoader(fileName.c_str());
    if(!fileLoader.is_open())
    {
        cout<<"open file failed"<<endl;
        return false;
    }
    featureN=24;
    instanceN=5456;
    positiveN=0;
    negtiveN=0;
    
    featureTypeList.resize(featureN);
    for(int i=0;i<featureN;i++)
    {
        featureTypeList[i]= GAUSSIAN;
    }
    
    string label;
    string posLabel="Slight-Left-Turn";// Slight-Right-Turn,  Slight-Left-Turn
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
    cout<<"Wall       "<<posLabel<<endl;
    UpdateDataInfo();
    //GenerateTrainAndTestData();

    return true;
}

bool TestExample::LoadWineDataSet()
{
    string fileName="/Users/guotaiwang/Documents/workspace/wgtRandomForest/dataset/winequality.data";
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

bool TestExample::LoadYeastDataSet()
{
    string fileName="/Users/guotaiwang/Documents/workspace/wgtRandomForest/dataset/yeast.data";
    ifstream fileLoader(fileName.c_str());
    if(!fileLoader.is_open())
    {
        cout<<"open file failed"<<endl;
        return false;
    }
    featureN=8;
    instanceN=1484;
    positiveN=0;
    negtiveN=0;
    
    featureTypeList.resize(featureN);
    for(int i=0;i<featureN;i++)
    {
        featureTypeList[i]= GAUSSIAN;
    }
    string label;
    string posLabel="MIT";//ME1, MIT
    shared_ptr<vector<shared_ptr<vector<double> > > > readData(new vector<shared_ptr<vector<double> > >);
    readData->reserve(instanceN);
    for(int i=0;i<instanceN;i++)
    {
        shared_ptr<vector<double> > tempSample(new vector<double>);
        tempSample->reserve(featureN+1);
        
        string tempStr;
        fileLoader>>tempStr;
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
    cout<<"Yeast     "<<posLabel<<endl;
    UpdateDataInfo();
    //GenerateTrainAndTestData();

    
    return true;
}

bool TestExample::LoadImageSegDataSet()
{
    string fileName="/Users/guotaiwang/Documents/workspace/wgtRandomForest/dataset/image_segment.data";
    ifstream fileLoader(fileName.c_str());
    if(!fileLoader.is_open())
    {
        cout<<"open file failed"<<endl;
        return false;
    }
    featureN=19;
    instanceN=2310;
    positiveN=0;
    negtiveN=0;
    
    featureTypeList.resize(featureN);
    for(int i=0;i<featureN;i++)
    {
        featureTypeList[i]= GAUSSIAN;
    }
    
    shared_ptr<vector<shared_ptr<vector<double> > > > readData(new vector<shared_ptr<vector<double> > >);
    readData->reserve(instanceN);
    
    string label;
    string posLabel="BRICKFACE";
    for(int i=0;i<instanceN;i++)
    {
        shared_ptr<vector<double> > tempSample(new vector<double>);
        tempSample->reserve(featureN+1);
        
        
        fileLoader>>label;
        
        for(int j=0;j<featureN;j++)
        {
            double tempV;
            fileLoader>>tempV;
            tempSample->push_back(tempV);
        }
        
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
    cout<<"ImageSeg   "<<posLabel<<endl;
    UpdateDataInfo();
    //GenerateTrainAndTestData();
    return true;
}


bool TestExample::LoadTransfusionDataSet()
{
    string fileName="/Users/guotaiwang/Documents/workspace/wgtRandomForest/dataset/transfusion.txt";
    ifstream fileLoader(fileName.c_str());
    if(!fileLoader.is_open())
    {
        cout<<"open file failed"<<endl;
        return false;
    }
    featureN=4;
    instanceN=748;
    positiveN=0;
    negtiveN=0;
    
    featureTypeList.resize(featureN);
    for(int i=0;i<featureN;i++)
    {
        featureTypeList[i]= GAUSSIAN;
    }
    
    shared_ptr<vector<shared_ptr<vector<double> > > > readData(new vector<shared_ptr<vector<double> > >);
    readData->reserve(instanceN);
    
    int label;
    int posLabel=1;
    for(int i=0;i<instanceN;i++)
    {
        shared_ptr<vector<double> > tempSample(new vector<double>);
        tempSample->reserve(featureN+1);
        
        
        
        
        for(int j=0;j<featureN;j++)
        {
            int tempV;
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
    cout<<"DataSet      Feature Value"<<endl;
    cout<<"Transfusion "<<posLabel<<endl;
    UpdateDataInfo();
    //GenerateTrainAndTestData();
    return true;
}

bool TestExample::LoadClimateDataSet()
{
    string fileName="/Users/guotaiwang/Documents/workspace/wgtRandomForest/dataset/climate.txt";
    ifstream fileLoader(fileName.c_str());
    if(!fileLoader.is_open())
    {
        cout<<"open file failed"<<endl;
        return false;
    }
    featureN=18;
    instanceN=540;
    positiveN=0;
    negtiveN=0;
    
    featureTypeList.resize(featureN);
    for(int i=0;i<featureN;i++)
    {
        featureTypeList[i]= GAUSSIAN;
    }
    
    shared_ptr<vector<shared_ptr<vector<double> > > > readData(new vector<shared_ptr<vector<double> > >);
    readData->reserve(instanceN);
    
    int label;
    int posLabel=0;
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
    cout<<"DataSet      Feature Value"<<endl;
    cout<<"Climate "<<posLabel<<endl;
    UpdateDataInfo();
    //GenerateTrainAndTestData();
    return true;
}

bool TestExample::LoadCMCDataSet()
{
    string fileName="/Users/guotaiwang/Documents/workspace/wgtRandomForest/dataset/cmc.data";
    ifstream fileLoader(fileName.c_str());
    if(!fileLoader.is_open())
    {
        cout<<"open file failed"<<endl;
        return false;
    }
    featureN=9;
    instanceN=1473;
    positiveN=0;
    negtiveN=0;
    
    featureTypeList.resize(featureN);
    for(int i=0;i<featureN;i++)
    {
        featureTypeList[i]= GAUSSIAN;
    }
    
    shared_ptr<vector<shared_ptr<vector<double> > > > readData(new vector<shared_ptr<vector<double> > >);
    readData->reserve(instanceN);
    
    int label;
    int posLabel=1;
    for(int i=0;i<instanceN;i++)
    {
        shared_ptr<vector<double> > tempSample(new vector<double>);
        tempSample->reserve(featureN+1);
        
        for(int j=0;j<featureN;j++)
        {
            int tempV;
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
    cout<<"CMC       "<<posLabel<<endl;
    UpdateDataInfo();
    //GenerateTrainAndTestData();
    return true;
}

bool TestExample::LoadMessidorDataSet()
{
    string fileName="/Users/guotaiwang/Documents/workspace/wgtRandomForest/dataset/messidor.txt";
    ifstream fileLoader(fileName.c_str());
    if(!fileLoader.is_open())
    {
        cout<<"open file failed"<<endl;
        return false;
    }
    featureN=18;
    instanceN=1151;
    positiveN=0;
    negtiveN=0;
    
    featureTypeList.resize(featureN);
    for(int i=0;i<featureN;i++)
    {
        featureTypeList[i]= GAUSSIAN;
    }
    
    shared_ptr<vector<shared_ptr<vector<double> > > > readData(new vector<shared_ptr<vector<double> > >);
    readData->reserve(instanceN);
    
    int label;
    int posLabel=0;
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
    cout<<"DataSet   Feature Value"<<endl;
    cout<<"Messidor  "<<posLabel<<endl;
    UpdateDataInfo();
    //GenerateTrainAndTestData();
    return true;
}

bool TestExample::LoadEEGDataSet()
{
    string fileName="/Users/guotaiwang/Documents/workspace/wgtRandomForest/dataset/eeg.data";
    ifstream fileLoader(fileName.c_str());
    if(!fileLoader.is_open())
    {
        cout<<"open file failed"<<endl;
        return false;
    }
    featureN=14;
    instanceN=14980;
    positiveN=0;
    negtiveN=0;
    
    featureTypeList.resize(featureN);
    for(int i=0;i<featureN;i++)
    {
        featureTypeList[i]= GAUSSIAN;
    }
    
    shared_ptr<vector<shared_ptr<vector<double> > > > readData(new vector<shared_ptr<vector<double> > >);
    readData->reserve(instanceN);
    
    int label;
    int posLabel=1;//0,1
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
    cout<<"DataSet   Feature Value"<<endl;
    cout<<"EEG  "<<posLabel<<endl;
    UpdateDataInfo();
    //GenerateTrainAndTestData();
    return true;
}

bool TestExample::LoadDermatologyDataSet()
{
    string fileName="/Users/guotaiwang/Documents/workspace/wgtRandomForest/dataset/dermatology.data";
    ifstream fileLoader(fileName.c_str());
    if(!fileLoader.is_open())
    {
        cout<<"open file failed"<<endl;
        return false;
    }
    featureN=34;
    instanceN=366;
    positiveN=0;
    negtiveN=0;
    
    featureTypeList.resize(featureN);
    for(int i=0;i<featureN;i++)
    {
        featureTypeList[i]= GAUSSIAN;
    }
    
    shared_ptr<vector<shared_ptr<vector<double> > > > readData(new vector<shared_ptr<vector<double> > >);
    readData->reserve(instanceN);
    
    int label;
    int posLabel=6;
    for(int i=0;i<instanceN;i++)
    {
        shared_ptr<vector<double> > tempSample(new vector<double>);
        tempSample->reserve(featureN+1);
        
        for(int j=0;j<featureN;j++)
        {
            int tempV;
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
    cout<<"Dermatology  "<<posLabel<<endl;
    UpdateDataInfo();
    //GenerateTrainAndTestData();
    return true;
}

bool TestExample::LoadBanknoteDataSet()
{
    string fileName="/Users/guotaiwang/Documents/workspace/wgtRandomForest/dataset/banknote.txt";
    ifstream fileLoader(fileName.c_str());
    if(!fileLoader.is_open())
    {
        cout<<"open file failed"<<endl;
        return false;
    }
    featureN=4;
    instanceN=1372;
    positiveN=0;
    negtiveN=0;
    
    featureTypeList.resize(featureN);
    for(int i=0;i<featureN;i++)
    {
        featureTypeList[i]= GAUSSIAN;
    }
    
    shared_ptr<vector<shared_ptr<vector<double> > > > readData(new vector<shared_ptr<vector<double> > >);
    readData->reserve(instanceN);
    
    int label;
    int posLabel=1;
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
    cout<<"Banknote  "<<posLabel<<endl;
    UpdateDataInfo();
    //GenerateTrainAndTestData();
    return true;
}

bool TestExample::LoadBreastcancerDataSet()
{
    string fileName="/Users/guotaiwang/Documents/workspace/wgtRandomForest/dataset/breastcancer.txt";
    ifstream fileLoader(fileName.c_str());
    if(!fileLoader.is_open())
    {
        cout<<"open file failed"<<endl;
        return false;
    }
    featureN=9;
    instanceN=699;
    positiveN=0;
    negtiveN=0;
    
    featureTypeList.resize(featureN);
    for(int i=0;i<featureN;i++)
    {
        featureTypeList[i]= GAUSSIAN;
    }
    
    shared_ptr<vector<shared_ptr<vector<double> > > > readData(new vector<shared_ptr<vector<double> > >);
    readData->reserve(instanceN);
    
    int label;
    int posLabel=4;
    for(int i=0;i<instanceN;i++)
    {
        shared_ptr<vector<double> > tempSample(new vector<double>);
        tempSample->reserve(featureN+1);
        
        for(int j=0;j<featureN;j++)
        {
            int id;
            fileLoader>>id;
            
            int tempV;
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
    cout<<"Breastcancer  "<<posLabel<<endl;
    UpdateDataInfo();
    //GenerateTrainAndTestData();
    return true;
}

bool TestExample::LoadCarDataSet()
{
    string fileName="/Users/guotaiwang/Documents/workspace/wgtRandomForest/dataset/car.data";
    ifstream fileLoader(fileName.c_str());
    if(!fileLoader.is_open())
    {
        cout<<"open file failed"<<endl;
        return false;
    }
    featureN=6;
    instanceN=1728;
    positiveN=0;
    negtiveN=0;
    
    featureTypeList.resize(featureN);
    for(int i=0;i<featureN;i++)
    {
        featureTypeList[i]= GAUSSIAN;
    }
    
    shared_ptr<vector<shared_ptr<vector<double> > > > readData(new vector<shared_ptr<vector<double> > >);
    readData->reserve(instanceN);
    
    string label;
    string posLabel="acc"; //acc, good, vgood
    for(int i=0;i<instanceN;i++)
    {
        shared_ptr<vector<double> > tempSample(new vector<double>);
        tempSample->reserve(featureN+1);
        
        for(int j=0;j<featureN;j++)
        {
            string tempV;
            fileLoader>>tempV;
            double v=String2Double(tempV);
            tempSample->push_back(v);
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
    cout<<"Car  "<<posLabel<<endl;
    UpdateDataInfo();
    //GenerateTrainAndTestData();
    return true;
}

bool TestExample::LoadKidneyDisease()
{
    string fileName="/Users/guotaiwang/Documents/workspace/wgtRandomForest/dataset/kidneydisease.data";
    ifstream fileLoader(fileName.c_str());
    if(!fileLoader.is_open())
    {
        cout<<"open file failed"<<endl;
        return false;
    }
    featureN=24;
    instanceN=400;
    positiveN=0;
    negtiveN=0;
    
    featureTypeList.resize(featureN);
    for(int i=0;i<featureN;i++)
    {
        featureTypeList[i]= GAUSSIAN;
    }
    
    shared_ptr<vector<shared_ptr<vector<double> > > > readData(new vector<shared_ptr<vector<double> > >);
    readData->reserve(instanceN);
    
    string label;
    string posLabel="notckd"; //ckd
    for(int i=0;i<instanceN;i++)
    {
        shared_ptr<vector<double> > tempSample(new vector<double>);
        tempSample->reserve(featureN+1);
        
        for(int j=0;j<featureN;j++)
        {
            string tempV;
            fileLoader>>tempV;
            double v=String2Double(tempV);
            tempSample->push_back(v);
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
    cout<<"KidneyDisease  "<<posLabel<<endl;
    UpdateDataInfo();
    //GenerateTrainAndTestData();
    return true;
}

bool TestExample::LoadHouseVotesDataSet()
{
    string fileName="/Users/guotaiwang/Documents/workspace/wgtRandomForest/dataset/house-votes-84.data";
    ifstream fileLoader(fileName.c_str());
    if(!fileLoader.is_open())
    {
        cout<<"open file failed"<<endl;
        return false;
    }
    featureN=16;
    instanceN=435;
    positiveN=0;
    negtiveN=0;
    
    featureTypeList.resize(featureN);
    for(int i=0;i<featureN;i++)
    {
        featureTypeList[i]= MULTINOMIAL;
    }
    
    shared_ptr<vector<shared_ptr<vector<double> > > > readData(new vector<shared_ptr<vector<double> > >);
    readData->reserve(instanceN);
    
    string label;
    string posLabel="republican"; //democrat
    for(int i=0;i<instanceN;i++)
    {
        shared_ptr<vector<double> > tempSample(new vector<double>);
        tempSample->reserve(featureN+1);
        
        fileLoader>>label;
        
        for(int j=0;j<featureN;j++)
        {
            char tempV;
            fileLoader>>tempV;
            tempSample->push_back(tempV);
        }
        
        
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
    cout<<"HouseVotes  "<<posLabel<<endl;
    UpdateDataInfo();
    //GenerateTrainAndTestData();
    return true;
}

bool TestExample::LoadConnectDataSet()
{
    string fileName="/Users/guotaiwang/Documents/workspace/wgtRandomForest/dataset/connect-4.data";
    ifstream fileLoader(fileName.c_str());
    if(!fileLoader.is_open())
    {
        cout<<"open file failed"<<endl;
        return false;
    }
    featureN=42;
    instanceN=67557;
    positiveN=0;
    negtiveN=0;
    
    featureTypeList.resize(featureN);
    for(int i=0;i<featureN;i++)
    {
        featureTypeList[i]= MULTINOMIAL;
    }
    
    shared_ptr<vector<shared_ptr<vector<double> > > > readData(new vector<shared_ptr<vector<double> > >);
    readData->reserve(instanceN);
    
    string label;
    string posLabel="draw"; //loss, draw
    for(int i=0;i<instanceN;i++)
    {
        shared_ptr<vector<double> > tempSample(new vector<double>);
        tempSample->reserve(featureN+1);
        
        
        for(int j=0;j<featureN;j++)
        {
            char tempV;
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
    cout<<"Connect  "<<posLabel<<endl;
    UpdateDataInfo();
    //GenerateTrainAndTestData();
    return true;
}

bool TestExample::LoadCRXDataSet()
{
    string fileName="/Users/guotaiwang/Documents/workspace/wgtRandomForest/dataset/crx.data";
    ifstream fileLoader(fileName.c_str());
    if(!fileLoader.is_open())
    {
        cout<<"open file failed"<<endl;
        return false;
    }
    featureN=15;
    instanceN=690;
    positiveN=0;
    negtiveN=0;
    
    featureTypeList.resize(featureN);
    for(int i=0;i<featureN;i++)
    {
        featureTypeList[i]= MULTINOMIAL;
    }
    
    shared_ptr<vector<shared_ptr<vector<double> > > > readData(new vector<shared_ptr<vector<double> > >);
    readData->reserve(instanceN);
    
    string label;
    string posLabel="+"; //+,-
    for(int i=0;i<instanceN;i++)
    {
        shared_ptr<vector<double> > tempSample(new vector<double>);
        tempSample->reserve(featureN+1);
        
        
        for(int j=0;j<featureN;j++)
        {
            string tempV;
            fileLoader>>tempV;
            double v=String2Double(tempV);
            tempSample->push_back(v);
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
    cout<<"CRX  "<<posLabel<<endl;
    UpdateDataInfo();
    //GenerateTrainAndTestData();
    return true;
}


bool TestExample::LoadSonarDataSet()
{
    string fileName="/Users/guotaiwang/Documents/workspace/wgtRandomForest/dataset/sonar.data";
    ifstream fileLoader(fileName.c_str());
    if(!fileLoader.is_open())
    {
        cout<<"open file failed"<<endl;
        return false;
    }
    featureN=60;
    instanceN=208;
    positiveN=0;
    negtiveN=0;
    
    featureTypeList.resize(featureN);
    for(int i=0;i<featureN;i++)
    {
        featureTypeList[i]= GAUSSIAN;
    }
    
    shared_ptr<vector<shared_ptr<vector<double> > > > readData(new vector<shared_ptr<vector<double> > >);
    readData->reserve(instanceN);
    
    char label;
    char posLabel='R'; //R,M
    for(int i=0;i<instanceN;i++)
    {
        shared_ptr<vector<double> > tempSample(new vector<double>);
        tempSample->reserve(featureN+1);
        
        
        for(int j=0;j<featureN;j++)
        {
            float tempV;
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
    cout<<"SONAR  "<<posLabel<<endl;
    UpdateDataInfo();
    //GenerateTrainAndTestData();
    return true;
}

bool TestExample::LoadBandDataSet()
{
    string fileName="/Users/guotaiwang/Documents/workspace/wgtRandomForest/dataset/bands.data";
    ifstream fileLoader(fileName.c_str());
    if(!fileLoader.is_open())
    {
        cout<<"open file failed"<<endl;
        return false;
    }
    featureN=38;
    instanceN=512;
    positiveN=0;
    negtiveN=0;
    
    featureTypeList.resize(featureN);
    for(int i=0;i<featureN;i++)
    {
        featureTypeList[i]= MULTINOMIAL;
    }
    
    shared_ptr<vector<shared_ptr<vector<double> > > > readData(new vector<shared_ptr<vector<double> > >);
    readData->reserve(instanceN);
    
    string label;
    string posLabel="band"; //band, noband
    for(int i=0;i<instanceN;i++)
    {
        shared_ptr<vector<double> > tempSample(new vector<double>);
        tempSample->reserve(featureN+1);
        
        string id;
        fileLoader>>id;
        for(int j=0;j<featureN;j++)
        {
            string tempV;
            fileLoader>>tempV;
            double v=String2Double(tempV);
            tempSample->push_back(v);
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
    cout<<"Band  "<<posLabel<<endl;
    UpdateDataInfo();
    //GenerateTrainAndTestData();
    return true;
}

bool TestExample::LoadEcoliDataSet()
{
    string fileName="/Users/guotaiwang/Documents/workspace/wgtRandomForest/dataset/ecoli.data";
    ifstream fileLoader(fileName.c_str());
    if(!fileLoader.is_open())
    {
        cout<<"open file failed"<<endl;
        return false;
    }
    featureN=7;
    instanceN=336;
    positiveN=0;
    negtiveN=0;
    
    featureTypeList.resize(featureN);
    for(int i=0;i<featureN;i++)
    {
        featureTypeList[i]= MULTINOMIAL;
    }
    
    shared_ptr<vector<shared_ptr<vector<double> > > > readData(new vector<shared_ptr<vector<double> > >);
    readData->reserve(instanceN);
    
    string label;
    string posLabel="im"; //band, noband
    for(int i=0;i<instanceN;i++)
    {
        shared_ptr<vector<double> > tempSample(new vector<double>);
        tempSample->reserve(featureN+1);
        
        string id;
        fileLoader>>id;
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
    cout<<"Ecoli  "<<posLabel<<endl;
    UpdateDataInfo();
    //GenerateTrainAndTestData();
    return true;
}

bool TestExample::LoadFertilityDataSet()
{
    string fileName="/Users/guotaiwang/Documents/workspace/wgtRandomForest/dataset/fertility_Diagnosis.txt";
    ifstream fileLoader(fileName.c_str());
    if(!fileLoader.is_open())
    {
        cout<<"open file failed"<<endl;
        return false;
    }
    featureN=9;
    instanceN=100;
    positiveN=0;
    negtiveN=0;
    
    featureTypeList.resize(featureN);
    for(int i=0;i<featureN;i++)
    {
        featureTypeList[i]= GAUSSIAN;
    }
    
    shared_ptr<vector<shared_ptr<vector<double> > > > readData(new vector<shared_ptr<vector<double> > >);
    readData->reserve(instanceN);
    
    char label;
    char posLabel='O'; //band, noband
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
    cout<<"Fertility  "<<posLabel<<endl;
    UpdateDataInfo();
    //GenerateTrainAndTestData();
    return true;
}


bool TestExample::LoadLMProveDataSet()
{
    string fileName="/Users/guotaiwang/Documents/workspace/wgtRandomForest/dataset/ml-prove/train.csv";
    ifstream fileLoader(fileName.c_str());
    if(!fileLoader.is_open())
    {
        cout<<"open file failed"<<endl;
        return false;
    }
    featureN=56;
    instanceN=3059;
    positiveN=0;
    negtiveN=0;
    
    featureTypeList.resize(featureN);
    for(int i=0;i<featureN;i++)
    {
        featureTypeList[i]= GAUSSIAN;
    }
    
    shared_ptr<vector<shared_ptr<vector<double> > > > readData(new vector<shared_ptr<vector<double> > >);
    readData->reserve(instanceN);
    
    int label;
    int posLabel=1; //band, noband
    for(int i=0;i<instanceN;i++)
    {
        shared_ptr<vector<double> > tempSample(new vector<double>);
        tempSample->reserve(featureN+1);
        
        for(int j=0;j<featureN;j++)
        {
            double tempV;
            fileLoader>>tempV;
            tempSample->push_back(tempV);
            
//            if(tempV==0.27015)
//            {
//                int a=0;
//            }
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
    cout<<"LMProve  "<<posLabel<<endl;
    UpdateDataInfo();
    //GenerateTrainAndTestData();
    return true;
}

bool TestExample::LoadGestureDataSet()
{
    string fileName="/Users/guotaiwang/Documents/workspace/wgtRandomForest/dataset/gesture_phase_dataset/a1_raw.csv";
    ifstream fileLoader(fileName.c_str());
    if(!fileLoader.is_open())
    {
        cout<<"open file failed"<<endl;
        return false;
    }
    featureN=19;
    instanceN=1747;
    positiveN=0;
    negtiveN=0;
    
    featureTypeList.resize(featureN);
    for(int i=0;i<featureN;i++)
    {
        featureTypeList[i]= GAUSSIAN;
    }
    
    shared_ptr<vector<shared_ptr<vector<double> > > > readData(new vector<shared_ptr<vector<double> > >);
    readData->reserve(instanceN);
    
    string label;
    string posLabel="Stroke"; //Retraction, Rest, Stroke, Preparation
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
    cout<<"Gesture  "<<posLabel<<endl;
    UpdateDataInfo();
    //GenerateTrainAndTestData();
    return true;
}

bool TestExample::LoadGlassDataSet()
{
    string fileName="/Users/guotaiwang/Documents/workspace/wgtRandomForest/dataset/glass.data";
    ifstream fileLoader(fileName.c_str());
    if(!fileLoader.is_open())
    {
        cout<<"open file failed"<<endl;
        return false;
    }
    featureN=9;
    instanceN=214;
    positiveN=0;
    negtiveN=0;
    
    featureTypeList.resize(featureN);
    for(int i=0;i<featureN;i++)
    {
        featureTypeList[i]= GAUSSIAN;
    }
    
    shared_ptr<vector<shared_ptr<vector<double> > > > readData(new vector<shared_ptr<vector<double> > >);
    readData->reserve(instanceN);
    
    int label;
    int posLabel=1; //1,2
    for(int i=0;i<instanceN;i++)
    {
        shared_ptr<vector<double> > tempSample(new vector<double>);
        tempSample->reserve(featureN+1);
        
        int id;
        fileLoader>>id;
        
        char comma;
        fileLoader>>comma;
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
    cout<<"GLASS  "<<posLabel<<endl;
    UpdateDataInfo();
    //GenerateTrainAndTestData();
    return true;
}

bool TestExample::LoadHepatitisDataSet()
{
    string fileName="/Users/guotaiwang/Documents/workspace/wgtRandomForest/dataset/hepatitis/hepatitis_copy.data";
    ifstream fileLoader(fileName.c_str());
    if(!fileLoader.is_open())
    {
        cout<<"open file failed"<<endl;
        return false;
    }
    featureN=19;
    instanceN=155;
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
    int posLabel=2; //1,2
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
    cout<<"HEPATITIS  "<<posLabel<<endl;
    UpdateDataInfo();
    //GenerateTrainAndTestData();
    return true;
}

bool TestExample::LoadLLPDDataSet()
{
    string fileName="/Users/guotaiwang/Documents/workspace/wgtRandomForest/dataset/LLPD.csv";
    ifstream fileLoader(fileName.c_str());
    if(!fileLoader.is_open())
    {
        cout<<"open file failed"<<endl;
        return false;
    }
    featureN=10;
    instanceN=583;
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
    int posLabel=2; //1,2
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
    cout<<"LLPD  "<<posLabel<<endl;
    UpdateDataInfo();
    //GenerateTrainAndTestData();
    return true;
}

bool TestExample::LoadIonosphereDataSet()
{
    string fileName="/Users/guotaiwang/Documents/workspace/wgtRandomForest/dataset/ionosphere.data";
    ifstream fileLoader(fileName.c_str());
    if(!fileLoader.is_open())
    {
        cout<<"open file failed"<<endl;
        return false;
    }
    featureN=34;
    instanceN=351;
    positiveN=0;
    negtiveN=0;
    
    featureTypeList.resize(featureN);
    for(int i=0;i<featureN;i++)
    {
        featureTypeList[i]= GAUSSIAN;
    }
    
    shared_ptr<vector<shared_ptr<vector<double> > > > readData(new vector<shared_ptr<vector<double> > >);
    readData->reserve(instanceN*2);
    
    char label;
    char posLabel='b'; //g,b
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
    cout<<"Lonoshere  "<<posLabel<<endl;
    UpdateDataInfo();
    //GenerateTrainAndTestData();
    return true;
}

bool TestExample::LoadMonkDataSet()
{
    string fileName="/Users/guotaiwang/Documents/workspace/wgtRandomForest/dataset/monks-1.train";
    ifstream fileLoader(fileName.c_str());
    if(!fileLoader.is_open())
    {
        cout<<"open file failed"<<endl;
        return false;
    }
    featureN=6;
    instanceN=430;
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
    int posLabel=0;
    for(int i=0;i<instanceN;i++)
    {
        shared_ptr<vector<double> > tempSample(new vector<double>);
        tempSample->reserve(featureN+1);
        
        fileLoader>>label;
        for(int j=0;j<featureN;j++)
        {
            double tempV;
            fileLoader>>tempV;
            tempSample->push_back(tempV);
        }
        
        string id;
        fileLoader>>id;
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
    cout<<"MONK  "<<posLabel<<endl;
    UpdateDataInfo();
    //GenerateTrainAndTestData();
    return true;
}

bool TestExample::LoadMushroomDataSet()
{
    string fileName="/Users/guotaiwang/Documents/workspace/wgtRandomForest/dataset/mushroom.data";
    ifstream fileLoader(fileName.c_str());
    if(!fileLoader.is_open())
    {
        cout<<"open file failed"<<endl;
        return false;
    }
    featureN=22;
    instanceN=8124;
    positiveN=0;
    negtiveN=0;
    
    featureTypeList.resize(featureN);
    for(int i=0;i<featureN;i++)
    {
        featureTypeList[i]= GAUSSIAN;
    }
    
    shared_ptr<vector<shared_ptr<vector<double> > > > readData(new vector<shared_ptr<vector<double> > >);
    readData->reserve(instanceN*2);
    
    char label;
    char posLabel='p';
    for(int i=0;i<instanceN;i++)
    {
        shared_ptr<vector<double> > tempSample(new vector<double>);
        tempSample->reserve(featureN+1);
        
        fileLoader>>label;
        for(int j=0;j<featureN;j++)
        {
            char tempV;
            fileLoader>>tempV;
            tempSample->push_back(tempV);
            
            char comma;
            fileLoader>>comma;
        }
        
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
    cout<<"Mushroom  "<<posLabel<<endl;
    UpdateDataInfo();
    //GenerateTrainAndTestData();
    return true;
}

bool TestExample::LoadMuskDataSet()
{
    string fileName="/Users/guotaiwang/Documents/workspace/wgtRandomForest/dataset/musk1.data";
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

bool TestExample::LoadOptdigtsDataSet()
{
    string fileName="/Users/guotaiwang/Documents/workspace/wgtRandomForest/dataset/optdigits/optdigitsall.data";
    ifstream fileLoader(fileName.c_str());
    if(!fileLoader.is_open())
    {
        cout<<"open file failed"<<endl;
        return false;
    }
    featureN=64;
    instanceN=5620;
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
    int posLabel=7;
    for(int i=0;i<instanceN;i++)
    {
        shared_ptr<vector<double> > tempSample(new vector<double>);
        tempSample->reserve(featureN+1);
        
        for(int j=0;j<featureN;j++)
        {
            int tempV;
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
    cout<<"Optdigits  "<<posLabel<<endl;
    UpdateDataInfo();
    //GenerateTrainAndTestData();
    return true;
}


bool TestExample::LoadParkinsonDataSet()
{
    string fileName="/Users/guotaiwang/Documents/workspace/wgtRandomForest/dataset/parkinson/train_data.txt";
    ifstream fileLoader(fileName.c_str());
    if(!fileLoader.is_open())
    {
        cout<<"open file failed"<<endl;
        return false;
    }
    featureN=27;
    instanceN=1040;
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
        
        int id;
        fileLoader>>id;
        
        char comma;
        fileLoader>>comma;
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
    cout<<"Parkinson  "<<posLabel<<endl;
    UpdateDataInfo();
    //GenerateTrainAndTestData();
    return true;
}

bool TestExample::LoadPhishingDataSet()
{
    string fileName="/Users/guotaiwang/Documents/workspace/wgtRandomForest/dataset/phishing.arff";
    ifstream fileLoader(fileName.c_str());
    if(!fileLoader.is_open())
    {
        cout<<"open file failed"<<endl;
        return false;
    }
    featureN=30;
    instanceN=2456;
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
    int posLabel=-1;//1, -1
    for(int i=0;i<instanceN;i++)
    {
        shared_ptr<vector<double> > tempSample(new vector<double>);
        tempSample->reserve(featureN+1);
        
        for(int j=0;j<featureN;j++)
        {
            int tempV;
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
    cout<<"Phishing  "<<posLabel<<endl;
    UpdateDataInfo();
    //GenerateTrainAndTestData();
    return true;
}

bool TestExample::LoadBiodegDataSet()
{
    string fileName="/Users/guotaiwang/Documents/workspace/wgtRandomForest/dataset/biodeg.csv";
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


void TestExample::GenerateTrainAndTestData()
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

void TestExample::SetTrainDataChunk(double startPercent, double endPercent, double increasePercent)
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

void TestExample::UpdateDataInfo()
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
void TestExample::PrintDataInformation()
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

vector<double> TestExample::GetImbalanceRatio()
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
void TestExample::SetTrainDataOnline(int startIndex, int endIndex)
{
    trainIndexEachUpdate.resize(endIndex-startIndex+2);
    trainIndexEachUpdate.at(0)=0;
    for(int i=0;i<=endIndex-startIndex;i++)
    {
        trainIndexEachUpdate[1+i]=startIndex+i;
    }
}

void TestExample::Run(int MaxIter)
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

void TestExample::PrintPerformance()
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
