//
//  TestExample.h
//  ORF_test
//
//  Created by Guotai Wang on 03/12/2015.
//
//

#ifndef __ORF_test__TestExample__
#define __ORF_test__TestExample__

#include "ORForest.h"
#include "../Bayes/BayesEnsemble.h"
#include <iomanip>

void GetMeanAndStd(vector<vector<double> > i_array, vector<double> * o_mean, vector<double> * o_std);

enum DataSetName{CTG,ABALONE,CHESS,LETTER,COVTYPE,PAGE,WALL,WINE,YEAST,IMAGE_SEG,TRANSFUSION,
    CLIMATE,CMC,MESSIDOR,EEG,DERMATOLOGY,BANKNOTE, BREASTCANCER,CAR,KIDNEYDISEASE,HOUSEVOTES,
    CONNECT,CRX,SONAR, BAND, ECOLI,FERTILITY, LMPROVE, GESTURE, GLASS, HEPATITIS, LLPD, IONOSPHERE, MONK, MUSHROOM, MUSK, OPTDIGITS, PARKINSON, PHISHING, BIODEG};


class TestExample{
public:
    TestExample();
    ~TestExample();
    bool LoadData(DataSetName data);
    void UpdateDataInfo();
    void GenerateTrainAndTestData();
    void SetTrainDataChunk(double startPercent, double endPercent, double increasePercent);
    void SetTrainDataOnline(int startIndex, int endIndex);
    virtual void Run(int MaxIter);
    virtual void PrintPerformance();
    void PrintDataInformation();
    vector<double> GetImbalanceRatio();
    int GetTrainN(){return trainN;};
protected:

    bool LoadCTGDataSet();
    bool LoadAbaloneDataSet();
    bool LoadChessDataSet();
    bool LoadLetterDataSet();
    bool LoadCovtypeDataSet();
    bool LoadPageDataSet();
    bool LoadWallDataSet();
    bool LoadWineDataSet();
    bool LoadYeastDataSet();
    bool LoadImageSegDataSet();
    bool LoadTransfusionDataSet();
    bool LoadClimateDataSet();
    bool LoadCMCDataSet();
    bool LoadMessidorDataSet();
    bool LoadEEGDataSet();
    bool LoadDermatologyDataSet();
    bool LoadBanknoteDataSet();
    bool LoadBreastcancerDataSet();
    bool LoadCarDataSet();
    bool LoadKidneyDisease();
    bool LoadHouseVotesDataSet();
    bool LoadConnectDataSet();
    bool LoadCRXDataSet();
    bool LoadSonarDataSet();
    bool LoadBandDataSet();
    bool LoadEcoliDataSet();
    bool LoadFertilityDataSet();
    bool LoadLMProveDataSet();
    bool LoadGestureDataSet();
    bool LoadGlassDataSet();
    bool LoadHepatitisDataSet();
    bool LoadLLPDDataSet();
    bool LoadIonosphereDataSet();
    bool LoadMonkDataSet();
    bool LoadMushroomDataSet();
    bool LoadMuskDataSet();
    bool LoadOptdigtsDataSet();
    bool LoadParkinsonDataSet();
    bool LoadPhishingDataSet();
    bool LoadBiodegDataSet();
    int featureN;
    int instanceN;
    int positiveN;
    int negtiveN;
    int trainN;
    int testN;
    int posNtrain;
    int negNtrain;
    int posNtest;
    int negNtest;
    shared_ptr<vector<shared_ptr<vector<double> > > > originData;
    shared_ptr<vector<shared_ptr<vector<double> > > > trainData;
    shared_ptr<vector<shared_ptr<vector<double> > > > testData;
    vector<int> trainIndexEachUpdate;
    
    vector<vector<double> > imbalanceRatio;
    
    vector<vector<double> > Sensitivity;
    vector<vector<double> > Specificity;
    vector<vector<double> > Gmean;
    vector<vector<double> > Time;
    
    vector<vector<double> > compareSensitivity;
    vector<vector<double> > compareSpecificity;
    vector<vector<double> > compareGmean;
    vector<vector<double> > compareTime;
    


    vector<FeatureTypeName> featureTypeList;
};

#endif /* defined(__ORF_test__TestExample__) */
