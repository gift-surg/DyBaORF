//
//  AbstractTestExample.h
//  DyBaORF_test
//
//  Created by Guotai Wang on 03/12/2015.
//
//

#ifndef __ORF_test__AbstractTestExample__
#define __ORF_test__AbstractTestExample__

#include "ORForest.h"
#include <iomanip>

void GetMeanAndStd(vector<vector<double> > i_array, vector<double> * o_mean, vector<double> * o_std);

enum DataSetName{BIODEG, MUSK, CTG, WINE};
enum FeatureTypeName{GAUSSIAN, BENOULLI, MULTINOMIAL};

class AbstractTestExample{
public:
    AbstractTestExample();
    ~AbstractTestExample();
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
    bool LoadWineDataSet();
    bool LoadMuskDataSet();
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

#endif /* defined(__ORF_test__AbstractTestExample__) */
