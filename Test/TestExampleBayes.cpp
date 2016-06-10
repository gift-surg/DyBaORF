//
//  TestExampleBayes.cpp
//  ORF_test
//
//  Created by Guotai Wang on 08/12/2015.
//
//

#include "TestExampleBayes.h"

TestExampleBayes::TestExampleBayes()
{
    
}

TestExampleBayes::~TestExampleBayes()
{
    
}

void TestExampleBayes::Run(int MaxIter)
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
        cout<<"iteration "<<idx<<", data prepared"<<endl;
        BayesEnsemble<double> bysClassifiers;
        bysClassifiers.Init(20, true);
        bysClassifiers.SetClassifierFeatures(featureTypeList);

//        BayesEnsemble<double> bysClassifiers1;
//        bysClassifiers1.Init(20, false);
//        bysClassifiers1.SetClassifierFeatures(featureTypeList);
        
       
        vector<double>  Sensitivity0;
        vector<double>  Specificity0;
        vector<double>  Gmean0;
        vector<double>  Time0;
        
//        vector<double>  compareSensitivity0;
//        vector<double>  compareSpecificity0;
//        vector<double>  compareGmean0;
//        vector<double>  compareTime0;
        
        
//        time_t start=clock();
        for(int it=0;it<trainIndexEachUpdate.size()-1;it++)
        {
            int addTrainStart=trainIndexEachUpdate[it];
            int addtrainEnd=trainIndexEachUpdate[it+1];
            int addTrainN=trainIndexEachUpdate[it+1]-trainIndexEachUpdate[it];
            
            vector<double> predict;
            //vector<double> predict_compare;
            
            ///get online training data
            shared_ptr<vector<shared_ptr<vector<double> > > > tempOnlineTrainData(new vector<shared_ptr<vector<double> > >);
            tempOnlineTrainData->reserve(addTrainN);
            for(int i=addTrainStart;i<addtrainEnd;i++)
            {
                tempOnlineTrainData->push_back(trainData->at(i));
            }
            
            time_t start0=clock();
            bysClassifiers.Train(tempOnlineTrainData);
            double trainTime0=(double)(clock()-start0)/CLOCKS_PER_SEC;
            Time0.push_back(trainTime0);
            time_t start1=clock();
            bysClassifiers.Predict(testData,&predict);
            double testTime0=(double)(clock()-start1)/CLOCKS_PER_SEC;
            
            
//            time_t start2=clock();
//            bysClassifiers1.Train(tempOnlineTrainData);
//            double trainTime1=(double)(clock()-start2)/CLOCKS_PER_SEC;
//            
//            time_t start3=clock();
//            bysClassifiers1.Predict(testData, &predict_compare);
//            double testTime1=(double)(clock()-start3)/CLOCKS_PER_SEC;

//            cout<<"Train Time "<<trainTime0<<" "<<trainTime1<<" Test time "<<testTime0<<" "<<testTime1<<endl;
            int PosN=0;
            int NegN=0;
            int correctPosPredict=0;
            int correctNegPredict=0;
            int correctPosPredict_compare=0;
            int correctNegPredict_compare=0;
            
            int posPredict=0;
            int TP =0;
            int negPredict=0;
            int TN =0;
            int posPredict_compare=0;
            int TP_compare=0;
            int negPredict_compare=0;
            int TN_compare=0;
            for(int i=0;i<testN;i++)
            {
                
                double realLabel=testData->at(i)->back();
                double predictLabel=predict[i];
                //double predictLabel_compare=predict_compare[i];
                
                if(realLabel>=0.5)
                {
                    PosN++;
                    if(predictLabel>=0.5 )correctPosPredict++;
                    //if(predictLabel_compare>=0.5)correctPosPredict_compare++;
                }
                else{
                    NegN++;
                    if(predictLabel<0.5)correctNegPredict++;
                    //if(predictLabel_compare<0.5)correctNegPredict_compare++;
                }
                
                if(predictLabel>=0.5)
                {
                    posPredict ++;
                    if(realLabel>=0.5) TP ++;
                }
                else{
                    negPredict ++;
                    if(realLabel<0.5) TN ++;
                }
                
//                if(predictLabel_compare>=0.5)
//                {
//                    posPredict_compare++;
//                    if(realLabel>=0.5) TP_compare++;
//                }
//                else{
//                    negPredict_compare++;
//                    if(realLabel<0.5) TN_compare++;
//                }
            }
            
            double sensitivity =(double)correctPosPredict /PosN;
            double specificity =(double)correctNegPredict /NegN;
            double gMean =sqrt(sensitivity *specificity );
            
//            double sensitivity_compare=(double)correctPosPredict_compare/PosN;
//            double specificity_compare=(double)correctNegPredict_compare/NegN;
//            double gMean_compare=sqrt(sensitivity_compare*specificity_compare);
            
            Sensitivity0.push_back(sensitivity );
            Specificity0.push_back(specificity );
            Gmean0.push_back(gMean );
            
//            compareSensitivity0.push_back(sensitivity_compare);
//            compareSpecificity0.push_back(specificity_compare);
//            compareGmean0.push_back(gMean_compare);
        }
        
//        double during=(double)(clock()-start)/CLOCKS_PER_SEC;
//        cout<<"total time "<<during<<endl;
        Sensitivity.push_back(Sensitivity0);
        Specificity.push_back(Specificity0);
        Gmean.push_back(Gmean0);
        
//        compareSensitivity.push_back(compareSensitivity0);
//        compareSpecificity.push_back(compareSpecificity0);
//        compareGmean.push_back(compareGmean0);
        
    }
}

void TestExampleBayes::PrintPerformance()
{
    cout<<"----Performance of Bayes method----"<<endl;
    cout<<"       Balanced Sampling                            Imbalanced Sampling"<<endl;
    cout<<"       Sensitivity * Specificity * Gmean * "<<endl;//"         Sensitivity * Specificity * Gmean * "<<endl;
    
    vector<double> SensitivityMean;
    vector<double> SensitivityStd;
    vector<double> SpecificityMean;
    vector<double> SpecificityStd;
    vector<double> GmeanMean;
    vector<double> GmeanStd;
    
//    vector<double> compareSensitivityMean;
//    vector<double> compareSensitivityStd;
//    vector<double> compareSpecificityMean;
//    vector<double> compareSpecificityStd;
//    vector<double> compareGmeanMean;
//    vector<double> compareGmeanStd;
    
    GetMeanAndStd(Sensitivity, &SensitivityMean, &SensitivityStd);
    GetMeanAndStd(Specificity, &SpecificityMean, &SpecificityStd);
    GetMeanAndStd(Gmean, &GmeanMean, &GmeanStd);
    
//    GetMeanAndStd(compareSensitivity, &compareSensitivityMean, &compareSensitivityStd);
//    GetMeanAndStd(compareSpecificity, &compareSpecificityMean, &compareSpecificityStd);
//    GetMeanAndStd(compareGmean, &compareGmeanMean, &compareGmeanStd);
    
    for(int i=0;i<SensitivityMean.size();i++)
    {
        cout<< std::setw(4)<< std::fixed<< std::setprecision(2)<< (double)trainIndexEachUpdate.at(i+1)/trainN<<" ";
        cout<< std::setw(6)<< std::fixed<< std::setprecision(4)<< SensitivityMean[i]<<" ";
        cout<< std::setw(6)<< std::fixed<< std::setprecision(4)<< SensitivityStd[i]<<" ";
        cout<< std::setw(6)<< std::fixed<< std::setprecision(4)<< SpecificityMean[i]<<" ";
        cout<< std::setw(6)<< std::fixed<< std::setprecision(4)<< SpecificityStd[i]<<" ";
        cout<< std::setw(6)<< std::fixed<< std::setprecision(4)<< GmeanMean[i]<<" ";
        cout<< std::setw(6)<< std::fixed<< std::setprecision(4)<< GmeanStd[i]<<endl;//"    ";
        
//        cout<< std::setw(6)<< std::fixed<< std::setprecision(4)<< compareSensitivityMean[i]<<" ";
//        cout<< std::setw(6)<< std::fixed<< std::setprecision(4)<< compareSensitivityStd[i]<<" ";
//        cout<< std::setw(6)<< std::fixed<< std::setprecision(4)<< compareSpecificityMean[i]<<" ";
//        cout<< std::setw(6)<< std::fixed<< std::setprecision(4)<< compareSpecificityStd[i]<<" ";
//        cout<< std::setw(6)<< std::fixed<< std::setprecision(4)<< compareGmeanMean[i]<<" ";
//        cout<< std::setw(6)<< std::fixed<< std::setprecision(4)<< compareGmeanStd[i]<<" "<<endl;
    }
}
