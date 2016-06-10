//
//  RFTestExample.cpp
//  DyBaORF_test
//
//  Created by Guotai Wang on 07/12/2015.
//
//

#include "RFTestExample.h"
RFTestExample::RFTestExample()
{
    
}

RFTestExample::~RFTestExample()
{
    
}

void RFTestExample::Run(int MaxIter)
{
#if USE_PROPOSED_ORF
    Sensitivity.clear();
    Specificity.clear();
    Gmean.clear();
    Time.clear();
#endif
    
#if COMPARE_WITH_MPB_ORF
    compareSensitivity.clear();
    compareSpecificity.clear();
    compareGmean.clear();
    compareTime.clear();
#endif
    
#if COMPARE_WITH_SPB_ORF
    compareSensitivity2.clear();
    compareSpecificity2.clear();
    compareGmean2.clear();
    compareTime2.clear();
#endif
    
#if COMAPRE_WITH_OFFLINE
    compareSensitivity3.clear();
    compareSpecificity3.clear();
    compareGmean3.clear();
    compareTime3.clear();
#endif
    
    int treeN=50;
    int depth=20;
    int sampleN=6;
    for(int idx=0;idx<MaxIter;idx++)
    {
        bool singleClass=true;
        while(singleClass)
        {
            GenerateTrainAndTestData();
            PrintDataInformation();
            int posn=0;
            int negn=0;
            for(int i=0;i<trainIndexEachUpdate[1];i++)
            {
                double tempL=trainData->at(i)->back();
                if(tempL==1.0)posn++;
                else negn++;
                if(posn>0 && negn>0)
                {
                    singleClass=false;
                    break;
                }
            }
        }
        vector<double> tempImbalanceRatio=GetImbalanceRatio();
        imbalanceRatio.push_back(tempImbalanceRatio);
        
        cout<<"iteration "<<idx<<", data prepared"<<endl;
#if USE_PROPOSED_ORF
        // proposed method
        RandomForest::ORForest<double> rf;
        rf.Init(treeN, depth,sampleN);// tree number, depth, sample number in node
        rf.SetSamplingType(RandomForest::DownSamplingMajority);
        rf.SetBalanceType(RandomForest::DynamicImbalanceAdaptableBootstrap);
        vector<double>  Sensitivity_on;
        vector<double>  Specificity_on;
        vector<double>  Gmean_on;
        vector<double>  Time_on;
#endif
        
#if COMPARE_WITH_MPB_ORF
        // online RF with mutiple parameter boostrap
        RandomForest::ORForest<double> rf1;
        rf1.Init(treeN, depth,sampleN);// tree number, depth, sample number in node
        rf1.SetSamplingType(RandomForest::DownSamplingMajority);
        rf1.SetBalanceType(RandomForest::MultipleParameterBoostrap);
        vector<double>  Sensitivity_on1;
        vector<double>  Specificity_on1;
        vector<double>  Gmean_on1;
        vector<double>  Time_on1;
#endif
        
#if COMPARE_WITH_SPB_ORF
        // online RF with single parameter boostrap
        RandomForest::ORForest<double> rf2;
        rf2.Init(treeN, depth,sampleN);// tree number, depth, sample number in node
        rf2.SetSamplingType(RandomForest::DownSamplingMajority);
        rf2.SetBalanceType(RandomForest::SingleParameterBoostrap);
        vector<double>  Sensitivity_on2;
        vector<double>  Specificity_on2;
        vector<double>  Gmean_on2;
        vector<double>  Time_on2;
#endif
        
#if COMPARE_WITH_OFFLINE
        vector<double>  Sensitivity_off;
        vector<double>  Specificity_off;
        vector<double>  Gmean_off;
        vector<double>  Time_off;
#endif
        
        for(int it=0;it<trainIndexEachUpdate.size()-1;it++)
        {
            int addTrainStart=trainIndexEachUpdate[it];
            int addtrainEnd=trainIndexEachUpdate[it+1];
            int addTrainN=trainIndexEachUpdate[it+1]-trainIndexEachUpdate[it];
            if(addTrainN==0)break;
            
            ///get online training data
            shared_ptr<vector<shared_ptr<vector<double> > > > tempOnlineTrainData(new vector<shared_ptr<vector<double> > >);
            tempOnlineTrainData->reserve(addTrainN);
            for(int i=addTrainStart;i<addtrainEnd;i++)
            {
                tempOnlineTrainData->push_back(trainData->at(i));
            }
#if USE_PROPOSED_ORF
            time_t startTrain=clock();
            rf.Train(tempOnlineTrainData);
            double duringTrain=(double)(clock()-startTrain)/CLOCKS_PER_SEC;
            Time_on.push_back(duringTrain);
            //int node1=rf.GetActureMaxTreeNode();
            
            vector<float> *predict_on;
            time_t startTest=clock();
            rf.Predict(testData,&predict_on);
            double duringTest=(double)(clock()-startTest)/CLOCKS_PER_SEC;
            
            int correctPosPredict_on=0;
            int correctNegPredict_on=0;
            
            int posPredict_on=0;
            int TP_on=0;
            int negPredict_on=0;
            int TN_on=0;
#endif
            
#if COMPARE_WITH_MPB_ORF
            time_t startTrain1=clock();
            rf1.Train(tempOnlineTrainData);
            double duringTrain1=(double)(clock()-startTrain1)/CLOCKS_PER_SEC;
            Time_on1.push_back(duringTrain1);
            //int node2=rf1.GetActureMaxTreeNode();
            
            vector<float> *predict_on1;
            time_t startTest1=clock();
            rf1.Predict(testData, &predict_on1);
            double duringTest1=(double)(clock()-startTest1)/CLOCKS_PER_SEC;
            
            int correctPosPredict_on1=0;
            int correctNegPredict_on1=0;
            int posPredict_on1=0;
            int TP_on1=0;
            int negPredict_on1=0;
            int TN_on1=0;

#endif
#if COMPARE_WITH_SPB_ORF
            time_t startTrain2=clock();
            rf2.Train(tempOnlineTrainData);
            double duringTrain2=(double)(clock()-startTrain2)/CLOCKS_PER_SEC;
            Time_on2.push_back(duringTrain2);
            //int node2=rf1.GetActureMaxTreeNode();
            
            vector<float> *predict_on2;
            time_t startTest2=clock();
            rf2.Predict(testData, &predict_on2);
            double duringTest2=(double)(clock()-startTest2)/CLOCKS_PER_SEC;
            
            int correctPosPredict_on2=0;
            int correctNegPredict_on2=0;
            
            int posPredict_on2=0;
            int TP_on2=0;
            int negPredict_on2=0;
            int TN_on2=0;

#endif
            
#if COMPARE_WITH_OFFLINE
            vector<float> *predict_off;
            // offline RF
            RandomForest::ORForest<double> offrf;
            offrf.Init(treeN, depth,sampleN);// tree number, depth, sample number in node
            offrf.SetSamplingType(RandomForest::DownSamplingMajority);
            offrf.SetBalanceType(RandomForest::DynamicImbalanceAdaptableBootstrap);
            
            // get offline train data
            shared_ptr<vector<shared_ptr<vector<double> > > > tempOfflineTrainData(new vector<shared_ptr<vector<double> > >);
            tempOfflineTrainData->reserve(addtrainEnd);
            for(int i=0;i<addtrainEnd;i++)
            {
                tempOfflineTrainData->push_back(trainData->at(i));
            }
            
            time_t startTrain3=clock();
            offrf.Train(tempOfflineTrainData);
            double duringTrain3=(double)(clock()-startTrain3)/CLOCKS_PER_SEC;
            
            Time_off.push_back(duringTrain3);
            
            offrf.Predict(testData, &predict_off);
            
            int correctPosPredict_off=0;
            int correctNegPredict_off=0;
            int posPredict_off=0;
            int TP_off=0;
            int negPredict_off=0;
            int TN_off=0;
#endif

            int PosN=0;
            int NegN=0;
            for(int i=0;i<testN;i++)
            {
                double realLabel=testData->at(i)->back();
#if USE_PROPOSED_ORF
                double predictLabel_on =predict_on->at(i);
#endif
#if COMPARE_WITH_MPB_ORF
                double predictLabel_on1 =predict_on1->at(i);
#endif
#if COMPARE_WITH_SPB_ORF
                double predictLabel_on2 =predict_on2->at(i);
#endif
#if COMPARE_WITH_OFFLINE
                double predictLabel_off=predict_off->at(i);
#endif
                
                if(realLabel>=0.5)
                {
                    PosN++;
#if USE_PROPOSED_ORF
                    if(predictLabel_on>=0.5 )correctPosPredict_on++;
#endif
#if COMPARE_WITH_MPB_ORF
                    if(predictLabel_on1>=0.5 )correctPosPredict_on1++;
#endif
#if COMPARE_WITH_SPB_ORF
                    if(predictLabel_on2>=0.5 )correctPosPredict_on2++;
#endif
#if COMPARE_WITH_OFFLINE
                    if(predictLabel_off>=0.5)correctPosPredict_off++;
#endif
                }
                else{
                    NegN++;
#if USE_PROPOSED_ORF
                    if(predictLabel_on<0.5)correctNegPredict_on++;
#endif
#if COMPARE_WITH_MPB_ORF
                    if(predictLabel_on1<0.5)correctNegPredict_on1++;
#endif
#if COMPARE_WITH_SPB_ORF
                    if(predictLabel_on2<0.5)correctNegPredict_on2++;
#endif
#if COMPARE_WITH_OFFLINE
                    if(predictLabel_off<0.5)correctNegPredict_off++;
#endif
                }
#if USE_PROPOSED_ORF
                if(predictLabel_on>=0.5)
                {
                    posPredict_on++;
                    if(realLabel>=0.5) TP_on++;
                }
                else{
                    negPredict_on++;
                    if(realLabel<0.5) TN_on++;
                }
#endif
#if COMPARE_WITH_MPB_ORF
                if(predictLabel_on1>=0.5)
                {
                    posPredict_on1++;
                    if(realLabel>=0.5) TP_on1++;
                }
                else{
                    negPredict_on1++;
                    if(realLabel<0.5) TN_on1++;
                }
#endif
#if COMPARE_WITH_SPB_ORF
                if(predictLabel_on2>=0.5)
                {
                    posPredict_on2++;
                    if(realLabel>=0.5) TP_on2++;
                }
                else{
                    negPredict_on2++;
                    if(realLabel<0.5) TN_on2++;
                }
#endif
#if COMPARE_WITH_OFFLINE
                if(predictLabel_off>=0.5)
                {
                    posPredict_off++;
                    if(realLabel>=0.5) TP_off++;
                }
                else{
                    negPredict_off++;
                    if(realLabel<0.5) TN_off++;
                }
#endif
            }
#if USE_PROPOSED_ORF
            double sensitivity_on=(double)correctPosPredict_on/PosN;
            double specificity_on=(double)correctNegPredict_on/NegN;
            double gMean_on=sqrt(sensitivity_on*specificity_on);
            
            Sensitivity_on.push_back(sensitivity_on);
            Specificity_on.push_back(specificity_on);
            Gmean_on.push_back(gMean_on);
#endif
#if COMPARE_WITH_MPB_ORF
            double sensitivity_on1=(double)correctPosPredict_on1/PosN;
            double specificity_on1=(double)correctNegPredict_on1/NegN;
            double gMean_on1=sqrt(sensitivity_on1*specificity_on1);
            
            Sensitivity_on1.push_back(sensitivity_on1);
            Specificity_on1.push_back(specificity_on1);
            Gmean_on1.push_back(gMean_on1);
#endif
#if COMPARE_WITH_SPB_ORF
            double sensitivity_on2=(double)correctPosPredict_on2/PosN;
            double specificity_on2=(double)correctNegPredict_on2/NegN;
            double gMean_on2=sqrt(sensitivity_on2*specificity_on2);
            
            Sensitivity_on2.push_back(sensitivity_on2);
            Specificity_on2.push_back(specificity_on2);
            Gmean_on2.push_back(gMean_on2);
#endif
#if COMPARE_WITH_OFFLINE
            double sensitivity_off=(double)correctPosPredict_off/PosN;
            double specificity_off=(double)correctNegPredict_off/NegN;
            double gMean_off=sqrt(sensitivity_off*specificity_off);
            
            Sensitivity_off.push_back(sensitivity_off);
            Specificity_off.push_back(specificity_off);
            Gmean_off.push_back(gMean_off);
#endif
        }
#if USE_PROPOSED_ORF
        Sensitivity.push_back(Sensitivity_on);
        Specificity.push_back(Specificity_on);
        Gmean.push_back(Gmean_on);
        Time.push_back(Time_on);
#endif
#if COMPARE_WITH_MPB_ORF
        compareSensitivity.push_back(Sensitivity_on1);
        compareSpecificity.push_back(Specificity_on1);
        compareGmean.push_back(Gmean_on1);
        compareTime.push_back(Time_on1);
#endif
#if COMPARE_WITH_SPB_ORF
        compareSensitivity2.push_back(Sensitivity_on2);
        compareSpecificity2.push_back(Specificity_on2);
        compareGmean2.push_back(Gmean_on2);
        compareTime2.push_back(Time_on2);
#endif
#if COMPARE_WITH_OFFLINE
        compareSensitivity3.push_back(Sensitivity_off);
        compareSpecificity3.push_back(Specificity_off);
        compareGmean3.push_back(Gmean_off);
        compareTime3.push_back(Time_off);
#endif
//        if(idx==MaxIter-1)
//        {
//            shared_ptr<vector<int> > featureIndexList(new vector<int>);
//            shared_ptr<vector<double> > featureImportanceList(new vector<double>);
//            rf.GetRankedGiniImportance(&featureIndexList, &featureImportanceList);
//            cout<<"Feature Importance "<<endl;
//            for(int i=0;i<featureIndexList->size();i++)
//            {
//                cout<<std::setw(6)<<featureIndexList->at(i)<<" "<<featureImportanceList->at(i)<<endl;
//            }
//        }
    }
}

void RFTestExample::PrintPerformance()
{
    vector<double> imbalanceRatioMean;
    vector<double> imbalanceRatioStd;
    GetMeanAndStd(imbalanceRatio, &imbalanceRatioMean, &imbalanceRatioStd);
    for(int i=0;i<trainIndexEachUpdate.size()-1;i++)
    {
        cout<< std::setw(4)<< std::fixed<< std::setprecision(2)<< (double)trainIndexEachUpdate.at(i+1)/trainN<<" ";
        cout<< std::setw(6)<< std::fixed<< std::setprecision(2)<<imbalanceRatioMean[i]<<" ";
        cout<< std::setw(6)<< std::fixed<< std::setprecision(2)<<imbalanceRatioStd[i]<<endl;
    }
    
    cout<<"comparison between  ";
#if USE_PROPOSED_ORF
    cout<<"DIA ORF,  ";
#endif
#if COMPARE_WITH_MPB_ORF
    cout<<"MPB ORF,  ";
#endif
#if COMPARE_WITH_SPB_ORF
    cout<<"SPB ORF,  ";
#endif
#if COMPARE_WITH_OFFLINE
    cout<<"Offline RF";
#endif
    cout<<endl;
    
#if USE_PROPOSED_ORF
    cout<<std::setw(36)<<"Proposed Balanced Online Random Forests"<<"                           ";
#endif
#if COMPARE_WITH_MPB_ORF
    cout<<std::setw(36)<<"Balanced Online Random Forests with MPB"<<"     ";
#endif
    cout<<endl;
    
#if USE_PROPOSED_ORF
    vector<double> SensitivityMean;
    vector<double> SensitivityStd;
    vector<double> SpecificityMean;
    vector<double> SpecificityStd;
    vector<double> GmeanMean;
    vector<double> GmeanStd;
    vector<double> TimeMean;
    vector<double> TimeStd;
    GetMeanAndStd(Sensitivity, &SensitivityMean, &SensitivityStd);
    GetMeanAndStd(Specificity, &SpecificityMean, &SpecificityStd);
    GetMeanAndStd(Gmean, &GmeanMean, &GmeanStd);
    GetMeanAndStd(Time, &TimeMean, &TimeStd);
    cout<<"dia orf, sen, spec, gmean"<<endl;
    for(int i=0;i<Gmean.size();i++)
    {
        cout<<Sensitivity[i].back()<<" "<<Specificity[i].back()<<" "<<Gmean[i].back()<<endl;
    }
#endif
#if COMPARE_WITH_MPB_ORF
    vector<double> compareSensitivityMean;
    vector<double> compareSensitivityStd;
    vector<double> compareSpecificityMean;
    vector<double> compareSpecificityStd;
    vector<double> compareGmeanMean;
    vector<double> compareGmeanStd;
    vector<double> compareTimeMean;
    vector<double> compareTimeStd;
    GetMeanAndStd(compareSensitivity, &compareSensitivityMean, &compareSensitivityStd);
    GetMeanAndStd(compareSpecificity, &compareSpecificityMean, &compareSpecificityStd);
    GetMeanAndStd(compareGmean, &compareGmeanMean, &compareGmeanStd);
    GetMeanAndStd(compareTime, &compareTimeMean, &compareTimeStd);
    cout<<"mpb orf, sen, spec, gmean"<<endl;
    for(int i=0;i<compareGmean.size();i++)
    {
        cout<<compareSensitivity[i].back()<<" "<<compareSpecificity[i].back()<<" "<<compareGmean[i].back()<<endl;
    }
#endif
#if COMPARE_WITH_SPB_ORF
    vector<double> compareSensitivity2Mean;
    vector<double> compareSensitivity2Std;
    vector<double> compareSpecificity2Mean;
    vector<double> compareSpecificity2Std;
    vector<double> compareGmean2Mean;
    vector<double> compareGmean2Std;
    vector<double> compareTime2Mean;
    vector<double> compareTime2Std;
    GetMeanAndStd(compareSensitivity2, &compareSensitivity2Mean, &compareSensitivity2Std);
    GetMeanAndStd(compareSpecificity2, &compareSpecificity2Mean, &compareSpecificity2Std);
    GetMeanAndStd(compareGmean2, &compareGmean2Mean, &compareGmean2Std);
    GetMeanAndStd(compareTime2, &compareTime2Mean, &compareTime2Std);
    cout<<"spb orf, sen, spec, gmean"<<endl;
    for(int i=0;i<compareGmean2.size();i++)
    {
        cout<<compareSensitivity2[i].back()<<" "<<compareSpecificity2[i].back()<<" "<<compareGmean2[i].back()<<endl;
    }
#endif
#if COMPARE_WITH_OFFLINE
    vector<double> compareSensitivity3Mean;
    vector<double> compareSensitivity3Std;
    vector<double> compareSpecificity3Mean;
    vector<double> compareSpecificity3Std;
    vector<double> compareGmean3Mean;
    vector<double> compareGmean3Std;
    vector<double> compareTime3Mean;
    vector<double> compareTime3Std;
    GetMeanAndStd(compareSensitivity3, &compareSensitivity3Mean, &compareSensitivity3Std);
    GetMeanAndStd(compareSpecificity3, &compareSpecificity3Mean, &compareSpecificity3Std);
    GetMeanAndStd(compareGmean3, &compareGmean3Mean, &compareGmean3Std);
    GetMeanAndStd(compareTime3, &compareTime3Mean, &compareTime3Std);
    cout<<"offline rf, sen, spec, gmean"<<endl;
    for(int i=0;i<compareGmean3.size();i++)
    {
        cout<<compareSensitivity3[i].back()<<" "<<compareSpecificity3[i].back()<<" "<<compareGmean3[i].back()<<endl;
    }
#endif

    for(int i=0;i<trainIndexEachUpdate.size()-1;i++)
    {
        cout<< std::setw(4)<< std::fixed<< std::setprecision(2)<< (double)trainIndexEachUpdate.at(i+1)/trainN<<" ";
#if USE_PROPOSED_ORF
        cout<< std::setw(6)<< std::fixed<< std::setprecision(4)<< SensitivityMean[i]<<" ";
        cout<< std::setw(6)<< std::fixed<< std::setprecision(4)<< SensitivityStd[i]<<" ";
        cout<< std::setw(6)<< std::fixed<< std::setprecision(4)<< SpecificityMean[i]<<" ";
        cout<< std::setw(6)<< std::fixed<< std::setprecision(4)<< SpecificityStd[i]<<" ";
        cout<< std::setw(6)<< std::fixed<< std::setprecision(4)<< GmeanMean[i]<<" ";
        cout<< std::setw(6)<< std::fixed<< std::setprecision(4)<< GmeanStd[i]<<"  ";
        cout<< std::setw(6)<< std::fixed<< std::setprecision(4)<< TimeMean[i]<<" ";
        cout<< std::setw(6)<< std::fixed<< std::setprecision(4)<< TimeStd[i]<<"     ";
#endif
#if COMPARE_WITH_MPB_ORF
        cout<< std::setw(6)<< std::fixed<< std::setprecision(4)<< compareSensitivityMean[i]<<" ";
        cout<< std::setw(6)<< std::fixed<< std::setprecision(4)<< compareSensitivityStd[i]<<" ";
        cout<< std::setw(6)<< std::fixed<< std::setprecision(4)<< compareSpecificityMean[i]<<" ";
        cout<< std::setw(6)<< std::fixed<< std::setprecision(4)<< compareSpecificityStd[i]<<" ";
        cout<< std::setw(6)<< std::fixed<< std::setprecision(4)<< compareGmeanMean[i]<<" ";
        cout<< std::setw(6)<< std::fixed<< std::setprecision(4)<< compareGmeanStd[i]<<"  ";
        cout<< std::setw(6)<< std::fixed<< std::setprecision(4)<< compareTimeMean[i]<<" ";
        cout<< std::setw(6)<< std::fixed<< std::setprecision(4)<< compareTimeStd[i]<<"     ";
#endif
        cout<<endl;
    }

    cout<<endl;
#if COMPARE_WITH_SPB_ORF
    cout<<std::setw(36)<<"Balanced Online Random Forests with SPB"<<"                      ";
#endif
#if COMPARE_WITH_OFFLINE
    cout<<std::setw(36)<<"Balanced Offline Random Forests"<<"     ";
#endif
    cout<<endl;
    for(int i=0;i<trainIndexEachUpdate.size()-1;i++)
    {
#if COMPARE_WITH_SPB_ORF | COMPARE_WITH_OFFLINE
        cout<< std::setw(4)<< std::fixed<< std::setprecision(2)<< (double)trainIndexEachUpdate.at(i+1)/trainN<<" ";
#endif
#if COMPARE_WITH_SPB_ORF
        cout<< std::setw(6)<< std::fixed<< std::setprecision(4)<< compareSensitivity2Mean[i]<<" ";
        cout<< std::setw(6)<< std::fixed<< std::setprecision(4)<< compareSensitivity2Std[i]<<" ";
        cout<< std::setw(6)<< std::fixed<< std::setprecision(4)<< compareSpecificity2Mean[i]<<" ";
        cout<< std::setw(6)<< std::fixed<< std::setprecision(4)<< compareSpecificity2Std[i]<<" ";
        cout<< std::setw(6)<< std::fixed<< std::setprecision(4)<< compareGmean2Mean[i]<<" ";
        cout<< std::setw(6)<< std::fixed<< std::setprecision(4)<< compareGmean2Std[i]<<"  ";
        cout<< std::setw(6)<< std::fixed<< std::setprecision(4)<< compareTime2Mean[i]<<" ";
        cout<< std::setw(6)<< std::fixed<< std::setprecision(4)<< compareTime2Std[i]<<"     ";
#endif
#if COMPARE_WITH_OFFLINE
        cout<< std::setw(6)<< std::fixed<< std::setprecision(4)<< compareSensitivity3Mean[i]<<" ";
        cout<< std::setw(6)<< std::fixed<< std::setprecision(4)<< compareSensitivity3Std[i]<<" ";
        cout<< std::setw(6)<< std::fixed<< std::setprecision(4)<< compareSpecificity3Mean[i]<<" ";
        cout<< std::setw(6)<< std::fixed<< std::setprecision(4)<< compareSpecificity3Std[i]<<" ";
        cout<< std::setw(6)<< std::fixed<< std::setprecision(4)<< compareGmean3Mean[i]<<" ";
        cout<< std::setw(6)<< std::fixed<< std::setprecision(4)<< compareGmean3Std[i]<<"  ";
        cout<< std::setw(6)<< std::fixed<< std::setprecision(4)<< compareTime3Mean[i]<<" ";
        cout<< std::setw(6)<< std::fixed<< std::setprecision(4)<< compareTime3Std[i]<<"     ";
#endif
        cout<<endl;
    }

}
