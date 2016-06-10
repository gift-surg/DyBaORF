//
//  TestExampleSVM.cpp
//  ORF_test
//
//  Created by Guotai Wang on 23/12/2015.
//
//

#include "TestExampleSVM.h"
#include <fstream>
TestExampleSVM::TestExampleSVM()
{
    
}

TestExampleSVM::~TestExampleSVM()
{
    
}

void TestExampleSVM::NormalizeData()
{
    vector<double> min;
    vector<double> max;
    int fNum=originData->at(0)->size()-1;
    min.resize(fNum);
    max.resize(fNum);
    for(int i=0;i<fNum;i++)
    {
        min[i]=1e6;
        max[i]=-1e6;
    }
    
    for(int i=0;i<originData->size();i++)
    {
        shared_ptr<vector<double> > sample=originData->at(i);
        for(int j=0;j<fNum;j++)
        {
            if(sample->at(j)> max[j]) max[j]=sample->at(j);
            if(sample->at(j)< min[j]) min[j]=sample->at(j);
        }
    }
    
    vector<double> range;
    range.resize(fNum);
    for(int i=0;i<fNum;i++)
    {
        range[i]=max[i]-min[i];
    }
    
    for(int i=0;i<originData->size();i++)
    {
        shared_ptr<vector<double> > sample=originData->at(i);
        for(int j=0;j<fNum;j++)
        {
            double v=sample->at(j);
            double v1=(v-min[j])/range[j];
            sample->at(j)=v1;
        }
    }
}

void TestExampleSVM::SaveData(string name, int n)
{
    ofstream of;
    of.open(name);
//    of.open("/Users/guotaiwang/Documents/MATLAB/onlineRF/yeast.txt");
    for(int i=0;i<n;i++)
    {
        for(int j=0; j<originData->at(i)->size();j++)
        {
            of<<originData->at(i)->at(j)<<" ";
        }
        of<<endl;
    }
    of.close();
}

void TestExampleSVM::Run(int MaxIter)
{
    Sensitivity.clear();
    Specificity.clear();
    Gmean.clear();
    Time.clear();
    
    compareSensitivity.clear();
    compareSpecificity.clear();
    compareGmean.clear();
    compareTime.clear();
    NormalizeData();
    //SaveData("/Users/guotaiwang/Documents/MATLAB/onlineRF/ctg.txt",instanceN);
    
    for(int idx=0;idx<MaxIter;idx++)
    {
  
        cout<<"iteration "<<idx<<", data prepared"<<endl;

        LASVM<double> svm0;
        svm0.SampleSelectMethod(TRICK_59);
//        
//        LASVM<double> svm1;
//        svm1.SampleSelectMethod(RANDOM);
        //svm.UseBalancedSampling(true);
        //LasvmEnsemble<double> svms;
        //svms.Init(1, TOP_SUBSET);
        
        
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
//        for(int it=0; it<1; it++)
        {
                        int addTrainStart=trainIndexEachUpdate[it];
            int addtrainEnd=trainIndexEachUpdate[it+1];
            int addTrainN=trainIndexEachUpdate[it+1]-trainIndexEachUpdate[it];
            
            vector<float> *predict;
            //vector<float> *predict_compare;
            
            //get online training data
            shared_ptr<vector<shared_ptr<vector<double> > > > tempOnlineTrainData(new vector<shared_ptr<vector<double> > >);
            tempOnlineTrainData->reserve(addTrainN);
            for(int i=addTrainStart;i<addtrainEnd;i++)
            {
                tempOnlineTrainData->push_back(trainData->at(i));
            }
            
            time_t start0=clock();
//            svm0.Train(trainData);
            svm0.Train(tempOnlineTrainData);
            
            
            double trainTime0=(double)(clock()-start0)/CLOCKS_PER_SEC;
            Time0.push_back(trainTime0);
            time_t start1=clock();
            svm0.Predict(testData,&predict);
            double testTime0=(double)(clock()-start1)/CLOCKS_PER_SEC;
            
            
//            time_t start2=clock();
//            svm1.Train(trainData);
            //svm1.Train(tempOnlineTrainData);
//            double trainTime1=(double)(clock()-start2)/CLOCKS_PER_SEC;
//            
//            time_t start3=clock();
            //svm1.Predict(testData, &predict_compare);
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
                double predictLabel=predict->at(i);
                //double predictLabel_compare=predict_compare->at(i);
                
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
            double correctRate=((double)correctPosPredict+correctNegPredict)/(PosN+NegN);
            cout<<"correct rate = "<<correctRate<<endl;
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
        
        //svm.Print();
        
    }
}

void TestExampleSVM::PrintPerformance()
{
    cout<<"----Performance of svm method----"<<endl;

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
        cout<< std::setw(4)<< std::fixed<< std::setprecision(2)<< (double)trainIndexEachUpdate.at(i+1)/trainN<<"   ";
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
