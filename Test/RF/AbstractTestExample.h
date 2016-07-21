/*=========================================================================
 Program:   DyBa ORF
 Module:    AbstractTestExample.h
 
 Created by Guotai Wang on 01/12/2015.
 Copyright (c) 2014-2016 University College London, United Kingdom. All rights reserved.
 http://cmictig.cs.ucl.ac.uk
 
 Reference:
 Dynamically Balanced Online Random Forests for Interactive Scribble-based Segmentation.
 Presented at: MICCAI 2016
 Guotai Wang, Maria A. Zuluaga, Rosalind Pratt, Michael Aertsen, Tom Doel,
 Maria Klusmann, Anna L. David, Jan Deprest, Tom Vercauteren, and Sebastien Ourselin.
 
 Distributed under the BSD-3 licence. Please see the file licence.txt
 =========================================================================*/
#pragma once

#include "ORForest.h"
#include <iomanip>

enum DataSetName{BIODEG, MUSK, CTG, WINE};

enum FeatureTypeName{GAUSSIAN, BENOULLI, MULTINOMIAL};

/** \brief class AbstractTestExample
 *
 * An abstract class used to test
 */
class AbstractTestExample{
public:
    /** Construction function */
    AbstractTestExample();
    
    /** Deconstruction function */
    ~AbstractTestExample();
    
    /** Load data from file 
     * @param[in] data the name of data set that will be loaded
     * @return a bool value to indicate whethet the data is loades successfully or not
     */
    bool LoadData(DataSetName data);
    
    /** Set train data chunk
     * @param[in] startPercent the percentage of data used as initial training data
     * @param[in] endPercent the percentage of data used as the finial training data
     * @param[in] increasepercent the percentage of data added for each update
     */
    void SetTrainDataChunk(double startPercent, double endPercent, double increasePercent);
    
    /** Run the training and testing
     * @param[in] maxIter the iteraction for running
     */
    virtual void Run(int MaxIter);
    
    /** Print the performance */
    virtual void PrintPerformance();
    
    /** Print the data information */
    void PrintDataInformation();

protected:
    /** Load CTG dataset */
    bool LoadCTGDataSet();
    /** Load wine dataset */
    bool LoadWineDataSet();
    /** Load Musk dataset */
    bool LoadMuskDataSet();
    /** Load biodeg Dataset */
    bool LoadBiodegDataSet();
    /** Update data information */
    void UpdateDataInfo();
    
    /** Generate train and test data */
    void GenerateTrainAndTestData();
    /** Set the newly arrive data 
     * @param[in] startIndex the start index of the new training data
     * @param[in] endIndex the end index of the new training data
     */
    void SetTrainDataOnline(int startIndex, int endIndex);
    
    /** Get imbalance ratio at each update
     * \return a return vector storing the imbalance ratio at each time training data arrive.
     */
    std::vector<double> GetImbalanceRatio();
    
    /** Get he number of training data */
    int GetTrainN() const;
    
    /** Get meavlue and standard deviation of a list of numbers
     * @param[in] i_array the input array of data
     * @param[out] o_mean the mean value
     * @param[out] o_std the standard deviation
     */
    void GetMeanAndStd(std::vector<std::vector<double> > i_array, std::vector<double> * o_mean, std::vector<double> * o_std);

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
    std::shared_ptr<std::vector<std::shared_ptr<std::vector<double> > > > originData;
    std::shared_ptr<std::vector<std::shared_ptr<std::vector<double> > > > trainData;
    std::shared_ptr<std::vector<std::shared_ptr<std::vector<double> > > > testData;
    std::vector<int> trainIndexEachUpdate;
    
    std::vector<std::vector<double> > imbalanceRatio;
    
    std::vector<std::vector<double> > Sensitivity;
    std::vector<std::vector<double> > Specificity;
    std::vector<std::vector<double> > Gmean;
    std::vector<std::vector<double> > Time;
    
    std::vector<std::vector<double> > compareSensitivity;
    std::vector<std::vector<double> > compareSpecificity;
    std::vector<std::vector<double> > compareGmean;
    std::vector<std::vector<double> > compareTime;
    std::vector<FeatureTypeName> featureTypeList;
};

