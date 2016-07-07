/*=========================================================================
Program:   DyBa ORF
Module:    Node.h

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

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <vector>
#include <iostream>
#include "ODTree.h"

namespace RandomForest {

/** \brief class Node
 *
 *  There are two kinds of nodes: split node and leaf node
 *  For each split node, the best split feature and value are found
 *  based on Gini impurity. For each leaf node, the histogram of samples
 *  is calculated for inference.
 */
template<typename T>
class Node
{
public:
    /** Construction function
     * @param[in] parentTree The parent tree of this node.
     */
    Node(ODTree<T> * parentTree);
    /** Deconstruction function */
    ~Node();

    /** Calculate the value range of one feature
     * @param[in] fIndex the feature index
     * @param[out] min minimum value of feature \p fIndex
     * @param[out] max maximum value of feature \p fIndx
     */
    void GetFeatureRange(int fIndex,T * min,T * max) const;

    /** Split the data set based on one feature threshold
     * @param[in] i_indexList the index of all the data that is splitted
     * @param[in] feature the feature index that is used
     * @param[in] featureValue the threshold that is used
     * @param[out] o_indexList0 the first output sample index, with feature value less than featureValue
     * @param[out] o_indexList1 the second output sample index, with feature value higher than featureValue
     */
    void binSplitDataSet(const std::shared_ptr<std::vector<int> > i_indexList,
                         int feature,
                         T featureValue,
                         std::shared_ptr<std::vector<int> > o_indexList0,
                         std::shared_ptr<std::vector<int> > o_indexList1);

    /** \brief Find the best split
     *
     *  Return the corresponding feature index, the feature value and the decreased inpurity of the best split
     * @param[out] o_bestFeatureIndex the return feature index
     * @param[out] o_bestFeatureValue the return feature value
     * @param[out] o_decreadedImpurity the return decreased impurity
     */
    void chooseBestSplit(int * o_bestFeatureIndex,
                         T * o_bestFeatureValue,
                         double * o_decreadedImpurity);

    /** Calculate the mean label value of a leaf node */
    double meanLeaf() const;
     *  @param[in] i_sampleIndexList the sample index list of the leaf node
     */

    /** Calculate the impurity of a leaf
    double impurityLeaf(const std::shared_ptr<std::vector<int> > i_sampleIndexList) const;

    /** Create a new tree from scratch. Generate the left and right child recursively */
    void CreateTree();

    /** Update a tree with an add set.
     *  Grow the tree with newly arrived training data
     *  @param[in] i_addSampleList the set of data  that is added to the tree.
     */
    void UpdateTree(const std::shared_ptr<std::vector<int> > i_addSampleList);

    /** Update a tree with an add set and a remove set
     *  @param[in] i_rmvSampleList the set of data that would be removed from the tree.
     *  @param[in] i_addSampleList the set of data that would be added to the tree.
     */
    int UpdateTree(const std::shared_ptr<std::vector<int> > i_rmvSampleList,
                   const std::shared_ptr<std::vector<int> > i_addSampleList);

    /** Get the positive and negative sample list of a node
     *  @param[out] o_posSampleList the returned positive sample list
     *  @param[out] o_negSampleList the returned negative sample list
     */
    void GetSampleList(std::shared_ptr<std::vector<int> > o_posSampleList,
                       std::shared_ptr<std::vector<int> > o_negSampleList);

    /** Set the left child of this node
     *  @param[in] l the left child
     */
    void SetLeft(std::shared_ptr<Node<T> > l);

    /** Get the left child of this node */
    std::shared_ptr<Node<T> > GetLeft() const;

    /** Set the right child of this node
     * @param[in] r the right child
     */
    void SetRight(std::shared_ptr<Node<T> >r );

    /** Get the right child of this node */
    std::shared_ptr<Node<T> > GetRight() const;

    /** Set the feature index used for split
     * @param[in] idx the input feature index
     */
    void SetFeatureIndex(int idx);

    /** Get the feature index */
    int GetFeatureIndex() const;

    /** Set the feature value used for split
     * @param[in] v the input feature value
     */
    void SetSplitValue(double v);

    /** Get the split value */
    double GetSplitValue() const;

    /** Set the depth of this node
     * @param[in] d the input depth
     */
    void SetDepth(int d);

    /** Get the depth of this node */
    int GetDepth() const;

    /** Set the sample index list of this node
     * @param[in] list the input sample index list
     */
    void SetSampleIndexList(std::shared_ptr<std::vector<int> > list);

    /** Get the sample index list of this node */
    
    std::shared_ptr<std::vector<int> > GetSampleIndexList() const;
    /** Get the parent tree of this node */
    ODTree<T> * GetTree() const;

    /** Predict the probability of one sample
     * @param[in] i_inData the input sample data
     */
    double PredictOneSample(const std::shared_ptr<std::vector<T> > i_inData);

    /** Update the gini importance of all the fatures */
    void UpdateGiniImportance();


private:
    std::shared_ptr<Node<T> > left;
    std::shared_ptr<Node<T> > right;

    int featureIndex;
    double splitValue;
    double decreasedImpurity;
    int depth;
    std::shared_ptr<std::vector<int> > sampleIndexList;

    // to avoid circular pointer, keep raw pointer here
    ODTree<T> *  tree;
};

}

