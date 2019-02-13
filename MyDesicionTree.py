#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  5 13:36:13 2019

@author: pranavkanse
"""

import pandas as pd
import math
import operator
import datetime
import random

class MyDecisionTree:
    def _init_(self):
        self.attr_val='root'
        self.attr_name=''
        self.gainVal=''
        self.childNodesList = []
        self.leafNode = False
    
## Function to find Entropy OR Variance Impurity ##
def findEntropyVariance(dataset, heuristicType, distinctOutputCnt):
    temp_dataset = dataset.groupby(dataset.iloc[:,-1]).count().iloc[:,-1] 
    entropyVariance = 0
    if heuristicType == "Information Gain":
        for i,val in enumerate(temp_dataset):
            entropyVariance = entropyVariance - (val/sum(temp_dataset))*math.log((val/sum(temp_dataset)),2)
    elif heuristicType == "Variance Impurity":
        entropyVariance = 1
        loop_cnt = 0
        for i,val in enumerate(temp_dataset):
            loop_cnt = loop_cnt + 1
            entropyVariance = entropyVariance * (val/sum(temp_dataset))
        
        if loop_cnt != distinctOutputCnt:
            entropyVariance = 0
            
    return entropyVariance,temp_dataset.index[0]

## Function to find Entropy OR Variance Impurity for each attribute ##
def findEachIndividualEntropy(dataset,col_attr,heuristicType,distinctOutputCnt):
      temp_dataset = dataset.groupby(dataset.columns[col_attr])
      individual_entropy = 0
      total_cnt = sum(temp_dataset.count().iloc[:,-1])
      for attr_name,attr_data in temp_dataset:          
          cnt = (attr_data.count().iloc[-1])
          entropyVariance = findEntropyVariance(attr_data, heuristicType, distinctOutputCnt)
          individual_entropy += (cnt/total_cnt)*(entropyVariance[0])
      return individual_entropy 

## Function to find the next best attribute ##
def findNextBestAttribute(dataset,entropy,resultColumnName,heuristicType,distinctOutputCnt):
    gain_dict = {}
    for col_indx,column_name in enumerate(dataset):
        if column_name != resultColumnName:
            entropy_sum = findEachIndividualEntropy(dataset,col_indx,heuristicType,distinctOutputCnt)
            gain = entropy - entropy_sum
            gain_dict[column_name]=gain
    
    maxGainKey = max(gain_dict.items(), key=operator.itemgetter(1))[0]
    return maxGainKey, gain_dict[maxGainKey]

## Function to Create the Decision Tree ##
def createCompleteTree(training_data,node,heuristicType,distinctOutputCnt):
    for col_data,table_data in training_data:
        newData = table_data.drop(node.attr_name, axis=1)
        entropy = findEntropyVariance(newData,heuristicType, distinctOutputCnt)
        newNode = MyDecisionTree()
        newNode.attr_name, newNode.gainVal = findNextBestAttribute(newData,entropy[0],newData.columns.values[-1],heuristicType,distinctOutputCnt)
        newNode.childNodesList = []
        newNode.attr_val = col_data
        
        if newNode.gainVal == 0:
            newNode.attr_name = entropy[1]
        else:
            newNode = createCompleteTree(newData.groupby(newNode.attr_name),newNode,heuristicType,distinctOutputCnt)[1]
        
        node.childNodesList.append(newNode)
    return newData, node

## Function to Print the Decision Tree ##
def printTree(string,treeNode,nodeCnt):
    for val in treeNode.childNodesList:
        print(string+treeNode.attr_name+' = '+str(val.attr_val)+' : ',end=''),
        if not val.childNodesList:
            print(str(val.attr_name))
        else:
            print()
            nodeCnt = nodeCnt + 1
            nodeCnt = printTree(string+' ',val,nodeCnt)            
    return nodeCnt

## Function to find output Value from the Desicion Tree ##
def checkVal(row,node):   
    val=row[node.attr_name]
    if not node.childNodesList:
        return node
    else:
        for nd in node.childNodesList:
            if val == nd.attr_val:
                cNode = checkVal(row,nd)
                return cNode

## Function to find Accuracy of the Tree ##
def validate_data(validation_data,treeNode):
    correctCnt = 0
    for index,row in validation_data.iterrows():
        if row[-1] == checkVal(row,treeNode).attr_name:
            correctCnt +=1
    accuracy = 100 * (correctCnt/len(validation_data))
    return accuracy

## Function to Prune the Decision Tree ##
def getPrunedTree(bestTreeNode,cnt,intP,training_data_pruning):
    for nd in bestTreeNode.childNodesList:
        for newNode in nd.childNodesList:
            cnt = cnt + 1
            if cnt == intP:
                nd.childNodesList = []
                refreshedData = training_data_pruning.groupby(bestTreeNode.attr_name)
                for i,i_data in refreshedData:
                    if nd.attr_val == i:
                        tempData = i_data.groupby(i_data.iloc[:,-1]).count().iloc[:,-1]
                        nd.attr_name = tempData[tempData == max(tempData)].index[0]
                        break
            else:
                if cnt < intP:
                    nd = getPrunedTree(nd,cnt,intP,training_data_pruning)
    return bestTreeNode

training_path = "/Users/pranavkanse/Desktop/Pranav Stuff/Elearning Courses/CS 6375 Machine Learning/ASSIGNMENTS/A 1/data_sets1/training_set.csv"
validation_path = "/Users/pranavkanse/Desktop/Pranav Stuff/Elearning Courses/CS 6375 Machine Learning/ASSIGNMENTS/A 1/data_sets1/validation_set.csv"
intL = 10
intK = 15

orig_training_data = pd.read_csv(training_path)
training_data = orig_training_data
training_data_pruning = training_data
distinctOutputCount = len(training_data.groupby(training_data.columns[-1]))

##################################### INFORMATION GAIN HEURISTIC #####################################
heuristicType = "Information Gain"
print("\nLearning Started at: ",datetime.datetime.now())
entropy = findEntropyVariance(training_data,heuristicType,distinctOutputCount)
parent_child_con = {}

myDecisionTreeNode = MyDecisionTree()
myDecisionTreeNode.attr_name, myDecisionTreeNode.gainVal = findNextBestAttribute(training_data,entropy[0],training_data.columns.values[-1],heuristicType,distinctOutputCount)
myDecisionTreeNode.childNodesList = []
myDecisionTreeNode.attr_val = 'root'
temp_dataset = training_data.groupby(myDecisionTreeNode.attr_name)
updatedData,myDecisionTreeNode=createCompleteTree(temp_dataset,myDecisionTreeNode,heuristicType,distinctOutputCount)
print("Learning Finished at: ",datetime.datetime.now())

tempNodeCnt = 1
print("\nTree Printing Started at: ",datetime.datetime.now())
tempNodeCnt = printTree('',myDecisionTreeNode,tempNodeCnt)
print("\nTree Printing Finished at: ",datetime.datetime.now())

validation_data = pd.read_csv(validation_path)
print("\nValidation Started at: ",datetime.datetime.now())
accuracy = validate_data(validation_data,myDecisionTreeNode)
print("Accuracy := ",accuracy)
print("Validation Finished at: ",datetime.datetime.now())

bestTreeNode = myDecisionTreeNode
prunedAccuracy = 0
finalAccuracy = accuracy
treePruned = False
print("\nPruning Process Started at: ",datetime.datetime.now())
for i in range(intL):
    intM = random.randint(1,intK)
    for j in range(intM):        
        intP=random.randint(2,tempNodeCnt)
        cnt = 1
        prunedTreeRoot=getPrunedTree(bestTreeNode,cnt,intP,training_data_pruning)
        prunedAccuracy=validate_data(validation_data,prunedTreeRoot)
        if prunedAccuracy>accuracy:
            bestTreeNode = prunedTreeRoot
            treePruned = True
            finalAccuracy = prunedAccuracy
            print("New Accuracy := ",prunedAccuracy)
            print('\n Printing Current Pruned Tree at: ',datetime.datetime.now())
            tempNodeCnt=1
            tempNodeCnt=printTree('',myDecisionTreeNode,tempNodeCnt) 
print("\nPruning Finished at: ",datetime.datetime.now())
if treePruned:
    print("\n Tree pruned. \nFinal Accuracy: ",finalAccuracy)
else:
    print("Tree was not Pruned.")
            

##################################### VARIANCE IMPURITY HEURISTIC #####################################
heuristicType = "Variance Impurity"
print("\nLearning Started at: ",datetime.datetime.now())
entropy = findEntropyVariance(training_data,heuristicType,distinctOutputCount)
parent_child_con = {}

myDecisionTreeNode = MyDecisionTree()
myDecisionTreeNode.attr_name, myDecisionTreeNode.gainVal = findNextBestAttribute(training_data,entropy[0],training_data.columns.values[-1],heuristicType,distinctOutputCount)
myDecisionTreeNode.childNodesList = []
myDecisionTreeNode.attr_val = 'root'
temp_dataset = training_data.groupby(myDecisionTreeNode.attr_name)
updatedData,myDecisionTreeNode=createCompleteTree(temp_dataset,myDecisionTreeNode,heuristicType,distinctOutputCount)
print("Learning Finished at: ",datetime.datetime.now())

tempNodeCnt = 1
print("\nTree Printing Started at: ",datetime.datetime.now())
tempNodeCnt = printTree('',myDecisionTreeNode,tempNodeCnt)
print("\nTree Printing Finished at: ",datetime.datetime.now())

validation_data = pd.read_csv(validation_path)
print("\nValidation Started at: ",datetime.datetime.now())
accuracy = validate_data(validation_data,myDecisionTreeNode)
print("Accuracy := ",accuracy)
print("Validation Finished at: ",datetime.datetime.now())

bestTreeNode = myDecisionTreeNode
prunedAccuracy = 0
finalAccuracy = accuracy
treePruned = False
print("\nPruning Process Started at: ",datetime.datetime.now())
for i in range(intL):
    intM = random.randint(1,intK)
    for j in range(intM):        
        intP=random.randint(2,tempNodeCnt)
        cnt = 1
        prunedTreeRoot=getPrunedTree(bestTreeNode,cnt,intP,training_data_pruning)
        prunedAccuracy=validate_data(validation_data,prunedTreeRoot)
        if prunedAccuracy>accuracy:
            bestTreeNode = prunedTreeRoot
            treePruned = True
            finalAccuracy = prunedAccuracy
            print("New Accuracy := ",prunedAccuracy)
            print('\n Printing Current Pruned Tree at: ',datetime.datetime.now())
            tempNodeCnt=1
            tempNodeCnt=printTree('',myDecisionTreeNode,tempNodeCnt) 
print("\nPruning Finished at: ",datetime.datetime.now())
if treePruned:
    print("\n Tree pruned. \nFinal Accuracy: ",finalAccuracy)
else:
    print("Tree was not Pruned.")

