# required imports
from collections import Counter as Cn   
import argparse 
import pickle
import scipy
import copy
import math
import pandas as pd
from sklearn.metrics import accuracy_score

# calculating entropy 
def calculate_entropy(X,Y):  
        unqvals = X.unique()  # getting unique feature values 
        X_cnt = len(X)        # getting their count
        fin_entr = 0     
        
                val_prob = val_rows / X_cnt     # calculating probablity of the 
                Y_valcnt = Cn(val_Y)            # case where the feature has the particular feature value 'val'
                negative_rate = Y_valcnt[0] / val_rows   # getting probability of negative samples in current partition
                positive_rate = Y_valcnt[1] / val_rows   # getting probability of positive samples in current partition 
                entr = ((positive_rate * math.log(positive_rate,2)) + (negative_rate * math.log(negative_rate,2))) * -1  # calculating entropy with known formula
            fin_entr += entr * val_prob 
        return fin_entr  # returning total entropy for a feature/ attribute

# calculating chisqure p-value
def calculate_chi2(X,Y):   
        exp = []    # expected values
        act = []    # actual values
        unqvals = X.unique()  # getting unique values for a feature
        Y_vals = Cn(Y)    # using collections.counter to get the count for each value in target label in current transition
        p = Y_vals[1]  # positive examples in partition
        n = Y_vals[0]  # negative examples in partition
        N = p+n   # total examples
        
        return scipy.stats.chisquare(act,exp)[1]   # using scipy chisqure library for calculation

# base class for node in Tree
class Node:   
    def __init__(self,val=True, child=[-1]*82) -> None:
        self.val = val                   # value in each node of tree
        self.childnodes = list(child)   #  children nodes values

# base class for our ID3 Decision Tree
class ID3Tree:
    def __init__(self):
        self.root = None
        self.leaves = 0    # count for leaf nodes in Tree
        self.nonleaves = 0  # count for internal nodes in Tree
    
    # returning common occurence as value for a leaf node
    
    
    # fitting our ID3 Decision Tree model
    def fit(self,X,Y,remattrs,thresh,parent,childv): 
        if len(set(Y)) ==1:   #   if there is only either 1 or 0 in target label
            self.leaves += 1     
            if list(Y)[0] == 1:
                node = Node(val=True)   # returning leaf node as we arrived at homogenous rows branch
            else:
                node = Node(val=False)
          # calculating entropy for each feature
                if minentropy > entr: 
                    minattr = feat     # checking minimum entropy
                    minentropy = entr
            remattrs.remove(minattr)   # removing best split attribute from remaining features
            chi2_pval = calculate_chi2(X[minattr],tempY)   # calculating chisqure p value
            if chi2_pval >= thresh:  # testing if our split is significant using threshold provided as p value parameter
                node = self.get_maxcountl(Y)   # if not significant returning max occurence value as leaf node
              # recursively calling fit to populate nodes below the selected attribute
        if parent is None:   # root has no parent
            self.root = node
        else:     # adding child nodes to parent
            parent.childnodes[childv] = node
        return self 

    # prediction method for a single test data point 
    def pred(self,testd,start=None):   
        # print(start)
        if start is None:   # setting root for starting prediction
            start = self.root
           # recursively call prediction method after going to the appropriate branch
                else:
                    return 0   # return 0 as prediction in any other cases
            else:
                return 0

# argument parsing 
par = argparse.ArgumentParser()   
par.add_argument('-f1', dest='traindata', action='store', type=str, help='traindata')
par.add_argument('-p', dest='pval_thresh', action='store', type=float, help='pval_thresh')
par.add_argument('-o', dest='out_file', action='store', type=str, help='out_file')
par.add_argument('-f2', dest='testdata', action='store', type=str, help='testdata')
par.add_argument('-t', dest='tree', action='store', type=str, help='tree')
args = par.parse_args()

#storing argument values
train = args.traindata
test = args.testdata
thresh = args.pval_thresh
outf = args.out_file

# creating dataframes 
X_train = pd.read_csv(train, header=None, sep=' ')
Y_train = pd.read_csv(trainlabel, header=None, sep=' ')[0]
X_test = pd.read_csv(test, header=None, sep=' ')
Y_test = pd.read_csv(testlabel, header=None, sep=' ')[0]

# dict1 = {}
# for col in X_train.columns:
#     dict1[col] = X_train[col].unique()

# fitting decision tree model
cols = list(X_train.columns)
dct_model = ID3Tree().fit(X_train,Y_train,cols,thresh,None,None)

# storing the tree model file
pickle.dump(dct_model.root, open(treepath, 'wb'))

# getting prediction on test data
preds = []
for testd in X_test[X_test.columns].values: 
    preds.append(dct_model.pred(testd))
preds = pd.Series(preds) 

# storing the results as csv file
preds.to_csv(outf, header=False, index=False)  

#finding tree's accuracy
accuracy = accuracy_score(Y_test, preds) 
