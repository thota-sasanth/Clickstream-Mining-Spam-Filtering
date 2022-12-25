import argparse
import pandas as pd
import math
parser = argparse.ArgumentParser()

# Providing option to specify file names from the command line as flag inputs
parser.add_argument("-f1", "--train_dataset", help="Training Dataset Filename")
parser.add_argument("-f2", "--test_dataset",  help="Testing Dataset Filename")
parser.add_argument("-o", "--output_file",    help="Output Filename")

args = parser.parse_args() # List of all arguments given as input

# Function defined to train a Naive Bayes model and predict the labels of 
# the given testing dataset
def predictTestDataset():
    typeCounts = dict()
    trainingDF = pd.read_csv(args.train_dataset, header=None)
    
    
    # Getting the counts of 'spam' and 'ham' email types from the dataframe
    typeCounts['spam'] = len(dataframe[dataframe['emailType'].str.contains('spam')])
    typeCounts['ham']  = len(dataframe[dataframe['emailType'].str.contains('ham')])

    # Running a loop through each email entry in the training data and populating -
    # words as key and (spam word, ham word) tuple as its value
    for i in range(0,len(dataframe['emailWordFreqs'])):
        emailWordFreqs = dataframe['emailWordFreqs'][i]
        
            if emailType == 'ham':
                wordFrequencies[emailWordFreqs[j]] = (spamWords, hamWords + wordFreq)
            else:
                wordFrequencies[emailWordFreqs[j]] = (spamWords + wordFreq, hamWords)

    # Calculating the conditional probability of each word in the wordFrequencies dictionary using 
    # laplace smoothing as taught in the lecture
    total = typeCounts['spam'] + typeCounts['ham']
    conditionalProb = dict()
    priorSpamProb = float(typeCounts['spam']) / total
    priorHamProb = float(typeCounts['ham']) / total
    vocabularySize = len(wordFrequencies)
    

    # Splitting the given testing dataset based on ' ' delimiter and -
    # dividing them into colums in a dataframe
    testingDF = pd.read_csv(args.test_dataset, header=None)
    temp = testingDF[0].str.split(' ')

    headers = headers = ['emailID', 'emailType', 'emailWordFreqs']
    dataframeTest = pd.DataFrame(index=range(len(temp)), columns=headers)

    # Converting the dataset into a dataframe for eaiser manipulation
    

    predictedLabels = list()

    # Initializing the values of true +ves, -ves & false +ves, -ves to 0
    true_negatives = 0
    true_positives = 0
    false_negatives = 0
    false_positives = 0

    # Iterating over each row in the testing dataset and applying the Naive Bayes formula taught in lecture to calculate 
    # P(spam | word) or P(ham | word)
    for i in range(len(dataframeTest['emailWordFreqs'])):
        spamProb = 0
        hamProb = 0
        emailID = dataframeTest['emailID'][i]
        

        spamProb += math.log(priorSpamProb)
        hamProb += math.log(priorHamProb)

        # Classifying the email as spam if the probability that it is spam is greater than or
        # equal to the probability that the email is ham
        if spamProb >= hamProb:
            predictedLabels.append([emailID, 'spam']) # Classifying the email as spam
            
        else:
            predictedLabels.append([emailID, 'ham']) # Classifying the email as ham
            

    # Calculating accuracy, and f1Score as per the above calculated values
    accuracy = (true_positives + true_negatives) / float(dataframeTest.shape[0])
    precision = (true_positives) / float(true_positives + false_positives)
    recall = (true_positives) / float(true_positives + false_negatives)
    f1Score = 2 * precision * recall / float(precision + recall)
    
    # Writing output to csv file named as per the given output filename in the command line
    dataframeOutput = pd.DataFrame(predictedLabels)
    dataframeOutput.to_csv(args.output_file, index=False, header=False, sep =' ')
    
    print("The prediction metrics for Naive Bayes Algorithm using Laplace Smoothing: ")
    print("Accuracy:", round(accuracy * 100, 2), '%')
    print("F1 Score:", round(f1Score * 100, 2), '%')


predictTestDataset()
