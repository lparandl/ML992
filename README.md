# Sentiment Analysis on Movie Reviews

This project is for ML992 Lecture and done in two phases.  

## Phase 1:  
The task was to represent a general pipeline for simple binary classification on the dataset, test the model on test set-1 and report the results in a .csv file. 
In this phase following steps are implemented:

1. Data Preprocessing
   - Tokenizing
   - Stopwords and Punctuation Removal
   - Lemmatization

2. Data Vectorization
    - TF-IDF

3. Splitting the Dataset
    - Using `train_test_split` technique to evaluate the performance by dividing the dataset into two subsets: Train Set, Test Set  
        **Train Dataset:** Used to fit the machine learning model.  
        **Test Dataset:**  Used to evaluate the fit machine learning model.

4. Model Selection to Train on Dataset
    - Multinomial Naive Bayes

5. Predicting the Testset Results

6. Making the Confusion Matrix  

<p align="center">
    <img src="https://github.com/lparandl/NLP/blob/main/Phase%201/cm.JPG">
</p>

7. Plotting the Confusion Matrix

<p align="center">
    <img src="https://github.com/lparandl/NLP/blob/main/Phase%201/plot.png">
</p>

8. k-Fold Cross-Validation
    - Using a 10-fold resampling procedure to evaluate the performance of the model by splitting the dataset into ten groups and proceed the following for each group:
        1. Take the group as a holdout or test data set
        2. Take the remaining groups as a training data set
        3. Fit a model on the training set and evaluate it on the test set
        4. Retain the evaluation score and discard the model
        5. Summarize the skill of the model using the sample of model evaluation scores  
      
     ▪️ **Accuracy:** 85.98 %  
     ▪️ **Standard Deviation:** 0.47 %

9. Testing the Model

10. Preprocess the Test Dataset
11. TF-IDF Vectorization on Testset
12. Predicting the Test Results
     - Storing the test results into [`Phase1/predict_input`](https://github.com/lparandl/NLP/blob/main/Phase%201/predict_input.csv) .csv file.

## Phase 2:
The goal is to find the best model with the highest accuracy. 

