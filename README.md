# Machine-learning-steps

Drafting you the steps & corresponding sample code. 
Expectation  : Implement it for Customer Churn (Only consolidated Account Table) - we are already getting 87% accuracy for churn, let's see how much you can get :-) 

Part 1: EDA 

Source table: SB.LAB_LS.COAF_ACCT_CHURN_ML_POC_20181205

a) Numerical
      - Identify Continuous +  Plot histogram + Get statistics (Describe)
      - Identify Discrete + Plot histogram + Crosstab against target variable + Get statistics (Describe)
Refer:  https://github.com/sudiptawipro/FeatureEngineering/blob/master/02.1_Numerical_variables.ipynb   

b) Categorical  
      - Ordinal (Bar plot)
      - Nominal  (Bar plot)
      - Check for unique values
Refer : 
https://github.com/sudiptawipro/FeatureEngineering/blob/master/02.2_Categorical_variables.ipynb  
https://github.com/sudiptawipro/kaggle/blob/master/GiveMeSomeCredit.ipynb  (Defined function boxplotKDCContinuous - scroll down till the end ) 
https://github.com/sudiptawipro/kaggle/blob/master/GiveMeSomeCredit3.ipynb (Cross tab)


Part 2: Feature Engineering 

a)  Identify missing values 

Refer: https://github.com/sudiptawipro/kaggle/blob/master/GiveMeSomeCredit.ipynb
	 def missingValueTable
	 def missingPercent   

b) Impute missing data  

   - Any feature having more than 50% NULL -> Remove that feature
     Check ROC for both the below approaches & identify the best one.
 
   - Predict through KNN
     Refer : https://github.com/sudiptawipro/kaggle/blob/master/GiveMeSomeCredit3.ipynb
  -  Median impute 
     Refer: https://github.com/sudiptawipro/FeatureEngineering/blob/master/05.2_Mean_and_median_imputation.ipynb


c)  Identify Outlier
    - Plot KDC & boxplot for each feature & identify outliers (def boxplotKDCContinuous)
    - Detect outlier and run randomforest for both & check the accuracy 
    -  Use def detectOutlier & def IQR 
    Refer: https://github.com/sudiptawipro/kaggle/blob/master/GiveMeSomeCredit3.ipynb

d) Dummify categorical Variables
    Refer : encodeVariable in https://github.com/sudiptawipro/kaggle/blob/master/GiveMeSomeCredit3.ipynb

Part 3: Feature Selection 

For Numerical:

Approach 1: Using correlation & Random forest feature importance 

Step 1: 
Based on the correlation value identify high correlated features and drop one of them.
Expected output : List of the output feature names

Step 2:
Take the field names from the first step & apply random forest classifier to identify feature importance.
Expected output : Rank the features.

Test: Get confusion matrix, ROC for Random Forest classifier  
Refer : https://github.com/sudiptawipro/FeatureSelection/blob/master/04.2_Correlation.ipynb  
 
Approach 2: Information Gain
Expected output : Rank the features.
Test: Get confusion matrix, ROC for Random Forest classifier
Refer : https://github.com/sudiptawipro/FeatureSelection/blob/master/05.2_Information_gain.ipynb  

Approach 3: Chi-square 
Categorical - Apply label encoding 
Continuous - Binning and Apply label encoding 
Test: Get confusion matrix, ROC from Random Forest

chi-square : https://github.com/sudiptawipro/FeatureSelection/blob/master/05.3_Fisher_score.ipynb
binning : https://github.com/sudiptawipro/kaggle/blob/master/GiveMeSomeCredit3.ipynb

Approach 4: LASSO (L1)
Test: Get confusion matrix, ROC from Random Forest

Refer: https://github.com/sudiptawipro/FeatureSelection/blob/master/07.3_Lasso.ipynb 
 
Approach 5: Univariate Selection 
Test: Get confusion matrix, ROC from Random Forest

Refer : https://github.com/sudiptawipro/FeatureSelection/blob/master/05.4_Univariate_selection.ipynb 
 
Approach 6 : Univariate using ROC
Test: Get confusion matrix, ROC from Random Forest
Refer : https://github.com/sudiptawipro/FeatureSelection/blob/master/05.5_Univariate_roc_auc.ipynb

Approach 7: Step forward
Test: Get confusion matrix, ROC from Random Forest
Refer : https://github.com/sudiptawipro/FeatureSelection/blob/master/06.1_Step_forward.ipynb 
 
Approach 8: Step backward
Test: Get confusion matrix, ROC from Random Forest
Refer : https://github.com/sudiptawipro/FeatureSelection/blob/master/06.2_Step_backward.ipynb 
 
Part 4: Model Building
Randomforest & XGBoost  

Once you complete, we will get into the following areas.

Cross Validation
Grid Search
Ensemble approach: bagging,boosting,XGboost, ADAboost etc
