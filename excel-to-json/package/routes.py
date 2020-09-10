from flask import Flask, render_template,url_for,request,jsonify
from package import app,excelPath
import json
import requests
from datetime import date, timedelta,datetime
import os
import pandas as pd

######
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#Importing modeling libraries
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import RFE
from statsmodels.stats.outliers_influence import variance_inflation_factor

from imblearn.pipeline import make_pipeline as make_pipeline_imb
from imblearn.over_sampling import SMOTE
from collections import Counter

#Plotting the ROC curve.
from sklearn.metrics import roc_curve, roc_auc_score

from sklearn.metrics import confusion_matrix, f1_score, accuracy_score, roc_curve

####



@app.route('/',methods=['GET','POST'])
def index():
   return render_template('index.html')

@app.route('/json',methods=['GET','POST'])
def uploadExel():
    if request.method == "POST":
        if request.files:
            excel = request.files['fileupload']
            final_path = os.path.join(excelPath, excel.filename)

            if not os.path.exists(excelPath):
                os.mkdir(excelPath)

            excel.save(final_path)
#            data = pd.read_excel(final_path)
            # End of code

    try:

            warnings.filterwarnings('ignore')

            pd.set_option('display.max_columns', None)
            pd.set_option('display.max_rows', None)

            data = pd.read_csv(final_path)
            data.head()

            data.shape

            data.describe()

            data.info()

            # function to see meta data for a data frame
            def show_metadata(data, sort_by_nulls=False):
                metadata = pd.DataFrame(index=data.columns)
                metadata['type'] = data.dtypes
                metadata['null_percent'] = round((data.isnull().sum() / len(data.index) * 100.0), 2)
                metadata['unique_values'] = data.nunique()
                if (sort_by_nulls):
                    metadata.sort_values(by='null_percent', ascending=False, inplace=True)
                return metadata

            show_metadata(data, True)

            # Change the categorical columns of binary value from object type to int.
            # 0:Male
            # 1:Female
            data['Gender'] = data['Gender'].apply(lambda x: 0 if x == 'Male' else 1)

            # Change the categorical columns of binary value from object type to int.
            # 0:No
            # 1:Yes
            data['OverTime'] = data['OverTime'].apply(lambda x: 0 if x == 'No' else 1)
            data['Attrition'] = data['Attrition'].apply(lambda x: 0 if x == 'No' else 1)

            data.head()

            singular_col = []
            for col in data.columns:
                if data[col].nunique() == 1:
                    singular_col.append(col)
            singular_col

            data.drop(singular_col, axis=1, inplace=True)

            data.head()

            value_counts = np.round(data['Attrition'].value_counts(normalize=True) * 100.0, 2)
            ax = value_counts.plot.bar()
            for p in ax.patches:
                ax.annotate(np.round(p.get_height(), decimals=2), (p.get_x() + p.get_width() / 2., p.get_height()),
                            ha='center', va='center', xytext=(0, 10), textcoords='offset points')
            #plt.show()

            num_col = data.select_dtypes(include=['float64', 'int64']).columns

            data['JobRole'].value_counts(normalize=True).plot.bar()

            data['Department'].value_counts(normalize=True).plot.bar()

            data['EducationField'].value_counts(normalize=True).plot.bar()

            # function to show box plots for upto 30 cols
            def box_plots(cols):
                cols.sort()
                plt.figure(figsize=[20, 90])
                for v in enumerate(cols):
                    plt.subplot(30, 3, v[0] + 1)
                    sns.boxplot(y=data[v[1]])
                #plt.show()

            # Function for treating Outliers by capping them
            def treat_outliers(cols):
                for col in cols:
                    percentiles = data[col].quantile([0.01, 0.99]).values
                    data[col] = np.clip(data[col], percentiles[0], percentiles[1])

            box_plots(list(num_col))

            cols = ['YearsWithCurrManager', 'YearsSinceLastPromotion', 'YearsInCurrentRole', 'YearsAtCompany',
                    'TrainingTimesLastYear',
                    'TotalWorkingYears', 'NumCompaniesWorked', 'MonthlyIncome']
            treat_outliers(cols)

            box_plots(cols)

            plt.figure(figsize=[15, 5])
            plt.subplot(2, 3, 1)
            sns.boxplot(x='Attrition', y='MonthlyIncome', data=data)
            plt.subplot(2, 3, 2)
            sns.boxplot(x='Attrition', y='YearsSinceLastPromotion', data=data)
            plt.subplot(2, 3, 3)
            sns.boxplot(x='Attrition', y='TotalWorkingYears', data=data)
            #plt.show()

            plt.figure(figsize=[15, 5])
            plt.subplot(2, 3, 1)
            sns.boxplot(x='Attrition', y='NumCompaniesWorked', data=data)
            plt.subplot(2, 3, 2)
            sns.boxplot(x='Attrition', y='DistanceFromHome', data=data)
            plt.subplot(2, 3, 3)
            sns.boxplot(x='Attrition', y='JobSatisfaction', data=data)
            #plt.show()

            plt.figure(figsize=[8, 5])
            matrix = pd.pivot_table(data, columns='WorkLifeBalance', index='JobSatisfaction', values='Attrition')
            sns.heatmap(matrix, annot=True, cmap='Blues')
           # plt.show()

            plt.figure(figsize=[25, 10])
            sns.heatmap(data.corr(), annot=True, cmap='Blues', fmt='.2f')
            #plt.show()

            # Let us remove the highly correlated variables.
            corr_var = ['YearsAtCompany', 'TotalWorkingYears', 'JobLevel', 'YearsInCurrentRole', 'PercentSalaryHike',
                        'WorkLifeBalance', 'YearsWithCurrManager']
            data.drop(corr_var, inplace=True, axis=1)

            plt.figure(figsize=[25, 10])
            sns.heatmap(data.corr(), annot=True, cmap='Blues', fmt='.2f')
            #plt.show()

            show_metadata(data, True)

            # Creating Dummies
            col_prefixes = {
                'BusinessTravel': 'BusTravel',
                'Department': 'Dept',
                'EducationField': 'EduField',
                'JobRole': 'JobRole',
                'MaritalStatus': 'MarStatus',
                'JobInvolvement': "JobInvolvement",
                'JobSatisfaction': "JobSatisfaction",
                'EnvironmentSatisfaction': 'EnvSatisfaction',
                'Education': 'Education',
                'NumCompaniesWorked': 'NumCompaniesWorked',
                'OverTime': 'OverTime',
                'PerformanceRating': 'PerformanceRating',
                'RelationshipSatisfaction': 'RelationshipSatisfaction',
                'StockOptionLevel': 'StockOptionLevel',
                'Gender': 'Gender',
                'TrainingTimesLastYear': 'TTLY',
                'WorkLifeBalance': 'WorkLifeBalance',
            }
            for key, value in col_prefixes.items():
                if key in data.columns:
                    data[key] = data[key].astype('object')
                    dummies = pd.get_dummies(data[key], prefix=value, drop_first=True)
                    data = pd.concat([dummies, data], axis=1)
                    data.drop(key, axis=1, inplace=True)
            data.info()

            plt.figure(figsize=[35, 20])
            sns.heatmap(data.corr(), annot=True, cmap='Blues', fmt='.2f')
            #plt.show()

            corr_var = ['Dept_Sales', 'JobRole_Manager', 'BusTravel_Travel_Rarely', 'JobRole_Sales Executive',
                        'Dept_Research & Development', 'JobInvolvement_3']
            data.drop(corr_var, inplace=True, axis=1)
            data.shape

            # Putting features variable to X
            # df_train_X = data.drop(['EmployeeNumber','Attrition'], axis=1)
            df_train_X = data.drop(['Attrition'], axis=1)
            # Putting response variable to y
            df_test_y = data['Attrition']
            df_train_X.shape, df_test_y.shape

            df_train_X, df_test_y = SMOTE().fit_sample(df_train_X, df_test_y)
            print("SMOTE data distribution: {}".format(Counter(df_test_y)))

            # Splitting the data into train and test
            X_train, X_test, y_train, y_test = train_test_split(df_train_X, df_test_y, train_size=0.7, test_size=0.3,
                                                                random_state=100)

            X_train.shape, y_train.shape, X_test.shape, y_test.shape

            df_test = X_test.copy()
            X_test = X_test.drop(['EmployeeNumber'], axis=1)
            X_train = X_train.drop(['EmployeeNumber'], axis=1)

            X_train.shape, y_train.shape, X_test.shape, y_test.shape

            num_col = X_train.select_dtypes(include=['float64', 'int64']).columns

            num_col = list(num_col)
            num_col

            scaler = StandardScaler()
            col = X_train.columns
            # X_train[num_col] = scaler.fit_transform(X_train[num_col])
            X_train[col] = scaler.fit_transform(X_train[col])
            # For the test set we only transform
            # X_test[num_col] = scaler.transform(X_test[num_col])
            X_test[col] = scaler.transform(X_test[col])

            X_train.head()

            X_test.head()

            X_train.shape, X_test.shape

            logreg = LogisticRegression(random_state=43)
            rfe = RFE(logreg, 15)
            rfe = rfe.fit(X_train, y_train)
            pd.DataFrame(
                data={'Column': X_train.columns, 'Supported': rfe.support_, 'Ranking': rfe.ranking_}).sort_values(
                by=['Ranking', 'Column'])

            # List of column names that are chosen in RFE
            col = X_train.columns[rfe.support_]
            col

            # Method to perform logistic regression on y_train using training data X
            # It displays the LR summary and the VIF for the indenpendent variables
            def logistic_regression(X):
                # Add the constant
                X_scaled = sm.add_constant(X)
                # Create and fit the model
                logm = sm.GLM(y_train, X_scaled, family=sm.families.Binomial(), )
                model = logm.fit()
                # Check the paramters
                print(model.summary())
                # Draw a VIF table
                vif = pd.DataFrame()
                vif['Features'] = X.columns
                vif['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
                vif['VIF'] = round(vif['VIF'], 2)
                vif = vif.sort_values(by="VIF", ascending=False)
                print('\n')
                print(vif)
                return model

            # Creating Model 1
            logistic_regression(X_train[col])

            # Drop Column with high VIF
            # col = col.drop('JobRole_Research Director')
            col = col.drop('EduField_Life Sciences')

            # Creating Model 2
            logistic_regression(X_train[col])

            # Drop Column with high p-value
            col = col.drop('EduField_Marketing')

            # Creating Model 3
            logistic_regression(X_train[col])

            # Drop Column based on high p value
            col = col.drop('EduField_Technical Degree')

            # Creating Model 4
            logistic_regression(X_train[col])

            # Drop Column based on high p value
            col = col.drop('EduField_Other')

            # Creating Model 5
            final_model = logistic_regression(X_train[col])

            # We use absolute values for the co-efficients to ignore sign difference
            important_feature = np.abs(final_model.params[1:])
            important_feature = 100.0 * (important_feature / important_feature.max())
            important_feature

            sorted_idx = np.argsort(important_feature, kind='quicksort')
            sorted_idx

            fig = plt.figure(figsize=(8, 5))
            pos = np.arange(sorted_idx.shape[0]) + .5

            featfig = plt.figure(figsize=(10, 6))
            featax = featfig.add_subplot(1, 1, 1)
            featax.barh(pos, important_feature[sorted_idx], align='center', color='tab:green', alpha=0.8)
            featax.set_yticks(pos)
            featax.set_yticklabels(np.array(X_train[col].columns)[sorted_idx], fontsize=12)
            featax.set_xlabel('Relative Feature Importance', fontsize=12)

            plt.tight_layout()
            #plt.show()

            X_train_sm = sm.add_constant(X_train[col])
            y_pred = final_model.predict(X_train_sm)
            fpr, tpr, threshold = roc_curve(y_train, y_pred)
            auc_score = roc_auc_score(y_train, y_pred)
            plt.plot(fpr, tpr, label='ROC curve (area = %0.3f)' % auc_score)
            plt.plot([0, 1], [0, 1], 'o--')
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('ROC curve')
            plt.legend(loc="lower right")
            #plt.show()
            print(auc_score)

            X_train_scaled = sm.add_constant(X_train[col])
            train_pred = final_model.predict(X_train_scaled)
            cutoff_df = pd.DataFrame(columns=['Sensitivity', 'Specificity', 'Area under ROC'])
            # We will try cut offs between 0 and 1 with intervals of 0.02
            cut_off_options = np.arange(0.0, 1, 0.02)
            # TP = confusion[1,1] # true positive
            # TN = confusion[0,0] # true negatives
            # FP = confusion[0,1] # false positives
            # FN = confusion[1,0] # false negatives
            for c in cut_off_options:
                # Actual predicted labels
                y_train_pred_label = train_pred.map(lambda x: 1 if x > c else 0)
                # Confusion matrix for the cut off
                cm = confusion_matrix(y_train, y_train_pred_label)
                # Calculate the sensitivity as TP / (TP + FN)
                sensitivity = cm[1, 1] / (cm[1, 1] + cm[1, 0])
                # Calculate the specificity as TN / (TN + FP)
                specificity = cm[0, 0] / (cm[0, 0] + cm[0, 1])
                # Calculate the Area under ROC
                auc = roc_auc_score(y_train, y_train_pred_label)
                # Create the row for the DF
                cutoff_df.loc[c] = [sensitivity * 100.0, specificity * 100.0, auc * 100.0]
            cutoff_df.plot(figsize=[20, 7])
            plt.xticks(cut_off_options, rotation='vertical')
            plt.xlabel('Probability Cut-Off')
            plt.ylabel('Percent of metric')
            #plt.show()
            idx = np.argwhere(np.diff(np.sign(cutoff_df.Sensitivity - cutoff_df.Specificity))).flatten()
            cutoff_df.iloc[idx]

            choosen_cut_off = 0.46

            X_test_scaled = sm.add_constant(X_test[col])
            y_test_pred = final_model.predict(X_test_scaled)
            y_test_pred_label = y_test_pred.map(lambda x: 1 if x > choosen_cut_off else 0)
            # Confusion matrix
            cm = confusion_matrix(y_test, y_test_pred_label)
            # Calculate the sensitivity as TP / (TP + FN)
            sensitivity = cm[1, 1] / (cm[1, 1] + cm[1, 0])
            print('Sensitivity on Test set {}'.format(sensitivity))
            print('Accuracy on Test set {}'.format(accuracy_score(y_test, y_test_pred_label)))

            df_test['PredictedAttrtition'] = y_test_pred_label

            df_test[['EmployeeNumber', 'PredictedAttrtition']]

            # List of 10 Employees showing who might leave the company.
            df_test[df_test['PredictedAttrtition'] == 1].EmployeeNumber.head(10)

            # Begin Controller code

            result = {'data': df_test[['EmployeeNumber', 'PredictedAttrtition']].to_dict(),
                      'sensitivity': round((sensitivity*100),2), 'accuracy': round((accuracy_score(y_test, y_test_pred_label)*100),2)}

            #return jsonify(df_test[['EmployeeNumber', 'PredictedAttrtition']].head().to_dict())
            return jsonify(result)

            #return render_template('data.html', resp = df_test[['EmployeeNumber', 'PredictedAttrtition']].head())
    except:
        return jsonify({'error':'something wrong happened'})