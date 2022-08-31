import numpy as np
import pandas as pd
from Company import Company

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_absolute_error
from sklearn.tree import DecisionTreeClassifier

class Model(object):
    def __init__(self, Company):
        # Object containing values of Company (DF, TS)
        self.Company = Company
        # Dataframe with combined hl and ts data #
        self.df_BACKUP = self.Company.df
        self.df = self.PrepareDataFrame()
        
        # models - regression
        # self.rgr_01 = self.RegressionModel01()
               
        # model classification
        self.clf_01 = self.ClassificationModel01()

    def ClassificationModel01(self):
        X, y, X_orig, ts_cols, num_cols = self.ColumnSelector('clf')
        X_train, X_test, y_train, y_test = self.Split_TS(X, y)
        
        # Begin pipeline
        # Prepare text data for model
        transform_text = Pipeline([
        ('vect', CountVectorizer()), #
        ('tfidf', TfidfTransformer()),
        ])
        
        preprocessor = ColumnTransformer(
            transformers=[
                ('text_transform', transform_text, 'text'),
                ('select_ts_cols', 'passthrough', ts_cols),
                ('select_num_columns', 'passthrough', num_cols)
            ]
        )
    
        my_pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),            
            ('estimator', DecisionTreeClassifier(random_state=0)), #accuracy: 0.6027874564459931 # EVERYTHING - storyText!
        ])
        
        my_pipeline.fit(X_train, y_train)
        preds = my_pipeline.predict(X_test)
        print('accuracy:',accuracy_score(y_test, preds))
        
        
    def RegressionModel01(self):
        X, y, X_orig, ts_cols, num_cols = self.ColumnSelector('rgr')
        X_train, X_test, y_train, y_test = self.Split_TS(X, y)
        
        # Begin pipeline
        # Prepare text data for model
        transform_text = Pipeline([
            ('vect', CountVectorizer()), # stop_words='english', min_df=0, max_features=None, max_df=0.25, analyzer='char'
            ('tfidf', TfidfTransformer()), # norm='l1',smooth_idf=True, sublinear_tf=False, use_idf=False
        ])
        
        # No categorical variables
        # cat_encoder = Pipeline([
            # ('encoder', OrdinalEncoder())
        # ])
        
        # Pre process columns (apply transformation to data)
        preprocessor = ColumnTransformer(
            transformers=[
                ('text_transform', transform_text, 'text'),
                ('select_ts_cols', 'passthrough', ts_cols),
                ('select_num_columns', 'passthrough', num_cols),
                # ('select_cat_cols', cat_encoder, cat_cols) # No Categorical Variables
            ]
        )
        
        # Pass transformed data through RFR model
        my_pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('estimator', RandomForestRegressor(random_state=0)), # , n_estimators=250, max_features='auto',warm_start=True
        ])
        
        # Predictions
        my_pipeline.fit(X_train, y_train)
        preds = my_pipeline.predict(X_test)
        temp_df = self.regr_to_clf(X_orig, preds, X_test.index)
        print('accuracy:', accuracy_score(temp_df['Increase'].astype('int'), temp_df['Pred Increase'].astype('int')) , 'mae:', mean_absolute_error(y_test, preds))
        return temp_df
        
    
    # converts regression prediction to classfication (is this a thing?)
    def regr_to_clf(self, df, preds, test_index):
        new_df = pd.DataFrame({
            'CLOSE': df['CLOSE'].loc[test_index],
            '30M_CLOSE': df['30M_CLOSE'].loc[test_index],
            'Increase': df['Increase'].loc[test_index],
            'Pred 30M_CLOSE': preds
            }, index=test_index)
        new_df['Pred Increase'] = new_df['Pred 30M_CLOSE'] > new_df['CLOSE']
        new_df['Pred Increase'] = new_df['Pred Increase']*1
        return new_df
    
    
    # Final preparing before modelling
    def PrepareDataFrame(self):
        df = self.df_BACKUP.copy()
        df = self.SetIncrease(df)
        return df
    
    #### NOTE:  This code is awful.
    #########:  Instead rewrite for data_type (?)
    #########:  Working but messy and ugly and ugly and ugly and ugly
    #########:  Very embarrassing lol
    # Selects columns and their behaviour for the given model
    # Used for data pipeline (How to do better??)
    
    # clf_or_regr <string>[clf|rgr]
    def ColumnSelector(self, clf_or_rgr):
        # Output variable y
        y = None
        
        # Categorical columns
        # cat_cols = ['Score'] # Not Implemented Yet (Sentiment analysis for news text)
        
        # ps_cols = ['Polarity', 'Subjectivity'] # More sentiment Analysis
        
        # Timeseries Columns (Does this work as intended) <-- Question self knowledge & understanding of model
        ts_cols = [i for i in self.df.columns if 't-' in i]
        
        # Numerical columns (They are actually double not int)
        num_cols = ['HIGH', 'LOW', 'OPEN', 'CLOSE']
        
        # Independent variable X
        X = self.df[['text','30M_CLOSE','Increase'] + num_cols + ts_cols]
        
        # Backup of X
        X_orig = X.copy()
        
        # Remove input variable depending on model 
        #   1)  classification
        #   2)  regression
        if (clf_or_rgr == 'clf'):
            y = X.pop('Increase')
            X.pop('30M_CLOSE')
        elif (clf_or_rgr == 'rgr'):
            y = X.pop('30M_CLOSE')
            X.pop('Increase')
        
        return X, y, X_orig, ts_cols, num_cols
    
    # Timeseries split (needed because data is timeseries-esque)
    def Split_TS(self, X, y):
        # create training data for first 0.67 of data
        # create test data for 1 - 0.67 of data
        X_train, X_test = np.split(X, [int(.67 * len(X))])
        
        # subset the training y values for the given index of X
        y_train = y.loc[X_train.index]
        # same as above but for test
        y_test = y.loc[X_test.index]
        
        return X_train, X_test, y_train, y_test
    
    # Set [1/0][T/F] if stock closes higher after news headline
    def SetIncrease(self, df):
        df['Increase'] = df['30M_CLOSE'] > df['CLOSE']
        df['Increase'] = df['Increase']*1
        return df