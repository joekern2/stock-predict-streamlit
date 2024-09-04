# -*- coding: utf-8 -*-
"""
Created on Sun Dec 11 13:51:42 2022

@author: Joseph Kern
"""

import python
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# import scikit-learn
from sklearn import neighbors #, datasets, svm, linear_model, tree
from sklearn.model_selection import (cross_val_score, train_test_split, 
StratifiedKFold)
from sklearn.metrics import (confusion_matrix, precision_recall_curve, 
                             average_precision_score, roc_curve, auc)
from scipy import interp
import pickle

SKLEARN_ALLOW_DEPRECATED_SKLEARN_PACKAGE_INSTALL=True




def run():
    
    st.title('Machine Learning Algorithm to predict whether a stock will close up or down the following day')
    
    st.header('Options listed below are the top ten stocks, sorted by volume')
    
    st.write('***Disclaimer***: This is a school project, and not intended for use on the actual market. ',
             'I am not a registered investment, legal or tax advisor or broker / dealer. All investment / ',
             'financial opinions expressed below are from the personal research and experience of Joseph Kern',
             ' and are intended as educational material. Although best efforts are made to ensure that all',
             ' information is accurate and up to date, occasionally unintended errors and misprint may occur.',
             'DO NOT MAKE INVESTMENTS BASED SOLELY OFF THE RESULTS OF THE ALGORITHM.')
    
    st.write('\n=================================================================\n')
    st.write('\n')
    st.write('\n')
    st.header('What Stock would you like to predict for tomorrow?')
    
    ticker = st.radio(
        "",
        ('Tesla', 'Apple', 'Amazon', 'Advanced Micro Devices', 'Carvana Co', 
         'Nvidia', 'Lucid Group', 'Ford', 'Carnival Corporation', 'Bank of America'))
    
    st.write('\n')
    st.write('\n')
    
    ### Choose from top 10 traded stocks by volume
    
    if ticker == 'Tesla':
        # TSLA - Tesla
        string = r'https://query1.finance.yahoo.com/v7/finance/download/TSLA?period1=1639254003&period2=1770790003&interval=1d&events=history&includeAdjustedClose=true'
    elif ticker == 'Apple':
        # AAPL - Apple
        string = r'https://query1.finance.yahoo.com/v7/finance/download/AAPL?period1=1639235859&period2=1770771859&interval=1d&events=history&includeAdjustedClose=true'
    elif ticker == 'Amazon':
        # AMZN - Amazon
        string = r'https://query1.finance.yahoo.com/v7/finance/download/AMZN?period1=1639236090&period2=1770772090&interval=1d&events=history&includeAdjustedClose=true'
    elif ticker == 'Advanced Micro Devices':
        # AMD - Advanced Micro Devices
        string = r'https://query1.finance.yahoo.com/v7/finance/download/AMD?period1=1639236117&period2=1770772117&interval=1d&events=history&includeAdjustedClose=true'
    elif ticker == 'Carvana Co':
        # CVNA - Carvana Co
        string = r'https://query1.finance.yahoo.com/v7/finance/download/CVNA?period1=1639236142&period2=1770772142&interval=1d&events=history&includeAdjustedClose=true'
    elif ticker == 'Nvidia':
        # NVDA - Nvidia
        string = r'https://query1.finance.yahoo.com/v7/finance/download/NVDA?period1=1639236157&period2=1770772157&interval=1d&events=history&includeAdjustedClose=true'
    elif ticker == 'Lucid Group':
        # LCID - Lucid Group
        string = r'https://query1.finance.yahoo.com/v7/finance/download/LCID?period1=1639236173&period2=1770772173&interval=1d&events=history&includeAdjustedClose=true'
    elif ticker == 'Ford':
        # F - Ford
        string = r'https://query1.finance.yahoo.com/v7/finance/download/F?period1=1639236184&period2=1770772184&interval=1d&events=history&includeAdjustedClose=true'
    elif ticker == 'Carnival Corporation':
        # CCL - Carnival Corporation
        string = r'https://query1.finance.yahoo.com/v7/finance/download/CCL?period1=1639236205&period2=1770772205&interval=1d&events=history&includeAdjustedClose=true'
    elif ticker == 'Bank of America':
        # BAC - Bank of America
        string = r'https://query1.finance.yahoo.com/v7/finance/download/BAC?period1=1639236220&period2=1770772220&interval=1d&events=history&includeAdjustedClose=true'
        
    
    stockdf = pd.read_csv(string)
    
    st.write('Currently Closed at: ', stockdf['Close'][len(stockdf) - 1])
    
    
    #Drop Date column as it is not important to analysis
    stockdf = stockdf.drop(['Date'], axis=1)
    
    ### TODAYS NUMBERS
    todx = stockdf.iloc[-1].tolist()
    todx = [[todx[0], todx[1], todx[2], todx[3], todx[4], todx[5]]]
    
    
    if ticker == 'Tesla':
        # TSLA - Tesla
        loaded_model = pickle.load(open('tsla_model.sav', 'rb'))
        result = loaded_model.predict(todx)[0]
    elif ticker == 'Apple':
        # AAPL - Apple
        loaded_model = pickle.load(open('aapl_model.sav', 'rb'))
        result = loaded_model.predict(todx)[0]
    elif ticker == 'Amazon':
        # AMZN - Amazon
        loaded_model = pickle.load(open('amzn_model.sav', 'rb'))
        result = loaded_model.predict(todx)[0]
    elif ticker == 'Advanced Micro Devices':
        # AMD - Advanced Micro Devices
        loaded_model = pickle.load(open('amd_model.sav', 'rb'))
        result = loaded_model.predict(todx)[0]
    elif ticker == 'Carvana Co':
        # CVNA - Carvana Co
        loaded_model = pickle.load(open('cvna_model.sav', 'rb'))
        result = loaded_model.predict(todx)[0]
    elif ticker == 'Nvidia':
        # NVDA - Nvidia
        loaded_model = pickle.load(open('nvda_model.sav', 'rb'))
        result = loaded_model.predict(todx)[0]
    elif ticker == 'Lucid Group':
        # LCID - Lucid Group
        loaded_model = pickle.load(open('lcid_model.sav', 'rb'))
        result = loaded_model.predict(todx)[0]
    elif ticker == 'Ford':
        # F - Ford
        loaded_model = pickle.load(open('f_model.sav', 'rb'))
        result = loaded_model.predict(todx)[0]
    elif ticker == 'Carnival Corporation':
        # CCL - Carnival Corporation
        loaded_model = pickle.load(open('ccl_model.sav', 'rb'))
        result = loaded_model.predict(todx)[0]
    elif ticker == 'Bank of America':
        # BAC - Bank of America
        loaded_model = pickle.load(open('bac_model.sav', 'rb'))
        result = loaded_model.predict(todx)[0]

    
    
    
    
    #add column showing if the following day closes higher than the current
# =============================================================================
#     newcol = []
#     for i in range(len(stockdf)):
#         if i == len(stockdf)-1:
#             newcol.append(0)
#         else:
#             nextrow = stockdf.iloc[i+1:i+2].values.tolist()
#             currow = stockdf.iloc[i:i+1].values.tolist()
#             nextclose = nextrow[0][3]
#             curclose = currow[0][3]
#             v = 0
#             if curclose <= nextclose:
#                 v = 1
#             newcol.append(v)
#             
#             
#     stockdf['bulltomorrow'] = newcol
#     
#     
#     # Drop todays numbers to get prediction for tmrw
#     stockdf.drop(stockdf.tail(1).index,inplace=True)
#     
#     
#     # split data and target
#     X = stockdf.iloc[:, :6].values
#     Y = stockdf.iloc[:, -1].values
#     
#     
#     X_train, X_test, y_train, y_test = train_test_split(X, Y, random_state=0)
#     
#     
#     n_neighbors = 5
#     clf = neighbors.KNeighborsClassifier(n_neighbors)
#     y_pred = clf.fit(X_train, y_train).predict(X_test)
#     
# =============================================================================
    
    ps = ''
    
    #clf.predict(todx)[0]
    if result == 0:
        ps = '\'s price is predicted to ***fall*** tomorrow.\n'
    else:
        ps = '\'s price is predicted to ***rise*** tomorrow.\n'
    
    st.write(ticker, ps, ' The current daily chart over the last year is shown below.')
    plotdf = pd.DataFrame()
    plotdf['Y'] = stockdf['Close']
    st.line_chart(plotdf)


if __name__ == '__main__':
    run()
