import itertools
import math

import matplotlib.pyplot as plt
import argparse
import pandas as pd
import numpy as np

from sklearn.metrics import confusion_matrix
from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTETomek, SMOTEENN
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix as sk_confusion_matrix
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA, SparsePCA
from sklearn.model_selection import KFold, StratifiedKFold, TimeSeriesSplit
from sklearn import svm
from imblearn.under_sampling import TomekLinks
from sklearn.neighbors import KNeighborsClassifier


import plotting as plotting
from currency_converter import CurrencyConverter

import argparse

#format: txid,bookingdate,issuercountrycode,txvariantcode,bin,amount,currencycode,shoppercountrycode,shopperinteraction,simple_journal,cardverificationcodesupplied,cvcresponsecode,creationdate,accountcode,mail_id,ip_id,card_id

def preprocessing(df, unix, diff):
    df.drop(df[df['simple_journal'] == 'Refused'].index, inplace=True)
    # df = df[df['shoppercountrycode'] != 'GB']
    df['cvcresponsecode'] = df['cvcresponsecode'].map(lambda x: float(x))
    df.loc[df['cvcresponsecode'] >= 3.0, 'cvcresponsecode'] = 3.0
    df['cvcresponsecode'] = df['cvcresponsecode'].apply(lambda x: x if x <=2.0 else 3.0)
    df['amount'] = df['amount'].map(lambda x: float(x))
    df['card_id'] = df['card_id'].map(lambda x: x[4:])
    df['ip_id'] = df['ip_id'].map(lambda x: x[2:])
    df['mail_id'] = df['mail_id'].map(lambda x: x[5:])
    df['country'] = df['shoppercountrycode'].apply(lambda x: x if x in ['MX', 'AU', 'GB'] else 'other')
    df = df.drop(['shoppercountrycode'], axis=1)
    if unix:
        df['creationdate_unix'] = pd.DatetimeIndex(df['creationdate']).astype(np.int64) / 1000000000
        df['creationdate_unix'].sub(df['creationdate_unix'].min(), axis=0)
        df['creationdate_unix'] = df['creationdate_unix'].map(lambda x: math.log(x))
    if diff:
        df = time_diff(df)
    df['AUD_currency'] = df[['currencycode', 'amount']].apply(lambda x: CurrencyConverter.convert_currency_from_AUD(x), axis=1)
    df.dropna(axis=0, how='any', inplace=True)
    df = df[~df.isin(['NaN', 'NaT', 'NA']).any(axis=1)]
    return df

def classification(X_train, y_train, X_test, classyType, cross_val = False):
    if classyType == 'NB':
        print('Naive Bayes')
        model = GaussianNB() # -- good
    elif classyType == 'MLP':
        print('MLP')
        model = MLPClassifier() # -- good?
    elif classyType == 'DT':
        print('Decision Tree')
        model = DecisionTreeClassifier(max_depth=10) # -- good?
    elif classyType == 'RF':
        print('Random Forest')
        model = RandomForestClassifier()
    elif classyType == 'LR':
        print('Linear Regression')
        model = LogisticRegression() # -- bad
    elif classyType == 'KNN':
        print('KNN')
        model = KNeighborsClassifier()

    model.fit(X_train, y_train)
    predict_prob = model.predict_proba(X_test)[:, 1]
    predict_bin = model.predict(X_test)
    return model, predict_prob, predict_bin


def pca(X_train, X_test):
    pca_model = PCA(n_components=0.99, svd_solver='full')
    X_train = pca_model.fit_transform(X_train)
    X_test = pca_model.transform(X_test)
    print('PCA k={}'.format(np.shape(X_train)[1]))
    return X_train, X_test


def plot_ROC(y_test, smoted_predict_prob, unsmoted_predict_prob, smoted_predict_bin, unsmoted_predict_bin):
    # Find and plot AUC
    sm_false_positive_rate, sm_true_positive_rate, sm_thresholds = roc_curve(y_test, smoted_predict_prob)
    sm_roc_auc = auc(sm_false_positive_rate, sm_true_positive_rate)

    unsm_false_positive_rate, unsm_true_positive_rate, unsm_thresholds = roc_curve(y_test, unsmoted_predict_prob)
    unsm_roc_auc = auc(unsm_false_positive_rate, unsm_true_positive_rate)

    print('smoted AUC:   {}'.format(sm_roc_auc))
    print('unsmoted AUC: {}'.format(unsm_roc_auc))

    # confusion matrix
    con_matrix = [['TN', 'FN'],
                  ['FP', 'TP']]
    print(con_matrix)
    print('smoted: \n', confusion_matrix(y_test, smoted_predict_bin))
    print('unsmoted: \n', confusion_matrix(y_test, unsmoted_predict_bin))

    plt.title('ROC')
    plt.plot(sm_false_positive_rate, sm_true_positive_rate, label=('AUC-smoted' + '= %0.2f' % sm_roc_auc))
    plt.plot(unsm_false_positive_rate, unsm_true_positive_rate, label=('AUC-unsmoted' + '= %0.2f' % unsm_roc_auc))
    plt.legend(loc='lower right', prop={'size': 8})
    plt.plot([0, 1], [0, 1], color='lightgrey', linestyle='--')
    plt.xlim([-0.05, 1.0])
    plt.ylim([0.0, 1.05])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()

def cross_validation(clsType, X, y, df):
    k_fold = StratifiedKFold(10, shuffle=True)#TimeSeriesSplit(n_splits=11)#StratifiedKFold(10)
    final_conf_matrix = np.zeros((2, 2))
    for k, (train, test) in enumerate(k_fold.split(X, y)):
        if clsType == 'RF':

            print('Random Forest')
            model = RandomForestClassifier(n_estimators=100, max_depth=10)#class_weight='balanced')
        elif clsType == 'SVM':
            print('SVM')
            model = svm.LinearSVC()#class_weight='balanced')
        elif clsType == 'MLP':
            print('MLP')
            model = MLPClassifier()#class_weight='balanced')
        elif clsType == 'NB':
            print('NB')
            model = GaussianNB()  # class_weight='balanced')
        elif clsType == 'DT':
            print('DT')
            model = DecisionTreeClassifier(max_depth=10)
        elif clsType == 'LR':
            print('LR')
            model = LogisticRegression()
        print('Iteration {}'.format(k))
        sm =  SMOTE()# kind_smote='svm' #TomekLinks()# SMOTETomek()  #
        X_train, y_train = sm.fit_sample(X[train], y[train])
        X_test = X[test]; y_test = y[test]

        ######
        # for oversampling to use time diff feature
        # X_train = calc_time_diff(X_train, df)
        # X_test = calc_time_diff(X_test, df)
        ######

        # if 'creationdate_unix' in
        print('Train data size: {}'.format(np.shape(X_train)))
        print('Test data size: {}'.format(np.shape(X_test)))
        print('Test data fraud #{}'.format((y_test == 1).sum()))
        # X_train, X_test = pca(X_train, X_test)
        model.fit(X_train, y_train)
        score = model.score(X_test, y_test)
        y_pred = model.predict(X_test)
        conf_matrix = confusion_matrix(y_test, y_pred)
        for i in range(len(conf_matrix)):
            final_conf_matrix[i] = [sum(x) for x in zip(final_conf_matrix[i], conf_matrix[i])]
        print('Score: {} Confusion matrix: \n {}'.format(score, conf_matrix))
    print('Final confusion matrix: \n {}'.format(final_conf_matrix))

def calc_time_diff(X, df):
    df_X = pd.DataFrame(X, columns=df.columns.values)
    df_X['creationdate_unix'] = df_X['creationdate_unix'].map(lambda x: round(math.exp(x)))
    df_X['creationdate'] = pd.to_datetime(df_X['creationdate_unix'], unit='s')
    df_X = plotting.time_diff(df_X)
    df_X = df_X.drop(['creationdate'], axis=1)
    X = df_X.values
    return X

def time_diff(df):
    # for undersampling
    df['date'] = pd.to_datetime(df['creationdate'])

    df['diff_time'] = df.sort_values(['card_id', 'creationdate']).groupby('card_id')['date'].diff()
    time = pd.DatetimeIndex(df['diff_time'])
    df['diff_time_min'] = time.hour * 60 + time.minute + 1

    df['diff_time_min'] = df['diff_time_min'].fillna(0)
    df = df.drop(['date', 'diff_time'], axis=1)
    return df

def compute(df, clsType, crossVal):
    # df['creationdate'] = pd.to_datetime(df['creationdate'], unit='s')

    y = df['simple_journal'].values
    y = label_binarize(y, classes=['Settled', 'Chargeback'])
    y = np.concatenate(y, 0)

    df = df.drop(['simple_journal', 'creationdate', 'bookingdate', 'amount', 'shopperinteraction'], axis=1)#, 'card_id' 'mail_id', 'ip_id' 'amount


    #one hot encoding -- used for ROC curves
    # df = pd.get_dummies(df, columns=['issuercountrycode', 'txvariantcode', 'currencycode', 'country',
    #                                  'shopperinteraction', 'accountcode', 'cardverificationcodesupplied'])

    ######
    #categorical into numerical -- used for ass3
    categorical_cols = ['txvariantcode',
                        'currencycode',
                        'cardverificationcodesupplied',
                        'cvcresponsecode',
                        'accountcode',
                        'country',
                        'issuercountrycode']

    # convert to categorical data
    df[categorical_cols] = df[categorical_cols].apply(lambda x: x.astype('category'))

    for col in categorical_cols:
        df[col] = df[col].cat.codes
    ######

    print(df.columns.values)

    # sorting values
    # df = df.sort_values(by=['creationdate_unix'])
    # drop if you don't want this feature
    # df = df.drop(['creationdate_unix'], axis=1)
    print(df.head())

    print('Amount of fraud data: {} and non fraud: {}'.format((y == 1).sum(), (y == 0).sum()))
    X = df.values

    if crossVal:
        cross_validation(clsType, X, y, df)
    else:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        print('Test data size: ' + str(np.shape(X_test)))
        print('Test data fraud #{}'.format((y_test == 1).sum()))
        # SMOTE Data
        sm = SMOTETomek()  # SMOTE()#
        X_resampled, y_resampled = sm.fit_sample(X_train, y_train)

        print('Train data size: {}'.format(np.shape(X_resampled)))

        if clsType == 'RF':
            X_resampled, X_test_sm = pca(X_resampled, X_test)
            X_train, X_test_unsm = pca(X_train, X_test)
        else:
            X_test_sm = X_test
            X_test_unsm = X_test
        smoted_model, smoted_predict_prob, smoted_predict_bin = classification(X_resampled, y_resampled, X_test_sm, clsType)
        unsmoted_model, unsmoted_predict, unsmoted_predict_bin = classification(X_train, y_train, X_test_unsm, clsType)

        print('smoted:   ' + str(smoted_model.score(X_test_sm, y_test)))
        print('unsmoted: ' + str(unsmoted_model.score(X_test_unsm, y_test)))

        dotfile = open("dt.dot", 'w')
        export_graphviz(smoted_model, out_file=dotfile, feature_names=df.columns.values)
        dotfile.close()

        plot_ROC(y_test, smoted_predict_prob, unsmoted_predict, smoted_predict_bin, unsmoted_predict_bin)

def parse_arguments():
    parser = argparse.ArgumentParser()

    # General commands
    # parser.add_argument('plot', help='Plot', action='store_true')
    parser.add_argument('cls', help='Classifier type', action='store', choices=['NB', 'MLP', 'DT', "RF", 'SVM', 'LR', 'KNN'], default='default')

    args = parser.parse_args()
    return args


def run(cls, unix, diff):
    df = pd.read_csv('data_for_student_case.csv', index_col=0)  # dataframe
    df = preprocessing(df, unix, diff)
    # true for cross validation
    compute(df, cls, True)


def main():
    args = parse_arguments()

    if args.cls is not None:
        print('1')
        run(args.cls, True, False)
        print('2')
        # run(args.cls, True, True)


main()