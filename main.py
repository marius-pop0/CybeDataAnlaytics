import matplotlib.pyplot as plt
import argparse
import pandas as pd
import numpy as np

from sklearn.metrics import confusion_matrix
from imblearn.over_sampling import SMOTE
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix as sk_confusion_matrix
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA

import plotting as plotting

import argparse


#format: txid,bookingdate,issuercountrycode,txvariantcode,bin,amount,currencycode,shoppercountrycode,shopperinteraction,simple_journal,cardverificationcodesupplied,cvcresponsecode,creationdate,accountcode,mail_id,ip_id,card_id

def preprocessing(df):
    df.drop(df[df['simple_journal'] == 'Refused'].index, inplace=True)
    # df = df[df['shoppercountrycode'] != 'GB']
    df['cvcresponsecode'] = df['cvcresponsecode'].map(lambda x: float(x))
    df['amount'] = df['amount'].map(lambda x: float(x))
    df['card_id'] = df['card_id'].map(lambda x: x[4:])
    df['ip_id'] = df['ip_id'].map(lambda x: x[2:])
    df['mail_id'] = df['mail_id'].map(lambda x: x[5:])
    # df['creationdate_unix'] = pd.DatetimeIndex(df['creationdate']).astype(np.int64) / 1000000000 #very bad results!!
    # df['AUD_currency'] = df[['currencycode', 'amount']].apply(lambda x: currencyconverter.convert_currency_from_aud(x), axis=1)
    df.dropna(axis=0, how='any', inplace=True)
    df = df[~df.isin(['NaN', 'NaT']).any(axis=1)]
    return df

def classification(X_train, y_train, X_test, classyType):
    if classyType == 'NB':
        print('Naive Bayes')
        model = GaussianNB() # -- good
    elif classyType == 'MLP':
        print('MLP')
        model = MLPClassifier() # -- good?
    elif classyType == 'DT':
        print('Decision Tree')
        model = DecisionTreeClassifier() # -- good?
    elif classyType == 'RF':
        print('Random Forest')
        model = RandomForestClassifier()
    elif classyType == 'LR':
        print('Linear Regression')
        model = LogisticRegression() # -- bad
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

def smote(df, ratio, clsType,  toPlot=False):
    # print(df.isnull().any())

    # df['creationdate'] = pd.to_datetime(df['creationdate'], unit='s')
    df = pd.get_dummies(df, columns=['issuercountrycode', 'txvariantcode', 'currencycode', 'shoppercountrycode',
                                     'shopperinteraction', 'accountcode', 'cardverificationcodesupplied'])

    y = df['simple_journal'].values
    print(y)
    y = label_binarize(y, classes=['Settled', 'Chargeback'])
    y = np.concatenate(y, 0)
    # print(y)
    df = df.drop(['simple_journal', 'creationdate', 'bookingdate', 'mail_id', 'ip_id', 'card_id'], axis=1)
    print('Amount of fraud data: {} and non fraud: {}'.format((y == 1).sum(), (y == 0).sum()))
    X = df.values

    length = np.shape(X)[0]#, np.shape(y))
    train_len = round(length*ratio)
    X_train = X[:train_len, :]
    y_train = y[:train_len]
    X_test = X[train_len:, :]
    y_test = y[train_len:]
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
    print('Size of test data: ' + str(np.shape(X_test)))
    print('# Fraud transactions in test data: {}'.format((y_test == 1).sum()))

    # SMOTE Data
    sm = SMOTE(random_state=15)
    X_resampled, y_resampled = sm.fit_sample(X_train, y_train)

    # print(np.shape(X_resampled)[1]==np.shape(X_train)[1])

    if toPlot:
        if clsType == 'RF':
            X_resampled, X_test_sm = pca(X_resampled, X_test)
            X_train, X_test_unsm = pca(X_train, X_test)
        smoted_model, smoted_predict_prob, smoted_predict_bin = classification(X_resampled, y_resampled, X_test_sm, clsType)
        unsmoted_model, unsmoted_predict, unsmoted_predict_bin = classification(X_train, y_train, X_test_unsm, clsType)

        print('smoted:   ' + str(smoted_model.score(X_test_sm, y_test)))
        print('unsmoted: ' + str(unsmoted_model.score(X_test_unsm, y_test)))

        plot_ROC(y_test, smoted_predict_prob, unsmoted_predict, smoted_predict_bin, unsmoted_predict_bin)

    return X_resampled, y_resampled, X_test, y_test

    # df2 = pd.DataFrame(X_resampled,columns=df.columns.values)
    # df3 = pd.DataFrame(y_resampled,columns=['simple_journal'])
    # df2 = df2.append(df3)

    #write to csv file (takes a long time)
    #print("Writing new Dataset!")
    #df2.to_csv('data_for_student_case_SMOTE2.csv')
    #print("Writing Dataset Finished!")

    #check the number of data points for each lable
    # print(df2.groupby(['simple_journal']).size())

def parse_arguments():
    parser = argparse.ArgumentParser()

    # General commands
    # parser.add_argument('plot', help='Plot', action='store_true')
    parser.add_argument('cls', help='Classifier type', action='store', choices=['NB', 'MLP', 'DT', "RF", 'SVM', 'LR'], default='default')

    args = parser.parse_args()
    return args


def main():

    df = pd.read_csv('data_for_student_case.csv', index_col=0)  # dataframe
    df = preprocessing(df)

    args = parse_arguments()

    # convert date to unix time
    # plotting.plots(df)
    # plotting.statistics(df)
    # print(df.columns.values)

    # plotting.plot_time_diff(df)
    # plotting.plot_daily_freq(df)
    # plotting.plot_amount_ave_diff(df)
    # plotting.plot_cards_per_mail(df)
    # plotting.plot_cards_per_ip(df)

    if args.cls is not None:
        smote(df, 2/3, args.cls, toPlot=True)



main()