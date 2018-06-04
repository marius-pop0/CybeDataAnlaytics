import pandas as pd
import numpy as np
import utils
from sax import SAX

def n_gram_model(n_grams):
    model = dict.fromkeys(n_grams)

    for i in range(len(n_grams)-1):
        if model[n_grams[i]] is None:
            model[n_grams[i]] = {}
        if n_grams[i+1] not in model[n_grams[i]]:
            model[n_grams[i]][n_grams[i+1]] = 0
        model[n_grams[i]][n_grams[i + 1]] += 1

    for key in model:
        if model[key] is None:
            continue
        total_count = float(sum(model[key].values()))
        for w2 in model[key]:
            model[key][w2] /= total_count

    return model

def n_gram_predict(model, n_grams, window, threshold, window_size):
    anomalies = []
    probalilites = []
    true_prob = []
    probalilites.append(0)
    for i in range(len(n_grams)-1):
        if n_grams[i] not in model or n_grams[i+1] not in model[n_grams[i]]:
            prob = 0
        else:
            prob = model[n_grams[i]][n_grams[i+1]]
        if prob < threshold:
            anomalies.append(window[i+1])
            true_prob.append(prob)
            probalilites.append(1)
        else:
            probalilites.append(0)
    for i in range(window_size-1):
        probalilites.append(0)
    return anomalies, probalilites


def main():
    train1_df = pd.read_csv('BATADAL_dataset03.csv', index_col='DATETIME')
    train2_df = pd.read_csv('BATADAL_dataset04.csv', index_col=0)
    train1_df.index = pd.to_datetime(train1_df.index, dayfirst=True)
    train2_df['n_gram'] = np.zeros(len(train2_df))
    for col in ['L_T1',  'F_PU11',  'S_PU6']:
        window_size = 10
        word_size = 3
        alphabet_size = 3
        stride = 1
        sax = SAX(wordSize=word_size, alphabetSize=alphabet_size)

        # for column_name in train1_df:
        train_string_rep, train_window_indices = sax.sliding_window(train1_df[col].values, cover=window_size,
                                                                    stride=stride)
        train_string_rep2, train_window_indices2 = sax.sliding_window(train2_df[col].values, cover=window_size,
                                                                      stride=stride)

        threshold = 1e-6

        model = n_gram_model(train_string_rep)
        anomalies, probabilities = n_gram_predict(model, train_string_rep2, train_window_indices2, threshold, window_size)
        print('window: {}, word: {}, alphabet: {}, threshold: {}'.format(window_size, word_size, alphabet_size, threshold))

        train2_df['ATT_FLAG_anom'] = np.where(train2_df['ATT_FLAG'] == 1, 100, 0)
        train2_df['n_gram'] += probabilities

    train2_df['n_gram'] =  np.where(train2_df['n_gram'] > 0, 1, 0)

    train2_df['diff'] = train2_df['ATT_FLAG_anom'] - train2_df['n_gram']

    arr = train2_df['diff'].value_counts()

    TTD = utils.TDD_metric(train2_df, probabilities)
    TP = arr[99]
    FP = arr[-1]
    TN = arr[0]
    FN = arr[100]
    S_CM = utils.S_cm(TP, FP, TN, FN)

    accuracy = (TP+TN)/len(train2_df)
    precision = TP/(TP+FP)

    print('accuracy: {}, precision: {}'.format(accuracy, precision))
    print('TDD: {}'.format(TTD))
    print('S_cm: {}'.format(S_CM))
    print('Ranked: {}'.format(0.5 * TTD + 0.5 * S_CM))


main()