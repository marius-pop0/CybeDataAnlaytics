import pandas as pd

def TDD_metric(train2_df, y_pred):
    counter = 0
    found_c = 0
    bit = 0
    counter_list = []
    for rindex, row in train2_df.iterrows():
        if row['ATT_FLAG'] == 1:
            if bit == 0:
                counter = 0
                bit = 1
                found_c += 1
            counter += 1
        else:
            if bit == 1:
                for c in range(counter):
                    counter_list.append((c + 1) / counter)
                bit = 0
            counter_list.append(0)

    train2_df['TDD'] = counter_list
    train2_df['y_pred'] = y_pred

    counter = 0
    found_c = 0

    arr = train2_df[['ATT_FLAG', 'TDD']].values
    for i in range(len(arr)):  # rindex, row in train2_df.iterrows():
        if arr[i][0] == 1:
            if y_pred[i] == 1 and bit == 0:
                #                 print(arr[i][1])
                counter += arr[i][1]
                bit = 1
                found_c += 1
        else:
            bit = 0
            #     print('A')
    #     print(counter, found_c)
    S_TDD = 1 - counter / found_c
    return S_TDD


def S_cm(TP, FP, TN, FN):
    TPR = TP/(TP + FN)
    TNR = TN/(FP + TN)
    return (TPR + TNR)/2