import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from imblearn.over_sampling import SMOTE

#FORMAT: txid,bookingdate,issuercountrycode,txvariantcode,bin,amount,currencycode,shoppercountrycode,shopperinteraction,simple_journal,cardverificationcodesupplied,cvcresponsecode,creationdate,accountcode,mail_id,ip_id,card_id


def plots(df):
    df['simple_journal'], labels = pd.factorize(df.simple_journal)
    stats_df = df[['cvcresponsecode', 'simple_journal', 'amount']]
    stats_df = stats_df.groupby(['cvcresponsecode', 'simple_journal'], as_index=False)[
        'amount'].mean()  # ['amount'].agg('sum')
    pivot = stats_df.pivot(index='cvcresponsecode', columns='simple_journal', values='amount')
    print(labels)
    sns.heatmap(pivot)

    stats_df2 = df[['cvcresponsecode', 'simple_journal', 'amount']]
    stats_df2 = stats_df2.loc[(stats_df2['cvcresponsecode'] == 1) & (stats_df2['simple_journal'] == 0)]

    stats_df2.hist(column='amount')

    plt.show()


def smote(df):
    df['card_id'] = df['card_id'].map(lambda x: x.lstrip('card'))
    df['ip_id'] = df['ip_id'].map(lambda x: x.lstrip('ip'))
    df['mail_id'] = df['mail_id'].map(lambda x: x.lstrip('email'))
    df = pd.get_dummies(df, columns=['issuercountrycode', 'txvariantcode', 'currencycode', 'shoppercountrycode',
                                     'shopperinteraction', 'accountcode', 'cardverificationcodesupplied'])

    X = df.drop(['simple_journal', 'creationdate', 'mail_id', 'ip_id', 'card_id'], axis=1).values
    y = df['simple_journal'].values
    print(np.shape(X), np.shape(y))

    # SMOTE Data
    sm = SMOTE(random_state=15, kind='svm')
    X_resampled, y_resampled = sm.fit_sample(X, y)
    print(X_resampled)
    # pd.to_csv(df2)


def main():
    df = pd.read_csv('data_for_student_case.csv', index_col=0)  # dataframe
    df['cvcresponsecode'].apply(lambda x: float(x))
    df['amount'].apply(lambda x: float(x))
    # plots(df)
    smote(df)



main()