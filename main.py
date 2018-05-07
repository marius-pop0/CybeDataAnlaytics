import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from imblearn.over_sampling import SMOTE

#format: txid,bookingdate,issuercountrycode,txvariantcode,bin,amount,currencycode,shoppercountrycode,shopperinteraction,simple_journal,cardverificationcodesupplied,cvcresponsecode,creationdate,accountcode,mail_id,ip_id,card_id

def preprocessing(df):
    df = df[df['simple_journal'] != 'Refused']
    # df = df[df['shoppercountrycode'] != 'GB']
    df['cvcresponsecode'].apply(lambda x: float(x))
    df['amount'].apply(lambda x: float(x))
    df['card_id'] = df['card_id'].map(lambda x: x.lstrip('card'))
    df['ip_id'] = df['ip_id'].map(lambda x: x.lstrip('ip'))
    df['mail_id'] = df['mail_id'].map(lambda x: x.lstrip('email'))
    return df

def statistics(df):
    df1 = df.groupby(['shoppercountrycode', 'currencycode', 'simple_journal']).size().reset_index(name='freq').sort_values(by=['freq'], ascending=False).head()
    df2 = df[(df['shoppercountrycode'] == 'AU') & (df['currencycode'] == 'AUD')].groupby(['txvariantcode', 'simple_journal']).size().reset_index(name='freq').sort_values(by=['txvariantcode', 'simple_journal', 'freq'], ascending=False)
    df3 = df[df['simple_journal'] == 'Chargeback'].groupby(['shoppercountrycode', 'currencycode']).size().reset_index(name='freq').sort_values(by=['freq'], ascending=False)
    df4 = df[(df['simple_journal'] == 'Chargeback') & (df['shoppercountrycode']=='AU')].groupby(['card_id', ]).size().reset_index(name='freq').sort_values(by=['freq'], ascending=False).head(10)
    df5 = df[df['simple_journal'] == 'Settled'].groupby(['card_id']).size().reset_index(name='freq').sort_values(by=['freq'], ascending=False).head(20)
    df6 = df[df['card_id'] == 'card182921'].groupby(['simple_journal', 'amount', 'creationdate', 'ip_id', 'currencycode', 'shopperinteraction', 'shoppercountrycode']).size().reset_index(name='freq').sort_values(by=['creationdate'])#.head(10)


    # print(df1)
    print(df1)
    # print(df6)

def plot_time_diff(df):
    df = df[(df['shoppercountrycode'] == 'AU') & (df['currencycode'] == 'AUD')]

    # df['AUD_currency'] = df[['currencycode', 'amount']].apply(lambda x: currencyconverter.convert_currency_from_aud(x), axis=1)
    df['date'] = pd.to_datetime(df['creationdate'])
    df['diff_time'] = df.sort_values(['card_id', 'creationdate']).groupby('card_id')['date'].diff()
    print(df.sort_values(['card_id', 'date']).head(20))
    time = pd.DatetimeIndex(df['diff_time'])
    df['diff_time_min'] = time.hour * 60 + time.minute + 1  # df['diff_time'].str.split(':').apply(lambda x: int(x[0]) * 60 + int(x[1]))
    df['diff_time_min'] = df['diff_time_min'].fillna(0)
    # df.sort_values(['card_id', 'date']).head(20)

    df2 = df[df['simple_journal'] == 'Chargeback']
    df3 = df[df['simple_journal'] == 'Settled']
    s = plt.scatter(df3['amount'], df3['diff_time_min'], s=4)
    f = plt.scatter(df2['amount'], df2['diff_time_min'], s=4)
    plt.legend((f, s), ('Fraud', 'Legitimate'))
    plt.xlabel('transaction amount')
    plt.ylabel('time delta minutes')

    plt.show()

def plot_daily_freq(df):
    df = df[(df['shoppercountrycode'] == 'AU') & (df['currencycode'] == 'AUD')]

    df['day'] = pd.to_datetime(df['creationdate']).dt.date
    # print(df.head())
    df['freq'] = df.sort_values(['creationdate']).groupby(['card_id', 'day'])['creationdate_unix'].rank(
        method='first').astype(int)
    df['freq'] = df['freq'] - 1

    df_uu = df[df['freq']>6]
    print(df_uu.head())

    df3 = df[df['simple_journal'] == 'Settled']
    df2 = df[df['simple_journal'] == 'Chargeback']
    s = plt.scatter(df3['amount'], df3['freq'], s=4)
    f = plt.scatter(df2['amount'], df2['freq'], s=4)
    plt.legend((f, s), ('Fraud', 'Legitimate'))
    plt.xlabel('transaction amount')
    plt.ylabel('daily frequency of so far seen transactions')

    plt.show()

def plot_amount_ave_diff(df):
    #Y
    df['date'] = pd.to_datetime(df['creationdate'])
    df['ave_amount'] = df.sort_values(['creationdate']).groupby(['card_id'])['amount'].apply(pd.expanding_mean)
    df['diff_from_ave_amount'] = df['amount'] - df['ave_amount']
    #X
    df['date'] = pd.to_datetime(df['creationdate'])
    df['diff_time'] = df.sort_values(['card_id', 'creationdate']).groupby('card_id')['date'].diff()
    print(df.sort_values(['card_id', 'date']).head(20))
    time = pd.DatetimeIndex(df['diff_time'])
    df['diff_time_min'] = time.hour * 60 + time.minute + 1  # df['diff_time'].str.split(':').apply(lambda x: int(x[0]) * 60 + int(x[1]))
    df['diff_time_min'] = df['diff_time_min'].fillna(0)
    print(df.head(20))

    #plot
    df2 = df[df['simple_journal'] == 'Chargeback']
    df3 = df[df['simple_journal'] == 'Settled']
    s = plt.scatter(df3['diff_time_min'], df3['diff_from_ave_amount'], s=4)
    f = plt.scatter(df2['diff_time_min'], df2['diff_from_ave_amount'], s=4)
    plt.legend((f, s), ('Fraud', 'Legitimate'))
    plt.xlabel('time delta minutes')
    plt.ylabel('amount delta from average')

    plt.show()


def plots(df):
    df['simple_journal'], labels = pd.factorize(df.simple_journal)

    stats_df = df[['creationdate','amount','ip_id','card_id']]
    ##change to  any intersting card number
    stats_df = stats_df[stats_df['card_id'] == 'card182921'].groupby(['creationdate', 'ip_id'], as_index=False)['amount'].mean()
    pivot = stats_df.pivot(index='ip_id', columns='creationdate', values='amount')
    sns.heatmap(pivot)

    # stats_df = df[['cvcresponsecode', 'simple_journal', 'amount']]
    # stats_df = stats_df.groupby(['cvcresponsecode', 'simple_journal'], as_index=false)[
    #     'amount'].mean()  # ['amount'].agg('sum')
    #
    # print(stats_df)
    # pivot = stats_df.pivot(index='cvcresponsecode', columns='simple_journal', values='amount')
    # print(labels)
    # sns.heatmap(pivot)
    #
    # stats_df2 = df[['cvcresponsecode', 'simple_journal', 'amount']]
    # stats_df22 = stats_df2.loc[(stats_df2['cvcresponsecode'] == 0) & (stats_df2['simple_journal'] == 2)]
    # stats_df3 = stats_df2.loc[(stats_df2['cvcresponsecode'] == 0) & (stats_df2['simple_journal'] == 0)]
    # stats_df22.hist(column='amount', range=(df['amount'].min(), df['amount'].max()))
    # stats_df3.hist(column='amount', range=(df['amount'].min(), df['amount'].max()))

    plt.show()


def smote(df):
    print(df.isnull().any())
    # df['creationdate'] = pd.to_datetime(df['creationdate'], unit='s')
    df = pd.get_dummies(df, columns=['issuercountrycode', 'txvariantcode', 'currencycode', 'shoppercountrycode',
                                     'shopperinteraction', 'accountcode', 'cardverificationcodesupplied'])

    y = df['simple_journal'].values
    df = df.drop(['simple_journal','creationdate','bookingdate', 'mail_id', 'ip_id', 'card_id'], axis=1)
    X = df.values

    print(np.shape(X), np.shape(y))


    # SMOTE Data
    sm = SMOTE(random_state=15)
    X_resampled, y_resampled = sm.fit_sample(X, y)

    df2 = pd.DataFrame(X_resampled,columns=df.columns.values)
    df3 = pd.DataFrame(y_resampled,columns=['simple_journal'])
    df2 = df2.append(df3)

    #write to csv file (takes a long time)
    #print("Writing new Dataset!")
    #df2.to_csv('data_for_student_case_SMOTE2.csv')
    #print("Writing Dataset Finished!")

    #check the number of data points for each lable
    print(df2.groupby(['simple_journal']).size())

def main():
    df = pd.read_csv('data_for_student_case.csv', index_col=0)  # dataframe
    df = preprocessing(df)
    # convert date to unix time
    df['creationdate_unix'] = pd.DatetimeIndex(df['creationdate']).astype(np.int64) / 1000000000
    #plots(df)
    smote(df)
    statistics(df)
    # print(df.columns.values)

    #plot_time_diff(df)
    plot_daily_freq(df)
    #plot_amount_ave_diff(df)



main()