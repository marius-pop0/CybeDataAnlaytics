import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

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

def time_diff(df):
    df['date'] = pd.to_datetime(df['creationdate'])
    df['diff_time'] = df.sort_values(['card_id', 'creationdate']).groupby('card_id')['date'].diff()
    print(df.sort_values(['card_id', 'date']).head(20))
    time = pd.DatetimeIndex(df['diff_time'])
    df['diff_time_min'] = time.hour * 60 + time.minute + 1  # df['diff_time'].str.split(':').apply(lambda x: int(x[0]) * 60 + int(x[1]))
    df['diff_time_min'] = df['diff_time_min'].fillna(0)
    return df

def plot_time_diff(df):
    df = df[(df['shoppercountrycode'] == 'AU') & (df['currencycode'] == 'AUD')]
    df = time_diff(df)

    df2 = df[df['simple_journal'] == 'Chargeback']
    df3 = df[df['simple_journal'] == 'Settled']
    s = plt.scatter(df3['amount'], df3['diff_time_min'], s=4)
    f = plt.scatter(df2['amount'], df2['diff_time_min'], s=4)
    plt.legend((f, s), ('Fraud', 'Legitimate'))
    plt.xlabel('transaction amount')
    plt.ylabel('time delta minutes')

    plt.show()

def plot_daily_freq(df):
    # df = df[(df['shoppercountrycode'] == 'AU') & (df['currencycode'] == 'AUD')]

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

def plot_cards_per_ip(df):
    df.dropna(inplace=True)
    df1 = df.groupby(['ip_id', 'simple_journal'], as_index=False)['card_id'].size().reset_index(name='freq')
    df1 = df1[df1['ip_id']!= 'NA']
    df1['ip_id'] = df1['ip_id'].map(lambda x: float(x))
    # df1 = df1[df1['freq'] > 1]
    print(df1)
    df2 = df1[df1['simple_journal'] == 'Chargeback']
    df3 = df1[df1['simple_journal'] == 'Settled']
    s = plt.scatter(df3['ip_id'], df3['freq'], s=8)
    f = plt.scatter(df2['ip_id'], df2['freq'], s=8)
    plt.legend((f, s), ('Fraud', 'Legitimate'))
    plt.xlabel('IP ID')
    plt.ylabel('Frequency of IP used')
    plt.show()

def plot_cards_per_mail(df):
    # df.dropna(axis=0, how='any', inplace=True)
    df1 = df.groupby(['mail_id', 'simple_journal'], as_index=False)['card_id'].size().reset_index(name='freq')
    print(df1[df1['mail_id']=='NA'].head())
    df1.dropna(axis=0, how='any', inplace=True)
    df1= df1[df1['mail_id']!= 'NA']
    df1['mail_id'] = df1['mail_id'].map(lambda x: float(x))
    df2 = df1[df1['simple_journal'] == 'Chargeback']
    df3 = df1[df1['simple_journal'] == 'Settled']
    print(df3[df3['freq']>1].head())
    s = plt.scatter(df3['mail_id'], df3['freq'], s=8)
    f = plt.scatter(df2['mail_id'], df2['freq'], s=8)
    plt.legend((f, s), ('Fraud', 'Legitimate'))
    plt.xlabel('E-mail')
    plt.ylabel('Frequency of e-mail used')
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