import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from imblearn.over_sampling import SMOTE
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split


#format: txid,bookingdate,issuercountrycode,txvariantcode,bin,amount,currencycode,shoppercountrycode,shopperinteraction,simple_journal,cardverificationcodesupplied,cvcresponsecode,creationdate,accountcode,mail_id,ip_id,card_id

def preprocessing(df):
    df.drop(df[df['simple_journal'] == 'Refused'].index, inplace=True)
    # df = df[df['shoppercountrycode'] != 'GB']
    df['cvcresponsecode'] = df['cvcresponsecode'].map(lambda x: float(x))
    df['amount'] = df['amount'].map(lambda x: float(x))
    df['card_id'] = df['card_id'].map(lambda x: x[4:])
    df['ip_id'] = df['ip_id'].map(lambda x: x[2:])
    df['mail_id'] = df['mail_id'].map(lambda x: x[5:])
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
    df1 = df.groupby(['ip_id','simple_journal'], as_index=False)['card_id'].size().reset_index(name='freq')
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

def classification(X, y):
    model = GaussianNB()
    model.fit(X, y)
    return model


def smote(df):
    print(df.isnull().any())

    # df['creationdate'] = pd.to_datetime(df['creationdate'], unit='s')
    df = pd.get_dummies(df, columns=['issuercountrycode', 'txvariantcode', 'currencycode', 'shoppercountrycode',
                                     'shopperinteraction', 'accountcode', 'cardverificationcodesupplied'])

    y = df['simple_journal'].values
    df = df.drop(['simple_journal', 'creationdate', 'bookingdate', 'mail_id', 'ip_id', 'card_id'], axis=1)
    X = df.values

    length = np.shape(X)[0]#, np.shape(y))
    train_len = round(length*2/3)
    X_train = X[:train_len, :]
    y_train = y[:train_len]
    X_test = X[train_len:, :]
    y_test = y[train_len:]
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
    print('Size of test data: ' + str(np.shape(X_test)))

    # SMOTE Data
    sm = SMOTE(random_state=15)
    X_resampled, y_resampled = sm.fit_sample(X_train, y_train)

    smoted_model = classification(X_resampled, y_resampled)
    unsmoted_model = classification(X_train, y_train)

    print('smoted:   ' + str(smoted_model.score(X_test, y_test)))
    print('unsmoted: ' + str(unsmoted_model.score(X_test, y_test)))

    # df2 = pd.DataFrame(X_resampled,columns=df.columns.values)
    # df3 = pd.DataFrame(y_resampled,columns=['simple_journal'])
    # df2 = df2.append(df3)

    #write to csv file (takes a long time)
    #print("Writing new Dataset!")
    #df2.to_csv('data_for_student_case_SMOTE2.csv')
    #print("Writing Dataset Finished!")

    #check the number of data points for each lable
    # print(df2.groupby(['simple_journal']).size())

def main():
    df = pd.read_csv('data_for_student_case.csv', index_col=0)  # dataframe
    df = preprocessing(df)
    # convert date to unix time
    # df['creationdate_unix'] = pd.DatetimeIndex(df['creationdate']).astype(np.int64) / 1000000000
    #plots(df)
    # smote(df)
    # statistics(df)
    # print(df.columns.values)

    #plot_time_diff(df)
    # plot_daily_freq(df)
    #plot_amount_ave_diff(df)
    # plot_cards_per_mail(df)
    plot_cards_per_ip(df)


main()