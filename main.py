import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from imblearn.over_sampling import SMOTE

#FORMAT: txid,bookingdate,issuercountrycode,txvariantcode,bin,amount,currencycode,shoppercountrycode,shopperinteraction,simple_journal,cardverificationcodesupplied,cvcresponsecode,creationdate,accountcode,mail_id,ip_id,card_id

df = pd.read_csv('data_for_student_case.csv', index_col=0) #dataframe
df['cvcresponsecode'].apply(lambda x: float(x))
df['amount'].apply(lambda x: float(x))
df['simple_journal'], labels = pd.factorize(df.simple_journal)

stats_df = df[['cvcresponsecode', 'simple_journal', 'amount']]
stats_df = stats_df.groupby(['cvcresponsecode', 'simple_journal'], as_index=False)['amount'].mean()#['amount'].agg('sum')
pivot = stats_df.pivot(index='cvcresponsecode', columns='simple_journal', values='amount')
print(labels)
sns.heatmap(pivot)

stats_df2 = df[['cvcresponsecode', 'simple_journal', 'amount']]
stats_df2 = stats_df2.loc[(stats_df2['cvcresponsecode'] == 1) & (stats_df2['simple_journal'] == 0)]

stats_df2.hist(column='amount')

plt.show()


#SMOTE Data

sm = SMOTE(random_state=15)

df2 = sm.fit_sample(df,df.iteritems())

