import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd



df = pd.read_csv('data_for_student_case.csv', index_col=0) #dataframe
df['cvcresponsecode'].apply(lambda x: float(x))
df['amount'].apply(lambda x: float(x))
df['simple_journal'], labels = pd.factorize(df.simple_journal)

stats_df = df[['cvcresponsecode', 'simple_journal', 'amount']]
stats_df = stats_df.groupby(['cvcresponsecode', 'simple_journal'], as_index=False)['amount'].mean()#['amount'].agg('sum')
pivot = stats_df.pivot(index='cvcresponsecode', columns='simple_journal', values='amount')
print(labels)
sns.heatmap(pivot)

plt.show()

#make_classification()

