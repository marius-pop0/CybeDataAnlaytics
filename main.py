import csv
import matplotlib.pyplot as plt
import numpy as np;
import seaborn as sns
from sklearn.datasets import make_classification

fraud = []
test = []
with open("data_for_student_case.csv") as csv_file:
    transactions = csv.DictReader(csv_file)
    for row in transactions:
        #FORMAT: txid,bookingdate,issuercountrycode,txvariantcode,bin,amount,currencycode,shoppercountrycode,shopperinteraction,simple_journal,cardverificationcodesupplied,cvcresponsecode,creationdate,accountcode,mail_id,ip_id,card_id
        fraud.append(row)

        if row['simple_journal'] == 'Settled':
            test.append([float(row['cvcresponsecode']), 0, float(row['amount'])])
        elif row['simple_journal'] == 'Chargeback':
            test.append([float(row['cvcresponsecode']), 1, float(row['amount'])])
        else:
            pass


print(test)
#fraudHeat = sns.load_dataset(test)
#fraudHeat = fraudHeat.pivot('card_id','simple_journal','amount')
ax = sns.heatmap(test)
plt.show()

#make_classification()