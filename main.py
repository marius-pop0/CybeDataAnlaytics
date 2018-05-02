import csv
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification

with open("data_for_student_case.csv") as csv_file:
    transactions = csv.DictReader(csv_file)
    for row in transactions:
        #FORMAT: txid,bookingdate,issuercountrycode,txvariantcode,bin,amount,currencycode,shoppercountrycode,shopperinteraction,simple_journal,cardverificationcodesupplied,cvcresponsecode,creationdate,accountcode,mail_id,ip_id,card_id
        print(row['txvariantcode'], row['amount'])


#make_classification()