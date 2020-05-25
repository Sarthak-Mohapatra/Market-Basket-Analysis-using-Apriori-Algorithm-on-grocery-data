### Market Basket Analysis on Groceries Data using Apriori Algorithm.
##
### Importing the required libraries.
##

import numpy                                                  as np
import pandas                                                 as pd
import matplotlib                                             as mp
from   matplotlib                    import pyplot            as plt
from   mlxtend.frequent_patterns     import apriori           as ap
from   mlxtend.frequent_patterns     import association_rules as ar

##
## Let's import the dataset from local system.
##

csv_path            = 'D:/Datasets/market-basket-optimization/items_purchased_data.csv'
items_data_df       = pd.read_csv(csv_path, header=None)
print(items_data_df.head())

##
## Majority of the records have Na or NaN. Let's analyze how many unique values we have.
##

unique_items        = (items_data_df[0].unique())
print(unique_items)

##
## Based on the unique items we got, we will perform one-hot-encoding.
## The dataset will be converted to a format that has just 0s or 1s. It can also have True or False.
## Custom One Hot Encoding
##
## Python code for one hot encoding
##

encoded_vals        = []
for i, rows in items_data_df.iterrows():
	labels          = {}
	uncommons       = list(set(unique_items) - set(rows))
	commons         = list(set(unique_items).intersection(rows))
	for uc in uncommons:
		labels[uc]  = 0
	for com in commons:
		labels[com] = 1
	encoded_vals.append(labels)
encoded_vals[0]

encode_df           = pd.DataFrame(encoded_vals)

print(encode_df.head())

##
## Defining the apriori algorithm.
##

freq_items          = ap(encode_df, min_support=0.0085, use_colnames=True, verbose=1, low_memory=False)
print(freq_items.head())

##
## Defining the association rules algorithms to match and find similar items together based on confidence.
##

assocn_rules_conf   = ar(freq_items, metric="confidence", min_threshold=0.25)
print(assocn_rules_conf)

##
## Defining the association rules algorithms to match and find similar items together based on support.
##
assocn_rules_supp   = ar(freq_items, metric="support", min_threshold=0.005)
print(assocn_rules_supp)

##
##
## Plotting the scatter plot of Confidence Vs Support
##

plt.scatter(assocn_rules_conf['support'], assocn_rules_conf['confidence'], alpha=0.5)
plt.xlabel('support')
plt.ylabel('confidence')
plt.title('Support vs Confidence')
plt.show()

##
## Plotting the scatter plot of Lift Vs Support
##

plt.scatter(assocn_rules_conf['support'], assocn_rules_conf['lift'], alpha=0.5)
plt.xlabel('support')
plt.ylabel('lift')
plt.title('Support vs Lift')
plt.show()

##
## Writing the generated rules based on confidence onto a excel file for business use.
##

w1 = pd.ExcelWriter('D:\Generated_Rules_conf.xlsx')
assocn_rules_conf.to_excel(w1, 'Rules')
w1.save() 

##
## Writing the generated rules based on support onto a excel file for business use.
##

w2 = pd.ExcelWriter('D:\Generated_Rules_supp.xlsx')
assocn_rules_supp.to_excel(w2, 'Rules')
w2.save()
