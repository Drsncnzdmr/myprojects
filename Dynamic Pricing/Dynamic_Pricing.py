############################################################

# Dynamic Pricing

# Imports

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.stats.api as sms
from scipy.stats import shapiro
from scipy import stats
import statsmodels.stats.api as sms
import itertools

# Config
pd.set_option("display.max_columns", None)
pd.set_option("display.float_format", lambda x: '%.5f' %x)

# Data
data = pd.read_csv("pricing.csv", delimiter= ";")

data.head()
data.shape

# Outliers functions
def outlier_thresholds(dataframe, variable):
    quartile1 = dataframe[variable].quantile(0.05)
    quartile3 = dataframe[variable].quantile(0.95)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit

def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit


replace_with_thresholds(data, "price")

data.shape
data["category_id"].value_counts()

############################
# Testing of Assumptions
############################

# Assumptions of normality

# H0: Normality assumption is provide.
# H1: Normality assumption isn't provided.


for a in data["category_id"].unique():
    test_stat, pvalue = shapiro(data.loc[data["category_id"] == a, "price"])
    if pvalue < 0.05:
        print('Test statistics = %.4f, p-value = %.4f' % (test_stat, pvalue), "H0 is reject")
    else:
        print('Test statistics = %.4f, p-value = %.4f' % (test_stat, pvalue), "H0 isn't reject")

# H0 REJECT! Mann–Whitney U test will be used.

komb_list = []

for a in itertools.combinations(data["category_id"].unique(), 2):
    komb_list.append(a)

# categories combined.
komb_list


list = []
for i in komb_list:
    test_stat, pvalue = stats.mannwhitneyu(data.loc[data["category_id"] == i[0], "price"],
                                                  data.loc[data["category_id"] == i[1], "price"])
    if pvalue < 0.05:
        list.append((i[0], i[1], "H0 is reject."))
        print(i[0], i[1], 'Test statistics = %.4f, p-value = %.4f' % (test_stat, pvalue), "H0 is reject.")
    else:
        list.append((i[0], i[1], "H0 isn't reject."))
        print(i[0], i[1], 'Test statistics = %.4f, p-value = %.4f' % (test_stat, pvalue), "H0 isn't reject.")


# New dataframe created for H0 hypothesis.

H0_df = pd.DataFrame()
H0_df["Cat1"] = [komb_list[0] for komb_list in list]
H0_df["Cat2"] = [komb_list[1] for komb_list in list]
H0_df["Hypothesis Results"] = [komb_list[2] for komb_list in list]

H0_df.head()


H0_df[H0_df["Hypothesis Results"] == "H0 isn't reject."]


cats = [361254,874521,675201,201436] # This categories chosen because they are similar

total = 0
for i in cats:
    total += data.loc[data["category_id"] == i, "price"].mean()
price = total / 4

prices = []
for i in data["category_id"].unique():
    prices.append((i,
                    data.loc[data["category_id"] == i, "price"].mean(),
                    sms.DescrStatsW(data.loc[data["category_id"] == i, "price"]).tconfint_mean()[0],
                    sms.DescrStatsW(data.loc[data["category_id"] == i, "price"]).tconfint_mean()[1]))

price_sim = pd.DataFrame()
price_sim["category_id"] = [i[0] for i in prices]
price_sim["Mean_Price"] = [i[1] for i in prices]
price_sim["Min_Price"] = [i[2] for i in prices]
price_sim["Max_Price"] = [i[3] for i in prices]

df_count= data["category_id"].value_counts()

df_count = df_count.reset_index()

df_count.columns = ["category_id","count"]

DF = price_sim.merge(df_count,how="inner",on="category_id")

DF["Mean"] = DF["Mean_Price"]*DF["count"]
DF["Min"] = DF["Min_Price"]*DF["count"]
DF["Max"] = DF["Max_Price"]*DF["count"]

print("Mean: ", DF["Mean"].sum())
print("Min: ", DF["Min"].sum())
print("Max: ", DF["Max"].sum())

