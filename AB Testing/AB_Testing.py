############################################################

"""
This project deals with the differences between Maximum bidding and average bidding.

"""

# imports

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.stats.api as sms
from scipy.stats import shapiro
from scipy import stats


# configs
pd.set_option("display.max_columns", None)
pd.set_option("display.float_format", lambda x: '%.5f' %x)

# Data
data_testing = pd.read_excel("ab_testing_data.xlsx", sheet_name="Test Group")
data_control = pd.read_excel("ab_testing_data.xlsx", sheet_name="Control Group")

df_testing_group = data_testing.copy()
df_control_group = data_control.copy()

df_testing_group.head()
df_control_group.head()

df_control_group.shape
df_control_group.shape

# na checked
df_testing_group.isnull().sum()
df_control_group.isnull().sum()

# Confidence Interval

# Testing group
sms.DescrStatsW(df_testing_group["Purchase"]).tconfint_mean()

# Control group
sms.DescrStatsW(df_control_group["Purchase"]).tconfint_mean()

############################
# Testing of Assumptions
############################

# Assumptions of normality

# H0: Normality assumption is provide.
# H1: Normality assumption isn't provided.

test_stat, pvalue = shapiro(df_testing_group["Purchase"])
print('Test statistics = %.4f, p-value = %.4f' % (test_stat, pvalue))
# Test statistics = 0.9589, p-value = 0.1541

test_stat, pvalue = shapiro(df_control_group["Purchase"])
print('Test statistics = %.4f, p-value = %.4f' % (test_stat, pvalue))

#Test statistics = 0.9773, p-value = 0.5891

# p value > 0.05
# H0 isn't rejected.


# Assumption of Homogeneity of Variance

# H0: Variances are homogeneous
# H1: Variances aren't homogeneous


test_stat, pvalue = stats.levene(df_testing_group["Purchase"],
                                        df_control_group["Purchase"])
print('Test statistics = %.4f, p-değeri = %.4f' % (test_stat, pvalue))

# Test statistics = 2.6393, p-value = 0.1083

# p-value > 0.05
# H0 isn't rejected.

############################
# Hypothesis
############################

# H0: There is no statistically significant difference between the maximum bidding and average bidding.
# H1: There is a statistically significant difference between the maximum bidding and average bidding.


test_stat, pvalue = stats.ttest_ind(df_testing_group["Purchase"],
                                        df_control_group["Purchase"],
                                        equal_var = True)
print('Test statistics = %.4f, p-değeri = %.4f' % (test_stat, pvalue))

# Test statistics = 0.9416, p-value = 0.3493

# p value > 0.05
# H0 isn't rejected.




