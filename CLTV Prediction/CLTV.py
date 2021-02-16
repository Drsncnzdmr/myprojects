########################################

########################################
# DATA PREPARATION
########################################

# Imports
import datetime as dt
import pandas as pd
import matplotlib.pyplot as plt
from lifetimes import BetaGeoFitter
from lifetimes import GammaGammaFitter
from lifetimes.plotting import plot_period_transactions

pd.set_option('display.max_columns', 20)
pd.set_option('display.max_rows', 20)
pd.set_option('display.float_format', lambda x: '%.5f' % x)

# These functions for outliers.
def outlier_thresholds(dataframe, variable):
    quartile1 = dataframe[variable].quantile(0.01)
    quartile3 = dataframe[variable].quantile(0.99)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return up_limit, low_limit


def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe,variable)
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit

# Data
data = pd.read_excel("datasets/online_retail_II.xlsx", sheet_name="Year 2010-2011")
# United Kingdom selected.
data_UK = data[data['Country'] == "United Kingdom"]
df = data_UK.copy()

df.dropna(inplace=True) # na values deleted.

df = df[~df["Invoice"].str.contains("C", na=False)] # In this data C mean is return. Return invoices deleted.
df = df[df["Quantity"] > 0] # Quantity checked.

replace_with_thresholds(df, "Quantity")
replace_with_thresholds(df, "Price")
# Quantity and Price's outliers checked.

df.describe().T # Integer Float variables checked.

df["TotalPrice"] = df["Quantity"] * df["Price"] # Total Price calculated.
df["InvoiceDate"].max()

today_date = dt.datetime(2011,12,11) # 2 days added to the last time.

########################################
# RFM TABLE
########################################

# user-specific dynamic recency
rfm = df.groupby("Customer ID").agg({'InvoiceDate': [lambda date: (date.max() - date.min()).days,
                                                     lambda date: (today_date - date.min()).days],
                                     'Invoice': lambda num: num.nunique(),
                                     'TotalPrice': lambda TotalPrice: TotalPrice.sum()})

rfm.colums = rfm.columns.droplevel(0)


rfm.columns = ['recency_cltv_p', "T", "frequency", "monetary"] # Variable names changed.

rfm["monetary"] = rfm["monetary"] / rfm["frequency"] # simplified monetary average.

rfm.rename(columns= {"monetary": "monetary_avg"}, inplace=True) # columns renamed.

rfm["recency_weekly_p"] = rfm["recency_cltv_p"] / 7 # recency_cltv_p was daily. Turned to weekly.
rfm["T_weekly"] = rfm["T"] / 7 # tenure was daily. Turned to weekly.

rfm = rfm[rfm["monetary_avg"] >0] # checked.

rfm = rfm[(rfm["frequency"] > 1)] # checked.
rfm["frequency"] = rfm["frequency"].astype(int) # frequency turned to int.

########################################
# BGNBD
########################################

# if you haven't lifetime library. You should install with pip install lifetimes.

bgf = BetaGeoFitter(penalizer_coef=0.001) # BGNBD created.
bgf.fit(rfm["frequency"],
        rfm["recency_weekly_p"],
        rfm["T_weekly"]) # BGNBD fitted.

########################################
# Gamma Gamma
########################################

ggf = GammaGammaFitter(penalizer_coef=0.01) # Gamma Gamma created.
ggf.fit(rfm["frequency"], rfm["monetary_avg"]) # Gamma gamma fitted.

# 6 Months CLTV Prediction

cltv_6_months = ggf.customer_lifetime_value(bgf,
                                   rfm['frequency'],
                                   rfm['recency_weekly_p'],
                                   rfm['T_weekly'],
                                   rfm['monetary_avg'],
                                   time= 6,
                                   freq="W",
                                   discount_rate=0.01)


cltv_6_months = cltv_6_months.reset_index() # indexes are broken. Reset_index fixed it.
cltv_6_months.sort_values(by="clv", ascending=False)

# 1 Month CLTV Prediction

cltv_1_month = ggf.customer_lifetime_value(bgf,
                                   rfm['frequency'],
                                   rfm['recency_weekly_p'],
                                   rfm['T_weekly'],
                                   rfm['monetary_avg'],
                                   time=1,
                                   freq="W",
                                   discount_rate=0.01)

cltv_1_month = cltv_1_month.reset_index()
cltv_1_month.sort_values(by="clv", ascending=False).head(10)

# 12 Months CLTV Prediction

cltv_12_months = ggf.customer_lifetime_value(bgf,
                                   rfm['frequency'],
                                   rfm['recency_weekly_p'],
                                   rfm['T_weekly'],
                                   rfm['monetary_avg'],
                                   time=12,
                                   freq="W",
                                   discount_rate=0.01)

cltv_12_months = cltv_12_months.reset_index()
cltv_12_months.sort_values(by="clv", ascending=False).head(10)


# Segmentation with 6 months
# I tried to found first %20 customers.

cltv_6_months_seg = cltv_6_months.copy()
cltv_6_months_seg["Segment"] = pd.qcut(cltv_6_months_seg["clv"], 3, labels=["C","B","A"])

fp20 = cltv_6_months_seg.sort_values(by="clv", ascending=False)

p20 = fp20.shape[0] * 0.2
# First 514 customers are %20.

a = cltv_6_months_seg.sort_values(by="clv", ascending=False) # Customers sent to new dataframe

a["top_flag"] = 0 # New variable created. Default 0.
a["top_flag"].iloc[0:514] = 1 # If customer in %20 top_flag value was 1.





