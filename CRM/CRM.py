################################################

# Imports

import datetime as dt
import pandas as pd
import pymysql
from sqlalchemy import create_engine
from sklearn.preprocessing import MinMaxScaler
from lifetimes import BetaGeoFitter
from lifetimes import GammaGammaFitter

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 50)
pd.set_option('display.float_format', lambda x: '%.5f' % x)


# Data

data = pd.read_excel('online_retail_II.xlsx', sheet_name="Year 2010-2011")

df = data.copy()

#####################################
# Data Preperation
#####################################

def check_df(dataframe):
    # Data information function

    print("################# Shape #####################")
    print(dataframe.shape)
    print("################# Types ####################")
    print(dataframe.dtypes)
    print("################## Head ####################")
    print(dataframe.head(3))
    print("################## NA ####################")
    print(dataframe.isnull().sum())
    print("################## Quantiles ####################")
    print(dataframe.quantile([0, 0.05, 0.50, 0.95, 0.99, 1]).T)


def outlier_thresholds(dataframe, variable):
    quartile1 = dataframe[variable].quantile(0.01)
    quartile3 = dataframe[variable].quantile(0.99)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit,up_limit

def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit

# thresholds functions

def crm_data_prep(dataframe):
    # data preprocessing function.
    dataframe.dropna(axis=0, inplace=True)
    dataframe = dataframe[~dataframe["Invoice"].str.contains("C", na=False)]
    dataframe = dataframe[dataframe["Quantity"] > 0]
    replace_with_thresholds(dataframe, "Quantity")
    replace_with_thresholds(dataframe, "Price")
    dataframe["TotalPrice"] = dataframe["Quantity"] * dataframe["Price"]
    return dataframe

check_df(df)
df_prep = crm_data_prep(df)
check_df(df_prep)


#####################################
# RFM Segments
#####################################

def create_rfm(dataframe):
    today_date = dt.datetime(2011, 12, 11)
    rfm = dataframe.groupby('Customer ID').agg({'InvoiceDate': lambda date: (today_date - date.max()).days,
                                                'Invoice': lambda num: num.nunique(),
                                                'TotalPrice': lambda price: price.sum()})
    rfm.columns = ['recency', 'frequency', 'monetary']
    rfm = rfm[(rfm['monetary'] > 0)]

    rfm["recency_score"] = pd.qcut(rfm['recency'], 5, labels=[5, 4, 3, 2, 1])
    rfm["frequency_score"] = pd.qcut(rfm["frequency"].rank(method="first"), 5, labels=[1, 2, 3, 4, 5])

    rfm['rfm_segment'] = rfm['recency_score'].astype(str) + rfm['frequency_score'].astype(str)

    seg_map = {
        r'[1-2][1-2]': 'hibernating',
        r'[1-2][3-4]': 'at_risk',
        r'[1-2]5': 'cant_loose',
        r'3[1-2]': 'about_to_sleep',
        r'33': 'need_attention',
        r'[3-4][4-5]': 'loyal_customers',
        r'41': 'promising',
        r'51': 'new_customers',
        r'[4-5][2-3]': 'potential_loyalists',
        r'5[4-5]': 'champions'
    }

    rfm['rfm_segment'] = rfm['rfm_segment'].replace(seg_map, regex=True)
    rfm = rfm[["recency", "frequency", "monetary", "rfm_segment"]]
    return rfm

rfm = create_rfm(df_prep) # create_rfm function used with df_prep and sent new df.
rfm.head()

#####################################
# CLTV Calculated
#####################################


def create_cltv_c(dataframe):
    dataframe['avg_order_value'] = dataframe['monetary'] / dataframe['frequency']
    dataframe['purchase_frequency'] = dataframe['frequency'] / dataframe.shape[0]

    repeat_rate = dataframe[dataframe.frequency > 1].shape[0] / dataframe.shape[0]
    churn_rate = 1 - repeat_rate

    dataframe['profit_margin'] = dataframe['monetary'] * 0.05
    dataframe['cv'] = (dataframe['avg_order_value'] * dataframe['purchase_frequency'])
    dataframe['cltv'] = (dataframe['cv'] / churn_rate) * dataframe['profit_margin']

    scaler = MinMaxScaler(feature_range=(1, 100))
    scaler.fit(dataframe[["cltv"]])
    dataframe["cltv_c"] = scaler.transform(dataframe[["cltv"]])
    dataframe["cltv_c_segment"] = pd.qcut(dataframe["cltv_c"], 3, labels=["C", "B", "A"])

    dataframe = dataframe[["recency", "frequency", "monetary", "rfm_segment", "cltv_c", "cltv_c_segment"]]
    return dataframe


check_df(rfm) # rfm dataframe checked.

rfm_cltv = create_cltv_c(rfm) # create_cltv function used with rfm dataframe and sent new dataframe.
check_df(rfm_cltv) # new dataframe checked.


##########################################
# CLTV Prediction
##########################################

def create_cltv_p(dataframe):
    today_date = dt.datetime(2011, 12, 11)

    rfm = dataframe.groupby('Customer ID').agg({'InvoiceDate': [lambda date: (date.max() - date.min()).days,
                                                                lambda date: (today_date - date.min()).days],
                                                'Invoice': lambda num: num.nunique(),
                                                'TotalPrice': lambda price: price.sum()})
    rfm.columns = rfm.columns.droplevel(0)

    rfm.columns = ['recency_cltv_p', 'T', 'frequency', 'monetary']
    rfm['monetary'] = rfm['monetary'] / rfm['frequency']

    rfm.rename(columns = {'monetary': 'monetary_avg'}, inplace =True)

    rfm["recency_weekly_cltv_p"] = rfm['recency_cltv_p'] / 7
    rfm['T_weekly'] = rfm['T'] / 7

    rfm = rfm[rfm['monetary_avg'] > 0]
    rfm = rfm[(rfm['frequency'] > 1)]
    rfm['frequency'] = rfm['frequency'].astype(int)

    #BGNBD
    bgf = BetaGeoFitter(penalizer_coef=0.01)
    bgf.fit(rfm['frequency'],
            rfm['recency_weekly_cltv_p'],
            rfm['T_weekly'])

    rfm["exp_sales_1_month"] = bgf.predict(4, rfm['frequency'], rfm['recency_weekly_cltv_p'], rfm['T_weekly'])
    rfm["exp_sales_3_month"] = bgf.predict(12, rfm['frequency'], rfm['recency_weekly_cltv_p'], rfm['T_weekly'])

    #Gamma Gamma
    ggf = GammaGammaFitter(penalizer_coef=0.01)
    ggf.fit(rfm['frequency'], rfm['monetary_avg'])
    rfm["expected_average_profit"] = ggf.conditional_expected_average_profit(rfm['frequency'],
                                                                             rfm['monetary_avg'])

    cltv = ggf.customer_lifetime_value(bgf, rfm['frequency'], rfm['recency_weekly_cltv_p'],
                                       rfm['T_weekly'], rfm['monetary_avg'], time=6,
                                       freq='W', discount_rate=0.01)

    rfm["cltv_p"] = cltv

    scaler = MinMaxScaler(feature_range=(1,100))
    scaler.fit(rfm[["cltv_p"]])
    rfm["cltv_p"] = scaler.transform(rfm[["cltv_p"]])

    rfm["cltv_p_segment"] = pd.qcut(rfm["cltv_p"], 3, labels=["C", "B", "A"])

    rfm = rfm[["recency_cltv_p", "T", "monetary_avg", "recency_weekly_cltv_p", "T_weekly",
               "exp_sales_1_month", "exp_sales_3_month", "expected_average_profit", "cltv_p",
               "cltv_p_segment"]]

    return rfm


rfm_cltv_p = create_cltv_p(df_prep) # create_cltv function used with df_prep and sent new dataframe.

crm_final = rfm_cltv.merge(rfm_cltv_p, on= "Customer ID", how = "left") # rfm_cltv and rfm_cltv_p merged and sent to new dataframe.
check_df(crm_final) # new dataframe checked

crm_final.sort_values(by="monetary_avg", ascending=False).head()
crm_final.sort_values(by="cltv_p", ascending=False).head()

crm_final.to_csv("crm_final.csv")


