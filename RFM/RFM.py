#########################################

# Imports
import pandas as pd
import datetime as dt

# Some configurations.
pd.set_option('display.max_columns', 20)
pd.set_option('display.max_rows', 30)
pd.set_option('display.flat_format', lambda x: '%.5f' % x)

# Data
df_ = pd.read_excel(r"C:\Users\Dursun Can\Desktop\VB\DSMLBC\datasets\online_retail_II.xlsx", sheet_name= "Year 2010-2011")
df = df_.copy()

# In this data C mean is return. Return invoices deleted.
df = df[~df["Invoice"].str.contains("C", na=False)]

df["TotalPrice"] = df["Quantity"] * df["Price"] # Total price calculated.

# Data Preprocessing

df.isnull().sum() # Total na count.
df.dropna(inplace=True) # na values deleted.

# RFM Metrics

df["InvoiceDate"].max() # last time checked.

today_date = dt.datetime(2011,12,11) # 2 days added to the last time.

rfm = df.groupby("Customer ID").agg({'InvoiceDate': lambda date: (today_date - date.max()).days,
                                     'Invoice': lambda num: len(num),
                                     'TotalPrice': lambda Totalprice: Totalprice.sum()})
# RFM calculated.

rfm

rfm.columns = ["Recency", "Frequency", "Monetary"] # Variables name changed.

rfm = rfm[(rfm["Monetary"]) > 0 & (rfm["Frequency"] > 0)] # Monetary & Frequency checked.

# RFM Scoring

# segments built with qcut.
rfm["RecencyScore"] = pd.qcut(rfm["Recency"], 5, labels=[5, 4, 3, 2, 1])

rfm["FrequencyScore"] = pd.qcut(rfm["Frequency"], 5, labels=[1, 2, 3, 4, 5])

rfm["MonetaryScore"] = pd.qcut(rfm["Monetary"], 5, labels=[1, 2, 3, 4, 5])

rfm["RFM_SCORE"] = (rfm["RecencyScore"].astype(str)+rfm["FrequencyScore"].astype(str)+rfm["MonetaryScore"].astype(str)) # RFM Scores converted to string and gathered. After that sent to new variable.


rfm[rfm["RFM_SCORE"] == "555"].head()


# RFM NAMING

seg_map = {
    r'[1-2][1-2]': 'Hibernating',
    r'[1-2][3-4]': 'At_Risk',
    r'[1-2][5]': 'Cant_Loose_Them',
    r'[3][1-2]': 'About_to_Sleep',
    r'[3][3]' : 'Need_Attention',
    r'[3-4][4-5]': 'Loyal_Customers',
    r'[4][1]' : 'Promising',
    r'[5][1]' : 'New_Customers',
    r'[4-5][2-3]' : 'Potential_Loyalists',
    r'[5][4-5]' : 'Champions'
}


rfm['Segment'] = rfm['RecencyScore'].astype(str) + rfm['FrequencyScore'].astype(str) # RecencyScore and FrequencyScore converted to string and gathered. After that sent to new variable.

rfm['Segment'] = rfm['Segment'].replace(seg_map, regex=True) # RFM naming replaced to Segment with Regex. After that sent to segment.
df[["Customer ID"]].nunique() #


rfm[["Segment", "Recency", "Frequency", "Monetary"]].groupby("Segment").agg(["count", "mean", "min", "max"])  # Count mean min and max checked on groupby for Segment Recency Frequency and Monetary

# Loyal_Customers segment checked.
rfm[rfm["Segment"] == "Loyal_Customers"].head()

# Loyal_Customers segment's indexes took.
rfm[rfm["Segment"] == "Loyal_Customers"].index

excel = pd.DataFrame() # created a new dataframe

excel["Loyal_Customers_ID"] = rfm[rfm["Segment"] == "Loyal_Customers"].index # Loyal_Customers segment's indexes sent to excel dataframe's variable.

excel.to_excel("Loyal_Customers_Customer_ID.xlsx") # excel dataframe's exported.

