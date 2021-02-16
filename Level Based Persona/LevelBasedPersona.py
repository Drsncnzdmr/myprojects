#############################################
# PROJE: LEVEL BASED PERSONA DEFINITION, BASIC SEGMENTATION and RULE BASED CLASSIFICATION
#############################################

# Import
import pandas as pd

# Data
users = pd.read_csv("users.csv")
purchases = pd.read_csv("purchases.csv")

data = users.merge(purchases, how= "inner", on= "uid") # Datas merged.

data.head() # For data check.

data.groupby(["country", "device", "gender", "age"]).agg({"price": "sum"}).head() # Price checked for country, device, gender, age.

agg_df = data.groupby(["country", "device", "gender", "age"]).agg({"price": "sum"}).sort_values(by = "price", ascending = False) # new dataframe created with last group by.

agg_df.head() # new dataframe checked.

agg_df.reset_index(inplace= True) # For index error.

agg_df["age"]
agg_df["age_cat"]=pd.cut(agg_df["age"],bins= [0,19,24,31,41,agg_df["age"].max()],labels=["0_18", "19_23", "24_30", "31_40", "41_"+str(agg_df["age"].max())]) # new variable created for categorical age.
agg_df.head()


agg_df["level_based_customers"] = [row[0] + "_" + row[1].upper()+ "_" + row[2] + "_" + row[5] for row in agg_df.values] # the necessary information took and sent to the new variable.
agg_df = agg_df[["level_based_customers","price"]] # new variable and price sent to dataframe.
agg_df


agg_df["level_based_customers"].count()
agg_df = agg_df.groupby("level_based_customers").agg({"price": "mean"})
agg_df = agg_df.reset_index()
agg_df["level_based_customers"].count()


agg_df["segment"] = pd.qcut(agg_df["price"], 4, labels=["D", "C", "B", "A"]) # Segments created with qcut.
agg_df


agg_df.groupby("segment").agg({"price":"mean"}).sort_values("price")

new_user = "TUR_IOS_F_41_75"

print(agg_df[agg_df["level_based_customers"] == new_user]) # Model tested with new user.
