#####################################

######################################
# IMPORTS
######################################
import warnings
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pickle
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split, GridSearchCV
from lightgbm import LGBMRegressor
from data_prep import *
from eda import *
from helpers import *

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter('ignore', category=ConvergenceWarning)

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.float_format',lambda x:'%.3f' % x)

# datas merged.
train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")
df = train.append(test).reset_index(drop=True)
df.head()


######################################
# EDA
######################################

check_df(df) # data checked.

cat_cols, cat_but_car, num_cols, num_but_cat = grab_col_names(df, car_th=10)  # kategorik değişkenler bulundu.

# CATEGORICAL VARIABLE ANALYSIS

for col in cat_cols:
    cat_summary(df,col)

for col in cat_but_car:
    cat_summary(df,col)

for col in num_but_cat:
    cat_summary(df,col)

# NUMERICAL VARIABLE ANALYSIS

df[num_cols].describe([0.05, 0.10, 0.25, 0.50, 0.75, 0.80, 0.90, 0.95, 0.99]).T

for col in num_cols:
    num_summary(df, col, plot=True)

# CORRELATION ANALYSIS
corr_matrix = df.corr()
sns.clustermap(corr_matrix, annot = True, figsize=(20,15), fmt=".2f" )
plt.title("Correlation Between Features")
plt.show()

threshold = 0.60
filtre = np.abs(corr_matrix["SalePrice"]) > threshold
corr_features = corr_matrix.columns[filtre].tolist()
sns.clustermap(df[corr_features].corr(), annot = True, fmt = ".2f")
plt.title("Correlation Between Features w/ Corr Threshold 0.75)")
plt.show()

# TARGET ANALYSIS

df["SalePrice"].describe([0.05, 0.10, 0.25, 0.50, 0.75, 0.80, 0.90, 0.95, 0.99])

# CORRELATIONS BETWEEN TARGET AND INDEPENDENT VARIABLES
def find_correlation(dataframe, numeric_cols, corr_limit=0.60):
    high_correlations = []
    low_correlations = []
    for col in numeric_cols:
        if col == "SalePrice":
            pass
        else:
            correlation = dataframe[[col, "SalePrice"]].corr().loc[col, "SalePrice"]
            print(col, correlation)
            if abs(correlation) > corr_limit:
                high_correlations.append(col + ": " + str(correlation))
            else:
                low_correlations.append(col + ": " + str(correlation))
    return low_correlations, high_correlations


low_corrs, high_corrs = find_correlation(df, num_cols)

for col in cat_cols:
    stalk(df, col)

######################################
# DATA PREPROCESSING & FEATURE ENGINEERING
######################################

# THIS VARIABLES ARE UNNECESSARY
drop_list = ["PoolQC", "MiscFeature", "Alley", "Fence", "FireplaceQu","Street","Utilities",
             "Condition2", "RoofStyle", "RoofMatl", "Heating", "Electrical", "Functional", "PoolQC", "Fence", "MiscFeature", "SaleType", "SaleCondition"]

df.drop(drop_list, inplace=True, axis=1) # VARIABLES DELETED.

# NA VALUES FILLED

none_cols = ['GarageType','GarageFinish', 'GarageQual', 'GarageCond', 'BsmtQual', 'BsmtCond',
             'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'MasVnrType']

zero_cols = ['BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', 'BsmtFullBath','BsmtHalfBath', 'GarageYrBlt', 'GarageArea', 'GarageCars', 'MasVnrArea']

freq_cols = ['Exterior1st', 'Exterior2nd', 'KitchenQual']

for col in zero_cols:
    df[col].replace(np.nan, 0, inplace=True) # THIS ARE NUMERICAL. FILLED BY 0.

for col in none_cols:
    df[col].replace(np.nan, "None", inplace=True) # THIS ARE ABSENCE. FILLED BY "None".

for col in freq_cols:
    df[col].replace(np.nan, df[col].mode()[0], inplace=True) # THIS ARE CATEGORICAL. FILLED BY MOD.

# Outliers
for col in num_cols:
    print(col, check_outlier(df, col)) # OUTLIER CHECKED.
replace_with_thresholds(df, "SalePrice")
df["MSZoning"] = df.groupby('MSSubClass')['MSZoning'].apply(lambda x: x.fillna(x.mode()[0])) # FILLED BY MODE.
df["LotFrontage"] = df.groupby(['Neighborhood'])['LotFrontage'].apply(lambda x: x.fillna(x.median())) # FILLED BY MEDIAN.

# DATA TYPE FIXED.

df["MSSubClass"] = df["MSSubClass"].astype(str)
df["YrSold"] = df["YrSold"].astype(str)
df["MoSold"] = df["MoSold"].astype(str)

# FEATURE ENGINEERING

df["MSZoning"].value_counts()
df.loc[(df["MSZoning"] == "RM"), "MSZoning"] = "RM"

df["LotShape"].value_counts()
df.loc[(df["LotShape"] == "IR2"), "LotShape"] = "IR1"
df.loc[(df["LotShape"] == "IR3"), "LotShape"] = "IR1"

df["LotShape"].value_counts()
df.loc[(df["LotConfig"] == "Corner"),"LotConfig"] = "FR2"
df.loc[(df["LotConfig"] == "Inside"),"LotConfig"] = "FR2"
df.loc[(df["LotConfig"] == "CulDSac"),"LotConfig"] = "FR3"

df.loc[(df["LandSlope"] == "Mod"),"LandSlope"] = "Sev"
df["LandSlope"].value_counts()

df.loc[(df["Condition1"] == "Feedr"),"Condition1"] = "Artery"
df.loc[(df["Condition1"] == "RRAe"),"Condition1"] = "Artery"
df.loc[(df["Condition1"] == "RRAn"),"Condition1"] = "Norm"
df.loc[(df["Condition1"] == "PosN"),"Condition1"] = "PosA"
df.loc[(df["Condition1"] == "RRNe"), "Condition1"] = "PosA"
df.loc[(df["Condition1"] == "RRNn"), "Condition1"] = "PosA"

df.loc[(df["HouseStyle"] == "1.5Fin"),"HouseStyle"] = "1.5Unf"
df.loc[(df["HouseStyle"] == "2.5Unf"), "HouseStyle"] = "1.5Unf"
df.loc[(df["HouseStyle"] == "SFoyer"), "HouseStyle"] = "1.5Unf"
df.loc[(df["HouseStyle"] == "SLvl"), "HouseStyle"] = "1Story"
df.loc[(df["HouseStyle"] == "2.5Fin"), "HouseStyle"] = "2Story"

df.loc[(df["MasVnrType"] == "BrkCmn"), "MasVnrType"] = "None"

df.loc[(df["GarageType"] == "2Types"), "GarageType"] = "Attchd"
df.loc[(df["GarageType"] == "Basment"), "GarageType"] = "Attchd"

df.loc[(df["GarageType"] == "2Types"), "GarageType"] = "Attchd"
df.loc[(df["GarageType"] == "CarPort"), "GarageType"] = "Detchd"


# ORDINALITY.

ext_map = {'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5}
df["ExterQual"] = df["ExterQual"].map(ext_map).astype('int')
df["ExterCond"] = df["ExterCond"].map(ext_map).astype('int')

bsm_map = {'None': 0, 'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5}
df['BsmtQual'] = df['BsmtQual'].map(bsm_map).astype('int')
df['BsmtCond'] = df['BsmtCond'].map(bsm_map).astype('int')

bsmf_map = {'None': 0,'Unf': 1,'LwQ': 2,'Rec': 3,'BLQ': 4,'ALQ': 5,'GLQ': 6}
df['BsmtFinType1'] = df['BsmtFinType1'].map(bsmf_map).astype('int')
df['BsmtFinType2'] = df['BsmtFinType2'].map(bsmf_map).astype('int')

heat_map = {'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5}
df['HeatingQC'] = df['HeatingQC'].map(heat_map).astype('int')
df['KitchenQual'] = df['KitchenQual'].map(heat_map).astype('int')
df['GarageCond'] = df['GarageCond'].map(bsm_map).astype('int')
df['GarageQual'] = df['GarageQual'].map(bsm_map).astype('int')


# FEATURE EXTRACTIONS

# TOTAL NUMBER OF BATHROOMS.
df["NEW_TotalBath"] = df['BsmtFullBath'] + df['BsmtHalfBath'] * 0.5 + df['FullBath'] + df['HalfBath'] * 0.5

# TOTAL NUMBER OF FLOORS.
df['NEW_TotalSF'] = (df['BsmtFinSF1'] + df['BsmtFinSF2'] + df['1stFlrSF'] + df['2ndFlrSF'])

# FIRST FLOOR AND BASEMENT SQUARE METERS.
df["NEW_SF"] = df["1stFlrSF"] + df["TotalBsmtSF"]

# TOTAL OF GARAGE AREA AND TOTAL SQUARE METERS
df["NEW_SF_G"] = df["NEW_SF"] + df["GarageArea"]

# TOTAL PATIO AREA
df['NEW_TotalPorchSF'] = (df['OpenPorchSF'] + df['3SsnPorch'] + df['EnclosedPorch'] + df['ScreenPorch'] + df['WoodDeckSF'])

# THIS VARIABLE ABOUT HOUSE AGE AND REPAIR
df["NEW_HOUSE_REPAIR"] = df["YearRemodAdd"] + df["YearBuilt"]

# CATEGORICAL HOUSE AGE
df["NEW_HOUSE_REPAIR_CAT"] = pd.qcut(df['NEW_HOUSE_REPAIR'], 5, labels=[1, 2, 3, 4, 5])

# TOTAL SQUARE METERS
df["NEW_TOTAL_M^2"] = df["NEW_SF"] + df["2ndFlrSF"]

# QUALITY VARIABLES
df["NEW_QUAL_COND"] = df['OverallQual'] + df['OverallCond']

df["NEW_BSMT_QUAL_COND"] = df['BsmtQual'] + df['BsmtCond']

df["NEW_EX_QUAL_COND"] = df['ExterQual'] + df['ExterCond']

df["NEW_BSMT_QUAL_COND"] = df['GarageQual'] + df['BsmtCond']

# GOOD HOUSES SELECTED
df['NEW_BEST'] = (df['NEW_QUAL_COND'] >= 14).astype('int')

# HOUSE WITH POOL
df["NEW_HAS_POOL"] = (df['PoolArea'] > 0).astype('int')

# LUX HOUSES
df.loc[(df['Fireplaces'] > 0) & (df['GarageCars'] >= 3), "NEW_LUX"] = 1
df["NEW_LUX"].fillna(0, inplace=True)
df["NEW_LUX"] = df["NEW_LUX"].astype(int)

# TOTAL AREA
df["NEW_AREA"] = df["GrLivArea"] + df["GarageArea"]


df.loc[(df['TotRmsAbvGrd'] >= 7) & (df['GrLivArea'] >= 1800), "NEW_TOTAL_GR"] = 1

ngb = df.groupby("Neighborhood").SalePrice.mean().reset_index()
ngb["CLUSTER_NEIGHBORHOOD"] = pd.cut(df.groupby("Neighborhood").SalePrice.mean().values, 4, labels=range(1, 5))
df = pd.merge(df, ngb.drop(["SalePrice"], axis=1), how="left", on="Neighborhood")

# GARAGE AGE
df["NEW_GARAGEBLTAGE"] = df.GarageYrBlt - df.YearBuilt


######################################
# ENCODING
######################################

# RARE ENCODING

df = rare_encoder(df, 0.01)


# ONE HOT ENCODING
cat_cols, cat_but_car, num_cols, num_but_cat = grab_col_names(df, car_th=10)  # kategorik değişkenler bulundu.
cat_cols_a = cat_cols + cat_but_car
df = one_hot_encoder(df, cat_cols_a, drop_first=True)

# DATA SEPARETED.
train_df = df[df['SalePrice'].notnull()]
test_df = df[df["SalePrice"].isnull()].drop("SalePrice", axis=1)

######################################
# MODELING
######################################

# HOLDOUT

X = train_df.drop(['SalePrice', "Id"], axis=1)
y = np.log1p(train_df['SalePrice'])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=46)


# LGBM MODEL

lgbm_model = LGBMRegressor(random_state=42)
lgbm_model.fit(X_train,y_train)

train_pred = lgbm_model.predict(X_train)
np.sqrt(mean_squared_error(y_train,train_pred))

test_pred = lgbm_model.predict(X_test)
np.sqrt(mean_squared_error(y_test, test_pred))


# MODEL TUNING

lgbm_params = {"learning_rate": [0.1, 0.01],
              "max_depth": [3, 5, 8],
              "n_estimators": [500, 1000],
              "colsample_bytree": [1, 0.8, 0.5]}

lgbm_model = LGBMRegressor()
lgbm_model.get_params()
lgbm_cv_model = GridSearchCV(lgbm_model, lgbm_params, cv= 10, n_jobs=-1, verbose=2).fit(X_train,y_train)

lgbm_cv_model.best_params_

lgbm_tuned = LGBMRegressor(colsample_bytree=0.3,learning_rate=0.1,max_depth=5, n_estimators=200, subsample=0.3, num_leaves=31).fit(X_train, y_train)
lgbm_tuned = LGBMRegressor(**lgbm_cv_model.best_params_).fit(X_train, y_train)
lgbm_tuned_test_pred = lgbm_tuned.predict(X_test)
np.sqrt(mean_squared_error(y_test, lgbm_tuned_test_pred))


#######################################
# Feature Importance
#######################################

def plot_importance(model, features, num=len(X), save=False):
    feature_imp = pd.DataFrame({'Value': model.feature_importances_, 'Feature': features.columns})
    plt.figure(figsize=(10, 10))
    sns.set(font_scale=1)
    sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value",
                                                                     ascending=False)[0:num])
    plt.title('Features')
    plt.tight_layout()
    plt.show()
    if save:
        plt.savefig('importances.png')

# BASE MODEL IMPORTANCE
plot_importance(lgbm_model, X_train, 50)

# TUNED MODEL IMPORTANCE
plot_importance(lgbm_tuned, X_train, 20)


#######################################
# KAGGLE SUBMISSIONS
#######################################

# BASE MODEL KAGGLE SUBMISSION

submission_df = pd.DataFrame()
submission_df['Id'] = test_df["Id"]

y_pred_sub = lgbm_model.predict(test_df.drop("Id", axis=1))
y_pred_sub = np.expm1(y_pred_sub)

submission_df['SalePrice'] = y_pred_sub

submission_df.head()

submission_df.to_csv('lgbm_base_submission.csv', index=False)


# TUNED MODEL KAGGLE SUBMISSION

kaggle_tuned_test_df = test_df.drop(["Id"], axis=1)
kaggle_pred = lgbm_tuned.predict(kaggle_tuned_test_df)
kaggle_pred = np.expm1(kaggle_pred)

kaggle_submission = pd.DataFrame()
kaggle_submission["Id"] = test_df["Id"]
kaggle_submission["SalePrice"] = kaggle_pred

kaggle_submission.head()

kaggle_submission.to_csv("last.csv", index=False)

#######################################
# EVALUATION
#######################################

# BASE MODEL LGBM RESULTS
# TRAIN ERROR: 0.04066802822909248
# TEST ERROR: 0.11961793842761954
# TUNED TEST ERROR: 0.11134531256917768
# BASE KAGGLE: 0.14116
# TUNED KAGGLE: 0.13257
