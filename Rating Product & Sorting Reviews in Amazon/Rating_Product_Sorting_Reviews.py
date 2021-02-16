###################################################
# Rating Product & Sorting Reviews in Amazon
###################################################

# Imports
import pandas as pd
import math
import scipy.stats as st

# Configs
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.float_format', lambda x: '%.5f' % x)

# Data
data = pd.read_csv("datasets/df_sub.csv") # This data have just 1 product.
df = data.copy()

df.head(20)
df.shape

df["overall"].value_counts() # Overall checked

df["overall"].mean() # Overall mean checked

df['reviewTime'] = pd.to_datetime(df['reviewTime'], dayfirst=True)
df["reviewTime"].max() # Timestamp('2014-12-07 00:00:00')
current_date = pd.to_datetime('2014-12-08 0:0:0') # 1 day added to max date.
df["day_diff"] = (current_date - df['reviewTime']).dt.days


# The time divided into quartiles.
a = df["day_diff"].quantile(0.25)
b = df["day_diff"].quantile(0.50)
c = df["day_diff"].quantile(0.75)

df["day_diff"].value_counts() # day_diff values checked.

# weighted score calculated.
df.loc[df["day_diff"] <= a, "overall"].mean() * 28 / 100 + \
    df.loc[(df["day_diff"] > a) & (df["day_diff"] <= b), "overall"].mean() * 26 / 100 + \
    df.loc[(df["day_diff"] > b) & (df["day_diff"] <= c), "overall"].mean() * 24 / 100 + \
    df.loc[(df["day_diff"] > c), "overall"].mean() * 22 / 100


df['helpful'] = df['helpful'].apply(lambda x: x[1:-1].split(',')) # helpful split by ","


df["helpful_yes"] = [int(i[0]) for i in df['helpful']]
df["total_vote"] = [int(i[1]) for i in df['helpful']]
df["helpful_no"] = [(int(i[1]) - int(i[0])) for i in df['helpful']]

# helpful have 2 values. 1. helpful_yes, 2. total_vote


df.sort_values("helpful_yes", ascending=False).head(10)

def score_pos_neg_diff(pos, neg):
    # calculation method.
    return pos - neg

df["score_pos_neg_diff"] = df.apply(lambda x: score_pos_neg_diff(x["helpful_yes"], x["helpful_no"]), axis=1) # score_pos_neg_diff functions used with apply.

df.head()
df.sort_values("score_pos_neg_diff", ascending=False).head(10)
# sorting by score_pos_neg_diff and checked.

def score_average_rating(pos, neg):
    # calculation method.
    if pos + neg == 0:
        return 0
    return pos / (pos + neg)


df["score_average_rating"] = df.apply(lambda x: score_average_rating(x["helpful_yes"], x["helpful_no"]), axis=1) # score_average_rating functions used with apply.


df.sort_values("score_average_rating", ascending=False).head(10) # sorting for check.


def wilson_lower_bound(pos, neg, confidence=0.95):
    """
    Calculate Wilson Lower Bound Score
    Parameters
    ----------
    pos: int
        number of positive comments
    neg: int
        number of negative comments
    confidence: float
        confidence interval

    Returns
    -------
    wilson score: float

    """
    n = pos + neg
    if n == 0:
        return 0
    z = st.norm.ppf(1 - (1 - confidence) / 2)
    phat = 1.0 * pos / n
    return (phat + z * z / (2 * n) - z * math.sqrt((phat * (1 - phat) + z * z / (4 * n)) / n)) / (1 + z * z / n)

df["wilson_lower_bound"] = df.apply(lambda x: wilson_lower_bound(x["helpful_yes"], x["helpful_no"]), axis=1) # wilson_lower_bound functions used with apply.
# apply ile Wilson lower bound fonksiyonu kullanıldı.

df.sort_values("wilson_lower_bound", ascending=False).head(20) # sorting for check.

df["TotalScore"] = ((df["score_pos_neg_diff"] * 27 / 100) + (df["score_average_rating"] * 20 / 100) + (df["wilson_lower_bound"] * 28 / 100)) # results multiplied by weight.

df.sort_values("TotalScore", ascending=False).head(20) # First 20 comment sorted.














# PS: You can download all data in # http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_Electronics.json.gz

# PS2: You can use 132 - 152 rows for unzip the data.

"""
import pandas as pd
import gzip

def parse(path):
    g = gzip.open(path, 'rb')
    for l in g:
        yield eval(l)

def get_df(path):
    i = 0
    df = {}
    for d in parse(path):
        df[i] = d
        i += 1
    return pd.DataFrame.from_dict(df, orient='index')


df_ = get_df('datasets/reviews_Electronics_5.json.gz')
df = df_.copy()
df.head()
df.shape

"""



