##########################################################
# HYBRID RECCOMENDER SYSTEM
##########################################################


# DATA PREPARATION

import pandas as pd

movie = pd.read_csv('datasets/movie.csv')
rating = pd.read_csv('datasets/rating.csv')
df = movie.merge(rating, how="left", on="movieId")

df['title'] = df.title.str.replace('(\(\d\d\d\d\))', '')
df['title'] = df['title'].apply(lambda x: x.strip())
a = pd.DataFrame(df["title"].value_counts())
rare_movies = a[a["title"] <= 1000].index
common_movies = df[~df["title"].isin(rare_movies)]
user_movie_df = common_movies.pivot_table(index=["userId"], columns=["title"], values="rating")
user_id = 108170


# USER WATCHED FILMS DETERMINED.

selected_user_df = user_movie_df[user_movie_df.index == user_id]

movies_watched = selected_user_df.columns[selected_user_df.notna().any()].tolist()


# ACCESING DATA AND IDS OF OTHER USERS WATCHING THE SAME MOVIES

movies_watched_df = user_movie_df[movies_watched]
movies_watched_df.head()
movies_watched_df.shape

user_movie_count = movies_watched_df.T.notnull().sum()
user_movie_count = user_movie_count.reset_index()
user_movie_count.columns = ["userId", "movie_count"]

perc = len(movies_watched) * 60 / 100
users_same_movies = user_movie_count[user_movie_count["movie_count"] > perc]["userId"]


# SAME USERS DETERMINED

final_df = pd.concat([movies_watched_df[movies_watched_df.index.isin(users_same_movies.index)],
                      selected_user_df[movies_watched]])

final_df.head()

final_df.T.corr()

corr_df = final_df.T.corr().unstack().sort_values().drop_duplicates()
corr_df = pd.DataFrame(corr_df, columns=["corr"])
corr_df.index.names = ['user_id_1', 'user_id_2']
corr_df = corr_df.reset_index()
corr_df.head()


top_users = corr_df[(corr_df["user_id_1"] == user_id) & (corr_df["corr"] >= 0.65)][
    ["user_id_2", "corr"]].reset_index(drop=True)

top_users = top_users.sort_values(by='corr', ascending=False)

top_users.rename(columns={"user_id_2": "userId"}, inplace=True)

top_users

rating = pd.read_csv('datasets/rating.csv')
top_users_ratings = top_users.merge(rating[["userId", "movieId", "rating"]], how='inner')

top_users_ratings


# WEIGHTED RATES CALCULATED.


top_users_ratings['weighted_rating'] = top_users_ratings['corr'] * top_users_ratings['rating']
top_users_ratings.head()


# WEIGHTED AVERAGE RECOMMENDATION SCORES CALCULATED AND FIRST 5 MOVIE FOUND.

temp = top_users_ratings.groupby('movieId').sum()[['corr', 'weighted_rating']]
temp.columns = ['sum_corr', 'sum_weighted_rating']

temp.head()

recommendation_df = pd.DataFrame()
recommendation_df['weighted_average_recommendation_score'] = temp['sum_weighted_rating'] / temp['sum_corr']
recommendation_df['movieId'] = temp.index
recommendation_df = recommendation_df.sort_values(by='weighted_average_recommendation_score', ascending=False)
recommendation_df.head(10)

movie = pd.read_csv('datasets/movie.csv')
movies_from_user_based = movie.loc[movie['movieId'].isin(recommendation_df['movieId'].head(10))]['title']
movies_from_user_based.head(10)
movies_from_user_based[:5].values


# 5 ITEM BASED - 5 USER BASED RECOMMENDATION

movie = pd.read_csv('datasets/movie.csv')
rating = pd.read_csv('datasets/rating.csv')
df = movie.merge(rating, how="left", on="movieId")
df.head(10)


movie_id = rating[(rating["userId"] == user_id) & (rating["rating"] == 5.0)].sort_values(by="timestamp", ascending=False)["movieId"][0:1].values[0]

movie_title = movie[movie["movieId"] == movie_id]["title"].str.replace('(\(\d\d\d\d\))', '').str.strip().values[0]

df['year_movie'] = df.title.str.extract('(\(\d\d\d\d\))', expand=False)
df['year_movie'] = df.year_movie.str.extract('(\d\d\d\d)', expand=False)
df['title'] = df.title.str.replace('(\(\d\d\d\d\))', '')
df['title'] = df['title'].apply(lambda x: x.strip())

df["genre"] = df["genres"].apply(lambda x: x.split("|")[0])
df.drop("genres", inplace=True, axis=1)
df.head()


df["timestamp"] = pd.to_datetime(df["timestamp"], format='%Y-%m-%d')
df["year"] = df["timestamp"].dt.year
df["month"] = df["timestamp"].dt.month
df["day"] = df["timestamp"].dt.day
df.head()


a = pd.DataFrame(df["title"].value_counts())
a.head()

rare_movies = a[a["title"] <= 1000].index
common_movies = df[~df["title"].isin(rare_movies)]
common_movies.shape
common_movies["title"].nunique()

user_movie_df = common_movies.pivot_table(index=["userId"], columns=["title"], values="rating")
user_movie_df.shape
user_movie_df.head(10)
user_movie_df.columns

len(user_movie_df.columns)
common_movies["title"].nunique()

movie = user_movie_df[movie_title]
movies_from_item_based = user_movie_df.corrwith(movie).sort_values(ascending=False)
movies_from_item_based[1:6].index

recommendation = pd.DataFrame()
recommendation['user_based_recommendations'] = movies_from_user_based[:5].values.tolist()
recommendation['item_based_recommendations'] = movies_from_item_based[1:6].index
recommendation

"""
            user_based_recommendations      item_based_recommendations
0     In the Name of the Father (1993)              My Science Project
1                          Rudy (1993)                    Mediterraneo
2            North by Northwest (1959)  National Lampoon's Senior Trip
3  Mr. Smith Goes to Washington (1939)        Old Man and the Sea, The
4            African Queen, The (1951)                        Cashback
"""
