#############################################
# PROJE: Hybrid Recommender System
#############################################


# Make a prediction by using item-based and user-based recommender methods for the user whose ID is given below.
# Take 5 suggestions from user-based model and 5 suggestions from item-based model and finally make 10 suggestions from 2 models.
# user_id = 108170

# How should we proceed?
# You can easily get suggestions from two methods in two steps.
# In the first step, perform exactly the same procedure that we follwed in the user-based lesson. Take 5 different suggestions.
# In the second step, get item based 5 suggestions by calling the item_based_recommender function in the item-based lesson.
# According to which movie will we get suggestions? According to the name of the movie
# with the last highest rated movie watched by the user with the given ID.


# Let's explain how to get 5 suggestions from the # USER-BASED model.
# Perform all the steps that we discussed in the # USER_BASED course step by step for the user whose id is given.
# These steps are exactly the same as the lesson and consist of 6 steps.
# There is 2 smalll differences.
# First, for the new user, you have to do step 3 by calculating a percentage, not by entering a number manually.
# You should calculate the 60 percent of the number of movies watched by the user.
# You can use the following code for this:
# perc = len (movies_watched) * 60/100
# users_same_movies = user_movie_count [user_movie_count ["movie_count"]> perc] ["userId"]
# Second, you should show 5, not 10, of the suggestions.


# Let's explain how to get 5 suggestions from the # ITEM-BASED model.
# We will make an item-based suggestion according to the name of the movie with the highest score among
# the movies watched by the user given the # ID.
# First of all, using the relevant tables according to the id of the user (movie, rating) and find the latest 5-star
# rated movie by the user.Extract its title and clear it (from history)
# Then ask this title to the item_based_recommender function that we saw in the item_based lesson.
# Finally, put together user-based 5 suggestions and item-based 5 suggestions.
# It doesn't need to be programmatic.
# The following output should be obtained:

# movies_from_item_based[1:6].index

# Index(['My Science Project', 'Mediterraneo', 'National Lampoon's Senior Trip',
#        'Old Man and the Sea, The', 'Cashback'],
#       dtype='object', name='title')

# movies_from_user_based

# array(['In the Name of the Father (1993)', 'Rudy (1993)',
#        'North by Northwest (1959)', 'Mr. Smith Goes to Washington (1939)',
#        'African Queen, The (1951)'], dtype=object)


#############################################
# Step 1: Preparing the Data
#############################################
import pandas as pd
pd.set_option('display.max_columns', 20)
movie = pd.read_csv('movie.csv')
movie.shape #(27278,3)
movie.head()
rating = pd.read_csv('rating.csv')
rating.head()
rating.shape #(20000263,4)
rating["userId"].nunique() # 138493 users
# Merging movie and rating data sets
df= movie.merge(rating, how="left", on="movieId")
df.head()
df.shape

df["genres"].value_counts()
df["title"].unique()
df["title"].head()

# Creating a sample

# Question: replace =False? What does it mean?

'''df_sample = df.sample(frac=0.0001, replace=False, random_state=1)
df_sample.shape

df_sample['title'] = df_sample.title.str.replace('(\(\d\d\d\d\))', '')
df_sample['title'] = df_sample['title'].apply(lambda x: x.strip())
a = pd.DataFrame(df_sample["title"].value_counts())
df_sample.head()
rare_movies = a[a["title"] <= 5].index
common_movies = df_sample[~df_sample["title"].isin(rare_movies)]
user_movie_df = common_movies.pivot_table(index=["userId"], columns=["title"], values="rating")'''


def create_user_movie_df():
    df['title'] = df.title.str.replace('(\(\d\d\d\d\))', '') #replace title with blank line
    df['title'] = df['title'].apply(lambda x: x.strip()) # remove the empty line
    a = pd.DataFrame(df["title"].value_counts())
    rare_movies = a[a["title"] <= 1000].index # filter the titles smaller than 1000
    common_movies = df[~df["title"].isin(rare_movies)]
    user_movie_df = common_movies.pivot_table(index=["userId"], columns=["title"], values="rating")
    return user_movie_df

user_movie_df = create_user_movie_df()
df.head()
user_movie_df.head()
user_movie_df.shape # (138493, 3134)

#############################################
# Step 2: Determination of the movies watched by the recommended user
#############################################

user=108170
user_df = user_movie_df[user_movie_df.index == user]
user_df.head()
user_df.shape #(1, 3134)
movies_watched = user_df.columns[user_df.notna().any()].tolist()
len(movies_watched) #186
movies_watched[0:10]


#############################################
# Step 3: Accessing data and ID's of other users watching the same movies
#############################################

movies_watched_df = user_movie_df[movies_watched]
movies_watched_df.head()
movies_watched_df.shape
user_movie_count = movies_watched_df.T.notnull().sum()
user_movie_count = user_movie_count.reset_index()
user_movie_count.columns = ["userId", "movie_count"]

'''
          userId  movie_count
0            1.0           54
1            2.0           11
2            3.0           47
3            4.0            5
4            5.0           16
          ...          ...
138488  138489.0           11
138489  138490.0           10
138490  138491.0            3
138491  138492.0           20
138492  138493.0           57
'''
perc = len(movies_watched) *60 / 100
users_same_movies = user_movie_count[user_movie_count["movie_count"] > perc]["userId"]
users_same_movies.head()
users_same_movies.shape #(2342,)

#############################################
# Step 4: Determining the most similar users to the user who receives recommendations
#############################################
final_df = pd.concat([movies_watched_df[movies_watched_df.index.isin(users_same_movies.index)],
                      user_df[movies_watched]])
final_df.head()

# Correlations between users

final_df.T.corr()

# Preparation of corr_df

corr_df = final_df.T.corr().unstack().sort_values().drop_duplicates()
corr_df = pd.DataFrame(corr_df, columns=["corr"])
corr_df.index.names = ['user_id_1', 'user_id_2']
corr_df = corr_df.reset_index()
corr_df.head()

# Finding the similar users

top_users = corr_df[(corr_df["user_id_1"] == 108170) & (corr_df["corr"] >= 0.65)][
    ["user_id_2", "corr"]].reset_index(drop=True)

top_users.shape #(55,2)

top_users = top_users.sort_values(by='corr', ascending=False)

top_users.rename(columns={"user_id_2": "userId"}, inplace=True)

top_users #Similar users with our user

#Adding rating information

rating = pd.read_csv('rating.csv')
top_users_ratings = top_users.merge(rating[["userId", "movieId", "rating"]], how='inner')
top_users_ratings.shape #(4298, 4)
top_users_ratings.head()

#############################################
# Step 5: Calculation of weighted ratings
#############################################

# High Correlation and High Rating
top_users_ratings['weighted_rating'] = top_users_ratings['corr'] * top_users_ratings['rating']
top_users_ratings.head()

'''
   userId      corr  movieId  rating  weighted_rating
0  24218.0  0.993808       19     0.5         0.496904
1  24218.0  0.993808      318     5.0         4.969040
2  24218.0  0.993808      344     0.5         0.496904
3  24218.0  0.993808      435     4.5         4.472136
4  24218.0  0.993808      541     4.5         4.472136
'''


#############################################
#Step 6: Calculating the weighted average recommendation score and keeping the top five films
#############################################

# Groupping according to the movie ID

temp = top_users_ratings.groupby('movieId').sum()[['corr', 'weighted_rating']]
temp.columns = ['sum_corr', 'sum_weighted_rating']

temp.head()

'''
          sum_corr  sum_weighted_rating
movieId                                
1        10.695522            33.171936
2         4.397580            16.206339
3         1.342875             3.359913
5         1.336727             3.347617
6         4.849560            18.394512
            ...                  ...
98809     0.900638             3.602554
106696    0.856235             4.281173
106916    0.856235             2.996821
106918    0.824632             2.886213
108548    0.856235             4.281173

'''

recommendation_df = pd.DataFrame()
recommendation_df['weighted_average_recommendation_score'] = temp['sum_weighted_rating'] / temp['sum_corr']
recommendation_df['movieId'] = temp.index
recommendation_df = recommendation_df.sort_values(by='weighted_average_recommendation_score', ascending=False)
recommendation_df.head(5)


recommendation_df

'''
         weighted_average_recommendation_score  movieId
movieId                                                
969                                        5.0      969
908                                        5.0      908
954                                        5.0      954
4080                                       5.0     4080
5971                                       5.0     5971
                                        ...      ...
5452                                       0.5     5452
1981                                       0.5     1981
5539                                       0.5     5539
5672                                       0.5     5672
1559                                       0.5     1559

'''

# Getting first 5 movies

movie = pd.read_csv('movie.csv')
movie.loc[movie['movieId'].isin(recommendation_df.head(5)['movieId'])]['title']
#recommendation_df[recommendation_df["weighted_average_recommendation_score"]==5.0]

'''
891                        North by Northwest (1959)
937              Mr. Smith Goes to Washington (1939)
952                        African Queen, The (1951)
3986                                Baby Boom (1987)
5872    My Neighbor Totoro (Tonari no Totoro) (1988)
Name: title, dtype: object

'''

#############################################
# Step 7: Make an item-based suggestion based on the name of the movie with the latest highest score from the movies watched.
#Make 5 user-based suggestions and 5 item-based suggestions (In Total 10 Suggestions)
#############################################

## Getting the movie ID (the most up-to-date with highest score)
movie_id = rating[(rating["userId"] == user) & (rating["rating"] == rating["rating"].max())]. \
     sort_values(by="timestamp", ascending=False)["movieId"][0:1].values[0]

## Getting the movie title
movie_title = movie[movie["movieId"] == movie_id]["title"].str.replace('(\(\d\d\d\d\))', '').str.strip().values[0]

#Item Based Sugestions based on the correlation


def item_based_recommender(movie_title, user_movie_df):
    movie = user_movie_df[movie_title]
    return user_movie_df.corrwith(movie).sort_values(ascending=False).head(5)


item_based_suggestions= item_based_recommender(movie_title, user_movie_df)
item_based_suggestions

'''
Wild at Heart                     1.000000
My Science Project                0.570187
Mediterraneo                      0.538868
National Lampoon's Senior Trip    0.533029
Old Man and the Sea, The          0.503236'''


