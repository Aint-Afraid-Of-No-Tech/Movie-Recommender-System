import pandas

#reading and preparing our data

movies_dataframe = pandas.read_csv("movies.csv")
movies_dataframe.set_index("movieId", inplace=True)

ratings_dataframe=pandas.read_csv("ratings.csv")

total_counts = ratings_dataframe["movieId"].value_counts()

movies_dataframe["ratingsCount"] = total_counts

SEGMENT_LENGTH = 10

#print(movies_dataframe.sort_values("ratingsCount", ascending=False).head(SEGMENT_LENGTH))

average_ratings = ratings_dataframe.groupby("movieId").mean()["rating"]

movies_dataframe["averageRating"] = average_ratings

#print(movies_dataframe.sort_values(["ratingsCount", "averageRating"], ascending=False).head(SEGMENT_LENGTH))

MINIMUM_RATINGS_COUNT = 100

minimum_ratings_subset = movies_dataframe.query(f"ratingsCount >= {MINIMUM_RATINGS_COUNT}").sort_values("ratingsCount")

#print(minimum_ratings_subset)

#Calculate similarity between Users

import numpy

'''
USER_1_POSITION_X = 3
USER_1_POSITION_Y = 7.5

USER_2_POSITION_X = 7
USER_2_POSITION_Y = 8

position_user_1 = numpy.array([USER_1_POSITION_X, USER_1_POSITION_Y])
position_user_2 = numpy.array([USER_2_POSITION_X, USER_2_POSITION_Y])

def find_distance_between_users(position_user_1, position_user_2):
    (delta_x, delta_y) = position_user_1 - position_user_2
    return (delta_x ** 2 + delta_y ** 2) ** (1/2)

find_distance_between_users(position_user_1, position_user_2)
'''
def find_user_ratings(userId):
    user_ratings = ratings_dataframe.query(f"userId == {userId}")
    return user_ratings[["movieId", "rating"]].set_index("movieId")
'''
ID_OF_USER_1 = 1
ID_OF_USER_2 = 610
USER_1_RATINGS = find_user_ratings(ID_OF_USER_1)
USER_2_RATINGS = find_user_ratings(ID_OF_USER_2)

ratings_comparison = USER_1_RATINGS.join(USER_2_RATINGS,
                                        lsuffix="_user1",
                                        rsuffix="_user2").dropna()

user1_compared = ratings_comparison["rating_user1"]
user2_compared = ratings_comparison["rating_user2"]

numpy.linalg.norm(user1_compared-user2_compared)
'''


def find_distance_between_real_users(userId_1, userId_2):
    rating_user1=find_user_ratings(userId_1)
    rating_user2=find_user_ratings(userId_2)
    ratings_comparison=rating_user1.join(rating_user2, lsuffix="_user1", rsuffix="_user2").dropna()
    user1_compared=ratings_comparison["rating_user1"]
    user2_compared=ratings_comparison["rating_user2"]
    distance_between_users=numpy.linalg.norm(user1_compared - user2_compared)
    return [userId_1, userId_2, distance_between_users]

#Find top similar Users

def find_relative_distances(userId):
    users = ratings_dataframe["userId"].unique()
    users = users[users != userId]
    distances=[find_distance_between_real_users(userId, every_id) for every_id in users]
    return pandas.DataFrame(distances, columns=["singleUserId", "userId", "distance"])

example_distances = find_relative_distances(7)

def find_top_similar_users(userId):
    distances_to_user = find_relative_distances(userId)
    distances_to_user = distances_to_user.sort_values("distance")
    distances_to_user = distances_to_user.set_index("userId")
    return distances_to_user

#recommend a movie based on user similarity

def make_movie_recommendation(userId):
    user_ratings = find_user_ratings(userId)
    top_similar_users = find_top_similar_users(userId)
    MOST_SIMILAR = 0
    most_similar_user = top_similar_users.iloc[MOST_SIMILAR]
    most_similar_user_ratings = find_user_ratings(most_similar_user.name)
    unwatched_movies = most_similar_user_ratings.drop(user_ratings.index,
                                                      errors = "ignore")
    unwatched_movies = unwatched_movies.sort_values("rating", ascending=False)
    recommended_movies = unwatched_movies.join(movies_dataframe)
    return recommended_movies

#Recommend movies based on K nearest user

NUMBER_OF_NEIGHBORS=5

def find_k_nearest_neighbors(userId, k=NUMBER_OF_NEIGHBORS):
    distances_to_user = find_relative_distances(userId)
    distances_to_user=distances_to_user.sort_values("distance")
    distances_to_user=distances_to_user.set_index("userId")
    return distances_to_user.head(k)

def make_recommendation_with_knn(userId, k=NUMBER_OF_NEIGHBORS):
    top_k_neighbors=find_k_nearest_neighbors(userId)
    ratings_by_index=ratings_dataframe.set_index("userId")
    top_similar_ratings=ratings_by_index.loc[top_k_neighbors.index]
    top_similar_ratings_average=top_similar_ratings.groupby("movieId").mean()[["rating"]]
    recommended_movie=top_similar_ratings_average.sort_values("rating", ascending=False)
    return recommended_movie.join(movies_dataframe)

#Create a sample user for testing

import random

NUMBER_OF_MOVIES=14
MINIMUM_NUMBER=1
ROWS_INDEX=0
MAXIMUM_NUMBER=movies_dataframe.shape[ROWS_INDEX]
test_user_watched_movies = []
MINIMUM_RATING=0
MAXIMUM_RATING=5
test_user_ratings=[]

for i in range(0, NUMBER_OF_MOVIES):
    random_movie_index=random.randint(MINIMUM_NUMBER, MAXIMUM_NUMBER)
    test_user_watched_movies.append(random_movie_index)

for index in range(0, NUMBER_OF_MOVIES):
    random_rating=random.randint(MINIMUM_RATING, MAXIMUM_RATING+1)
    test_user_ratings.append(random_rating)

user_data=[list(index) for index in zip(test_user_watched_movies, test_user_ratings)]

def create_new_user(user_data):
    new_user_id=ratings_dataframe["userId"].max()+1
    new_user_dataframe=pandas.DataFrame(user_data, columns= ["movieId", "rating"])
    new_user_dataframe["userId"]=new_user_id
    return pandas.concat([ratings_dataframe, new_user_dataframe])

#Recommend movies to sample user

NEW_USER_ID=611
print(make_recommendation_with_knn(NEW_USER_ID))