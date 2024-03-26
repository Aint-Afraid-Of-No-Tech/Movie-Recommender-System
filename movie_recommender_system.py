import pandas

#reading, cleaning and merging our data

#ratings_features = ["user_id", "item_id", "rating", "timestamp"]

ratings_dataframe = pandas.read_csv("ratings.csv")

FIRST_INDEX_ROW = 0
ratings_dataframe = ratings_dataframe.drop (ratings_dataframe.index[FIRST_INDEX_ROW])

ratings_dataframe = ratings_dataframe.astype("float")

ratings_dataframe.info()

movies_dataframe = pandas.read_csv("movies.csv")
movies_dataframe.info()

movie_titles_dataframe = movies_dataframe[["movieId", "title"]]
movie_titles_dataframe["movieId"] = movie_titles_dataframe["movieId"].astype(str).astype(float)
movie_titles_dataframe.info()

merged_dataframe = pandas.merge(ratings_dataframe, movie_titles_dataframe, on= "movieId")

merged_dataframe.groupby("movieId")["rating"].count().sort_values(ascending = False)

#building our Correlation Matrix

crosstab = merged_dataframe.pivot_table(values = "rating",
                             index = "userId",
                             columns = "title",
                             fill_value = 0)

X = crosstab.T

from sklearn.decomposition import TruncatedSVD

NUMBER_OF_COMPONENTS = 12

singular_value_decomposition = TruncatedSVD(n_components=NUMBER_OF_COMPONENTS, random_state=1)
matrix = singular_value_decomposition.fit_transform(X)

import numpy

correlation_matrix=numpy.corrcoef(matrix)

#Test Recommender

movie_titles = crosstab.columns

movies_list = list(movie_titles)

example_movie_index = movies_list.index("Batman: Year One (2011)")

example_correlations = correlation_matrix[example_movie_index]

MAXIMUM_CORRELATION = 1.0

MINIMUM_CORRELATION = 0.9

test1 = list(movie_titles[(example_correlations < MAXIMUM_CORRELATION) & (example_correlations > MINIMUM_CORRELATION)])
print(test1)

