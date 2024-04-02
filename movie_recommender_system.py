import pandas

#load in data

rating_features = ["user id", "movie id", "rating", "timestamp"]
ratings_dataframe = pandas.read_csv("u.data",
                                     sep="\t",
                                     names=rating_features,
                                     encoding="ISO-8859-1")


movie_features=["movie id", "movie title", "release date", "video release date",
                 "IMDB URL", "unknown", "Action", "Adventure", "Animation", "childrens",
                 "Comedy", "Crime", "Documentary", "Drama", "Fantasy", "Film-Noir",
                 "Horror", "Musical", "Mystery", "Romance", "Sci-Fi", "Thriller", "War",
                 "Western"]
movies_dataframe=pandas.read_csv("u.item",
                                 sep="|",
                                 names=movie_features,
                                 encoding="ISO-8859-1")

user_features=["user id", "age", "gender", "occupation", "zip code"]
users_dataframe=pandas.read_csv("u.user", sep="|", names=user_features, encoding="ISO-8859-1")

#Process Data

NUMBER_OF_GROUPS=5
users_dataframe["age group"]=pandas.qcut(users_dataframe["age"], q=NUMBER_OF_GROUPS, precision=0)

merged_dataframes=pandas.merge(pandas.merge(ratings_dataframe,
                                            users_dataframe[["user id",
                                                             "age group",
                                                             "gender",
                                                             "occupation"]],
                                            on = "user id",
                                            how="left"),
                                movies_dataframe,
                                on="movie id",
                                how="left")

merged_dataframes.drop(["movie id", "movie title", "release date", "timestamp",
                        "unknown", "IMDB URL", "video release date"],
                        axis =1,
                        inplace = True)

#Build Categories

merged_dataframes["age group"]=pandas.Categorical(merged_dataframes["age group"])
age_group_dummies=pandas.get_dummies(merged_dataframes["age group"])

merged_dataframes["gender"]=pandas.Categorical(merged_dataframes["gender"])
gender_dummies=pandas.get_dummies(merged_dataframes["gender"])

merged_dataframes["occupation"]=pandas.Categorical(merged_dataframes["occupation"])
occupation_dummies=pandas.get_dummies(merged_dataframes["occupation"])

merged_dataframes=pandas.concat([merged_dataframes,
                                 age_group_dummies,
                                 gender_dummies,
                                 occupation_dummies],
                                axis=1)

merged_dataframes.drop(["age group",
                        "gender",
                        "occupation"],
                        axis=1,
                        inplace=True)

#Build a ridge regression model

from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV
import numpy

ridge_model=Ridge()
alpha=[]
ALPHA_LENGTH=7
NUMBER_OF_FOLDS=5

for i in range(ALPHA_LENGTH):
    alpha.extend(numpy.arange(10**(i-5), 10**(i-4), 10**(i-5)*2))

parameters={ "alpha": alpha }

ridge_cross_validation=GridSearchCV(estimator=ridge_model,
                                    param_grid=parameters,
                                    scoring="neg_mean_absolute_error",
                                    cv=NUMBER_OF_FOLDS,
                                    return_train_score=True,
                                    verbose=1)

COLUMN_AXIS=1
X=merged_dataframes.drop("rating", 
                        axis=COLUMN_AXIS)
X.columns=X.columns.astype(str)
y=merged_dataframes.rating

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test=train_test_split(X, y, test_size=0.2)

#ridge_cross_validation.fit(X_train, y_train)

#Evaluate model error

'''
ridge_cross_validation.best_estimator_

best_alpha=1e-05
best_alpha_ridge=Ridge(alpha=best_alpha)
best_alpha_ridge.fit(X_train, y_train)
'''

from sklearn.metrics import mean_squared_error

'''
error = mean_squared_error(y_test, best_alpha_ridge.predict(X_test))

#figure out top features

ridge_results_dataframe=pandas.DataFrame({
    "Features": X_train.columns,
    "coefficient": best_alpha_ridge.coef_
})

import matplotlib.pyplot as pyplot

figure, axes = pyplot.subplots(figsize=[7, 15])

import seaborn

seaborn.barplot(x= "coefficient",
                y="Features",
                ax = axes,
                data=ridge_results_dataframe)

#removed_features=ridge_results_dataframe.coefficient == float(0).sum()

ridge_results_dataframe.sort_values(by="coefficient",
                                    ascending=False,
                                    inplace=True)

ridge_results_dataframe.reset_index(inplace=True, drop=True)

NUMBER_OF_TOP_FEATURES=15

ridge_results_dataframe = ridge_results_dataframe.iloc[:NUMBER_OF_TOP_FEATURES]


figure, axes =pyplot.subplots(figsize=[10, 10])
seaborn.barplot(y="Features",
                x="coefficient",
                ax= axes,
                data= ridge_results_dataframe)
'''

#Build a Lasso Regression Model

from sklearn.linear_model import Lasso

lasso_model = Lasso()

lasso_cross_validation = GridSearchCV(estimator = lasso_model,
             param_grid= parameters,
             scoring = "neg_mean_absolute_error",
             cv = NUMBER_OF_FOLDS,
             return_train_score = True,
             verbose = 1)

lasso_cross_validation.fit(X_train, y_train)

lasso_cross_validation.best_estimator_

best_alpha_value_lasso = 1e-05

best_alpha_lasso_model = Lasso(alpha = best_alpha_value_lasso)

best_alpha_lasso_model.fit(X_train, y_train)

lasso_error = mean_squared_error(y_test,
                   best_alpha_lasso_model.predict(X_test))