from os import listdir
from os.path import isfile, join
import pandas as pd
import json
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import skew
from scipy.stats.stats import pearsonr
from sklearn.linear_model import Ridge, RidgeCV, ElasticNet, LassoCV, LassoLarsCV
from sklearn.model_selection import cross_val_score
#
# # Read the files into two lists
# file_dir = "Data/RawData/"
#
# data_files = [f for f in listdir(file_dir) if isfile(join(file_dir, f))]
# data_list = []
#
# for data_file in data_files:
#     if "Store" not in data_file:
#         data_list.append(pd.read_csv(file_dir + data_file, engine='python'))
#
# merged = data_list[0].join(data_list[1], lsuffix='title', rsuffix='title')
#
# merged.drop(['movie_id'], axis=1)
# merged = merged.join(data_list[2], lsuffix='title', rsuffix='title')
#
#
# rows = []
# # Flatten countries filmed
# for i, row in enumerate(merged['production_countries']):
#     if len(json.loads(row)) > 0:
#         rows.append("|".join([country['name'] for country in json.loads(row)]))
#     else:
#         rows.append("")
# merged.drop(['production_countries'], axis=1)
# se = pd.Series(rows)
# merged['production_countries'] = se.values
# cleaned_pc = merged.production_countries.str.split('|', expand=True).stack()
# production_countries = pd.get_dummies(cleaned_pc, prefix='g').groupby(level=0).sum()
#
#
# # One-hot encode the genres
# cleaned_g = merged.genres.str.split('|', expand=True).stack()
# genres = pd.get_dummies(cleaned_g, prefix='g').groupby(level=0).sum()
#
# # One-hot encode color, content_rating, and language
# color = pd.get_dummies(merged['color'])
# content_rating = pd.get_dummies(merged['content_rating'])
# language = pd.get_dummies(merged['language'])
#
# # Add the one-hot encoded dataframes to the final set
# merged = pd.concat([merged, genres], axis=1, sort=True)
# merged = pd.concat([merged, color], axis=1, sort=True)
# merged = pd.concat([merged, content_rating], axis=1, sort=True)
# merged = pd.concat([merged, language], axis=1, sort=True)
# merged = pd.concat([merged, production_countries], axis=1, sort=True)
#
#
# # Drop irrelevant columns
# merged = merged.drop(['genres', 'movie_title', 'gross', 'production_countries', 'num_user_for_reviews',
#                       'director_name', 'actor_2_name', 'actor_1_facebook_likes',
#                       'actor_1_name', 'actor_3_name', 'color', 'content_rating', 'language'], axis=1)
#
# # Drop columns that we will not be using yet, but might later (keywords)
# merged = merged.drop(['plot_keywords', 'keywords'], axis=1)
#
# # I think production companies might be a bit much. lets remove it for now.
# merged = merged.drop(['production_companies'], axis=1)
#
# merged = merged.set_index('title')
#
# # Fix the release date (https://stackoverflow.com/questions/46428870/how-to-handle-date-variable-in-machine-learning-data-pre-processing)
# # Dropping, but this is definitely something to do feature engineering on!
# merged = merged.drop(['release_date'], axis=1)
#
#
# # set to average any 0 or nan for most of the continous columns
# # 0 or nan: revenue, duration
# zero_or_nan_average = ['revenue', 'duration']
# for col in zero_or_nan_average:
#     mean = merged[col].mean()
#     merged[col].fillna((mean), inplace=True)
#     merged = merged.replace({col: {0: mean}})
#
#
# just_nan_average = ['vote_average', 'vote_count', 'num_critic_for_reviews', 'director_facebook_likes', 'actor_3_facebook_likes', 'num_voted_users',
#                     'cast_total_facebook_likes', 'facenumber_in_poster', 'actor_2_facebook_likes', 'movie_facebook_likes']
# for col in just_nan_average:
#     merged[col].fillna((merged[col].mean()), inplace=True)
#
# merged = merged.dropna()
#
# msk = np.random.rand(len(merged)) < 0.8
# train = merged[msk]
# test = merged[~msk]
#
# #creating matrices for sklearn:
# X_train = merged[:train.shape[0]]
# X_test = merged[train.shape[0]:]
# # cross_ten = np.array_split(merged, 10)
# alphas = [0.0001, 0.001, 0.01, 0.1, 1, 10]
# y = train.revenue
#
# print(merged)


features_to_train = ["revenue", "vote_average", "vote_count"]
msk = np.random.rand(len(merged)) < 0.8
for feature in features_to_train:
    train.append(merged[msk])
    test.append(merged[~msk])

    test_sols.append(test[feature])
    train[len(train) - 1].to_csv(r'Data/train.csv', index=train[len(train) - 1].index.tolist(), header=True)
    test[len(test) - 1].drop(feature, axis=1).to_csv(r'Data/test.csv', index=test[len(test) - 1].index.tolist(),
                                                     header=True)
    test[len(test) - 1].to_csv(r'Data/test_sol.csv', index=test[len(test) - 1].index.tolist(), columns=[feature])

# if feature_to_train == "revenue":

# elif feature_to_train == "vote_count":
#     test_sols = test['vote_count']
#     train.to_csv(r'Data/train.csv', index = train.index.tolist(), header=True)
#     test.drop("vote_count", axis=1).to_csv(r'Data/test.csv', index = test.index.tolist(), header=True)
#     test.to_csv(r'Data/test_sol.csv', index = test.index.tolist(), columns = ["vote_count"])
# elif feature_to_train == "vote_average":
#     test_sols = test['vote_average']
#     train.to_csv(r'Data/train.csv', index = train.index.tolist(), header=True)
#     test.drop("vote_average", axis=1).to_csv(r'Data/test.csv', index = test.index.tolist(), header=True)
#     test.to_csv(r'Data/test_sol.csv', index = test.index.tolist(), columns = ["vote_average"])

for i, feature in enumerate(features_to_train):
    train[i] = pd.read_csv("Data/train.csv")
    test[i] = pd.read_csv("Data/test.csv")