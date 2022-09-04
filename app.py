import pandas as pd
import numpy as np
import skcriteria as skc
from mpl_toolkits.axes_grid1 import make_axes_locatable
from sklearn.cluster import KMeans
from sklearn.metrics import mean_squared_error
import itertools
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import silhouette_samples, silhouette_score
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st

#Import files to use
movies = pd.read_csv('movie_dataset.csv')
ratings = pd.read_csv('ratings.csv')

#Data cleaning and Preprocessing 
movies.drop(['homepage'], axis = 1,inplace = True)
final_dataset = ratings.pivot(index='movieId',columns='userId', values='rating')
final_dataset.fillna(0,inplace=True)

# First recommendation using collaborative filtering with user to item
noOfUsersVoted = ratings.groupby('movieId')['rating'].agg('count')
noOfMoviesVoted = ratings.groupby('userId')['rating'].agg('count')

# getting movies that have been voted above 50 times and users who have voted more than 250 times
final_dataset = final_dataset.loc[noOfUsersVoted[noOfUsersVoted > 50].index,:]
final_dataset=final_dataset.loc[:,noOfMoviesVoted[noOfMoviesVoted > 250].index]

#Converting result from users and movies relationship to a Compressed Sparse matrix
csr_data = csr_matrix(final_dataset.values)

#Resetting the index of the matrix
final_dataset.reset_index(inplace=True)

#Using KNN to fit the Sparse Matrix
knn = NearestNeighbors(metric='cosine', algorithm='brute', n_neighbors=20, n_jobs=-1)
knn.fit(csr_data)

#Get Movie Namefrom User

#Get Reccomendation using Collaborative Filtering 
def get_movie_recommendation(movie_name):
    n_movies_to_reccomend = 10
    movie_list = movies[movies['title'].str.contains(movie_name)]  
    if len(movie_list):        
        movie_idx= movie_list.iloc[0]['movieId']
        movie_idx = final_dataset[final_dataset['movieId'] == movie_idx].index[0]
        distances , indices = knn.kneighbors(csr_data[movie_idx],n_neighbors=n_movies_to_reccomend+1)    
        rec_movie_indices = sorted(list(zip(indices.squeeze().tolist(),distances.squeeze().tolist())),key=lambda x: x[1])[:0:-1]
        recommend_frame = []
        for val in rec_movie_indices:
            movie_idx = final_dataset.iloc[val[0]]['movieId']
            idx = movies[movies['movieId'] == movie_idx].index
            recommend_frame.append({'Title':movies.iloc[idx]['title'].values[0],'Distance':val[1]})
        df = pd.DataFrame(recommend_frame,index=range(1,n_movies_to_reccomend+1))
        records = df.to_records(index=False)
        result = list(records)
        return result
    else:
        return "No movies found. Please check your input"

#