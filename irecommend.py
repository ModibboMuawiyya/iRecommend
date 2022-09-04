import streamlit as st
import pickle
import pandas as pd
import numpy as np
import requests


movies_dict = pickle.load(open('movies_dict.pkl', 'rb'))
movies = pd.DataFrame.from_dict(movies_dict)
cosineSim = pickle.load(open('similarity.pkl', 'rb'))

#### UTILITY FUNCTIONS
def get_title_from_index(index):
    indexMovie = movies[movies.index == index]["title"].values[0]  
    return indexMovie

def get_index_from_title(title):
    return movies[movies['title'] == title].index[0]

def similar_movies(movie_index):
    listofMovies = list(enumerate(cosineSim[movie_index]))
    return listofMovies
def sorted_similar_movies(similar_movies):
    sortedMov = sorted(similar_movies, key=lambda x:x[1],reverse=True)
    return sortedMov

def getMostSimilar(sorted_similar_movies):
    i = 1
    listMovies = []
   # moviesPosters = []
    for movie in sorted_similar_movies:
        sortedMovies = get_title_from_index(movie[0])   
        #recMovPosters = fetch_poster(sortedMovies)     
        listMovies.append(sortedMovies)
        #moviesPosters.append(recMovPosters)
        i = i +1
        if i>10:
            break    
    return listMovies

def retMovList(movie_title):
    movsList = Reccosimilarity(movie_title) 
    movsList2 = movsList.iloc[:, -1]    
    list = []
    for movie in movsList2:        
        movie_index = get_index_from_title(movie)
        list.append(movie_index)
    return list

def getMoviesforTopsis(movie_title):
    movieList = retMovList(movie_title)
    re = movies.iloc[movieList]  
    re2 = re.iloc[: , :]    
    data = re2.drop([ 'movie_id','title','tags'], axis = 1)
    return data

def getmovieTitles(movie_title):
    data = retMovList(movie_title)    
    altMovies= []     
    for movie in data:
        mov = get_title_from_index(movie)
        altMovies.append(mov)
    return altMovies

def getIdfromTitle(title):
    movId = movies[movies['title'] == title]["movie_id"].values[0]
    return movId

def getMCRSmovieTitles(movresultIndex):
    movsSortedResult = movresultIndex 
    altMovies= [] 
    for movie in movsSortedResult:
        mov = get_title_from_index(movie)
        altMovies.append(mov)
    return altMovies
###UTILITY FUNCTIONS END 
### RECOMMENDATION ENGINES BEGIN
def Reccosimilarity(movie_title):
    movie_index = get_index_from_title(movie_title)    
    simMov = similar_movies(movie_index)
    sorted_simMov = sorted_similar_movies(simMov)
    chk = getMostSimilar(sorted_simMov)
    chkre = pd.DataFrame(chk)
    chkre.columns = ['Similar Movies']
    result = chkre.drop(0)
   
    return result

def getMCRSReccommend(movie_title):
    data = getMoviesforTopsis(movie_title)
    #Define the weights to be used for the topsis 
    w = [0.3,0.2,0.2,0.15,0.15]
    #Data Normalisation
    data_norm = data/np.sqrt(np.power(data,2).sum(axis=0))
    #Multiply the Normalised Data with the weight
    data_normW = data_norm*w
    #Get the Highest and Lowest Ideal Alternatives from the Normalised Data 
    positive_ideal = data_normW.max()
    negative_ideal = data_normW.min()
    #Calculate the Positive and Negative Ideals
    SM_P = np.sqrt(np.power(data_normW - positive_ideal,2).sum(axis=1))
    SM_N = np.sqrt(np.power(data_normW - negative_ideal,2).sum(axis=1))
    #Get the Result of the Calculations
    result = pd.DataFrame(SM_N/(SM_N+SM_P))
    result.columns = ['Score']
    sortedResult = result.sort_values(['Score'], ascending= False)
    movresultIndex = sortedResult.index
    sortMovieTitles = getMCRSmovieTitles(movresultIndex)
    sortedResult['title'] = np.array(sortMovieTitles)
    
    return sortedResult
#### Reccomendation Engine Ends

### FRONTEND
st.title('Sankara Reccomender System')

selected_movie_name = st.selectbox(
    'Write or Select the Name of a movie you liked ?',
    movies['title'].values)

if st.button('Multi criteria Reccommendation'):
    Similar_Movies = getMCRSReccommend(selected_movie_name)
    st.write(Similar_Movies)

if st.button('Recommend'):
    Similar_Movies = Reccosimilarity(selected_movie_name)
    st.write(Similar_Movies)