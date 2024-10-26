import numpy as np
import pandas as pd
import ast
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.stem.porter import PorterStemmer

# Load datasets
movies = pd.read_csv('/Users/ironman/Downloads/ML(PYTHON)/Projects/Movie-Recommendation/tmdb_5000_movies.csv')
credits = pd.read_csv('/Users/ironman/Downloads/ML(PYTHON)/Projects/Movie-Recommendation/tmdb_5000_credits.csv')

# Merge datasets
movies = movies.merge(credits, on='title')

# Select relevant columns
movies = movies[['movie_id', 'title', 'overview', 'genres', 'keywords', 'cast', 'crew']]

# Drop rows with missing values and duplicates
movies.dropna(inplace=True)
movies.drop_duplicates(inplace=True)

# Function to convert genres, keywords, cast, and crew columns
def convert(text):
    L = []
    for i in ast.literal_eval(text):
        L.append(i['name'])
    return L

def convert_cast(text):
    L = []
    for i, person in enumerate(ast.literal_eval(text)):
        if i < 3:  # Only take the top 3 cast members
            L.append(person['name'])
    return L

def fetch_director(text):
    L = []
    for i in ast.literal_eval(text):
        if i['job'] == 'Director':
            L.append(i['name'])
    return L

def collapse(L):
    return [i.replace(" ", "") for i in L]

# Apply functions to relevant columns
movies['genres'] = movies['genres'].apply(convert).apply(collapse)
movies['keywords'] = movies['keywords'].apply(convert).apply(collapse)
movies['cast'] = movies['cast'].apply(convert_cast).apply(collapse)
movies['crew'] = movies['crew'].apply(fetch_director).apply(collapse)
movies['overview'] = movies['overview'].apply(lambda x: x.split())

# Combine overview, genres, keywords, cast, and crew into 'tags'
movies['tags'] = movies['overview'] + movies['genres'] + movies['keywords'] + movies['cast'] + movies['crew']
new_df = movies[['movie_id', 'title', 'tags']]

# Convert tags to string format
new_df['tags'] = new_df['tags'].apply(lambda x: " ".join(x).lower())

# Initialize CountVectorizer
cv = CountVectorizer(max_features=5000, stop_words='english')
vectors = cv.fit_transform(new_df['tags']).toarray()

# Initialize PorterStemmer
ps = PorterStemmer()
def stem(text):
    return " ".join([ps.stem(word) for word in text.split()])

# Apply stemming to tags
new_df['tags'] = new_df['tags'].apply(stem)

# Calculate similarity
similarity = cosine_similarity(vectors)

# Recommendation function
def recommend(movie):
    movie_index = new_df[new_df['title'] == movie].index[0]
    distances = similarity[movie_index]
    movies_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:6]
    
    for i in movies_list:
        print(new_df.iloc[i[0]].title)

# Test recommendation
recommend('Batman')
