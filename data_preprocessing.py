import os
import pandas as pd

def load_and_preprocess(movies_path=None, ratings_path=None):
    """Load movies and ratings CSVs and perform basic preprocessing.

    Returns: movies_df, ratings_df
    """
    # Resolve files if not provided
    if movies_path is None:
        for candidate in ('movies.csv', 'movie.csv'):
            if os.path.exists(candidate):
                movies_path = candidate
                break
    if movies_path is None or not os.path.exists(movies_path):
        raise FileNotFoundError('Movies file not found. Tried movies.csv and movie.csv')

    if ratings_path is None:
        for candidate in ('ratings.csv', 'rating.csv'):
            if os.path.exists(candidate):
                ratings_path = candidate
                break
    if ratings_path is None or not os.path.exists(ratings_path):
        raise FileNotFoundError('Ratings file not found. Tried ratings.csv and rating.csv')

    movies = pd.read_csv(movies_path)
    ratings = pd.read_csv(ratings_path)

    # Clean genres
    movies['genres_clean'] = movies['genres'].str.replace('|', ' ')
    movies['genres_clean'] = movies['genres_clean'].str.replace('(no genres listed)', '')

    # Extract year and clean title
    movies['year'] = movies['title'].str.extract(r'\((\d{4})\)')
    movies['title_clean'] = movies['title'].str.replace(r'\(\d{4}\)', '').str.strip()

    # Rating counts
    rating_counts = ratings.groupby('movieId').size().reset_index(name='rating_count')
    movies = movies.merge(rating_counts, on='movieId', how='left')
    movies['rating_count'] = movies['rating_count'].fillna(0)

    return movies, ratings
