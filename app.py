import os
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class MovieRecommender:
    def __init__(self, movies_path=None, ratings_path=None):
        """Initialize the recommender with MovieLens data.

        The code will try common filename variants if explicit paths aren't provided
        (e.g. 'movies.csv' vs 'movie.csv', 'ratings.csv' vs 'rating.csv').
        """
        # Resolve movies file
        if movies_path is None:
            for candidate in ('movies.csv', 'movie.csv'):
                if os.path.exists(candidate):
                    movies_path = candidate
                    break
        if movies_path is None or not os.path.exists(movies_path):
            raise FileNotFoundError(f"Movies file not found. Tried common names: movies.csv, movie.csv.\nCurrent working dir: {os.getcwd()}")

        # Resolve ratings file
        if ratings_path is None:
            for candidate in ('ratings.csv', 'rating.csv'):
                if os.path.exists(candidate):
                    ratings_path = candidate
                    break
        if ratings_path is None or not os.path.exists(ratings_path):
            raise FileNotFoundError(f"Ratings file not found. Tried common names: ratings.csv, rating.csv.\nCurrent working dir: {os.getcwd()}")

        self.movies = pd.read_csv(movies_path)
        self.ratings = pd.read_csv(ratings_path)
        self.cosine_sim = None
        self._preprocess()
        
    def _preprocess(self):
        """Clean and prepare data"""
        # Clean genres
        self.movies['genres_clean'] = self.movies['genres'].str.replace('|', ' ')
        self.movies['genres_clean'] = self.movies['genres_clean'].str.replace('(no genres listed)', '')
        
        # Extract year from title
        self.movies['year'] = self.movies['title'].str.extract(r'\((\d{4})\)')
        self.movies['title_clean'] = self.movies['title'].str.replace(r'\(\d{4}\)', '').str.strip()
        
        # Add rating count for popularity
        rating_counts = self.ratings.groupby('movieId').size().reset_index(name='rating_count')
        self.movies = self.movies.merge(rating_counts, on='movieId', how='left')
        self.movies['rating_count'] = self.movies['rating_count'].fillna(0)
        
    def build_model(self):
        """Build TF-IDF matrix and compute cosine similarity"""
        # Create TF-IDF vectors from genres
        tfidf = TfidfVectorizer(stop_words='english', ngram_range=(1, 2))
        tfidf_matrix = tfidf.fit_transform(self.movies['genres_clean'])
        
        # Compute pairwise cosine similarity
        self.cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
        print(f"✓ Model built! Similarity matrix shape: {self.cosine_sim.shape}")
        
    def get_recommendations(self, title, top_n=5, min_ratings=10):
        """
        Get top N similar movies
        
        Parameters:
        - title: Movie title (partial match allowed)
        - top_n: Number of recommendations
        - min_ratings: Minimum rating count filter
        """
        if self.cosine_sim is None:
            raise ValueError("Model not built! Call build_model() first.")
        
        # Find movie index
        matches = self.movies[self.movies['title'].str.contains(title, case=False, na=False)]
        
        if len(matches) == 0:
            return f"❌ Movie '{title}' not found. Try a different search."
        
        if len(matches) > 1:
            print(f"Found {len(matches)} matches. Using: {matches.iloc[0]['title']}\n")
        
        idx = matches.index[0]
        movie_title = self.movies.iloc[idx]['title']
        
        # Get similarity scores
        sim_scores = list(enumerate(self.cosine_sim[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        
        # Filter by minimum ratings and get top N
        recommendations = []
        for i, score in sim_scores[1:]:  # Skip first (itself)
            if self.movies.iloc[i]['rating_count'] >= min_ratings:
                recommendations.append((i, score))
            if len(recommendations) >= top_n:
                break
        
        # Build results dataframe
        movie_indices = [i[0] for i in recommendations]
        scores = [i[1] for i in recommendations]
        
        results = self.movies.iloc[movie_indices][['title', 'genres', 'rating_count']].copy()
        results['similarity'] = scores
        results['similarity'] = results['similarity'].round(3)
        
        print(f"🎬 Movies similar to: {movie_title}")
        print("=" * 80)
        return results
    
    def search_movies(self, keyword, top_n=10):
        """Search for movies by keyword"""
        matches = self.movies[self.movies['title'].str.contains(keyword, case=False, na=False)]
        return matches[['title', 'genres', 'rating_count']].head(top_n)

    def top_by_genre(self, genre, top_n=10, min_ratings=0):
        """Return top movies for a given genre sorted by rating_count"""
        mask = self.movies['genres'].str.contains(genre, case=False, na=False)
        filtered = self.movies[mask].copy()
        filtered = filtered[filtered['rating_count'] >= min_ratings]
        filtered = filtered.sort_values('rating_count', ascending=False)
        return filtered[['title', 'genres', 'rating_count']].head(top_n)


# ============ USAGE EXAMPLE ============

if __name__ == "__main__":
    # Initialize recommender
    print("Loading MovieLens data...")
    recommender = MovieRecommender()
    
    # Build similarity model
    print("Building recommendation model...")
    recommender.build_model()
    
    print("\n" + "="*80)
    
    # Example 1: Toy Story
    print("\n📺 Example 1:")
    recs = recommender.get_recommendations('Toy Story', top_n=5)
    print(recs.to_string(index=False))
    
    # Example 2: Dark Knight
    print("\n\n📺 Example 2:")
    recs = recommender.get_recommendations('Dark Knight', top_n=5)
    print(recs.to_string(index=False))
    
    # Example 3: Search functionality
    print("\n\n🔍 Search Example:")
    print(recommender.search_movies('Matrix').to_string(index=False))

    # --------- Simple interactive CLI ---------
    def _input(prompt):
        try:
            return input(prompt)
        except EOFError:
            return ''

    print('\nInteractive mode: Ask for recommendations, search, or top-by-genre.')
    while True:
        print('\nOptions:\n 1) Recommend by title\n 2) Search by keyword\n 3) Top movies by genre\n 4) Exit')
        choice = _input('Choose an option (1-4): ').strip()
        if choice == '1':
            title = _input('Enter movie title (partial OK): ').strip()
            try:
                recs = recommender.get_recommendations(title, top_n=5)
                if isinstance(recs, str):
                    print(recs)
                else:
                    print(recs.to_string(index=False))
            except Exception as e:
                print('Error:', e)
        elif choice == '2':
            kw = _input('Enter search keyword: ').strip()
            print(recommender.search_movies(kw).to_string(index=False))
        elif choice == '3':
            genre = _input('Enter genre (e.g. Comedy, Drama): ').strip()
            min_r = _input('Minimum rating count (enter for 0): ').strip()
            try:
                min_r = int(min_r) if min_r else 0
            except ValueError:
                min_r = 0
            print(recommender.top_by_genre(genre, top_n=10, min_ratings=min_r).to_string(index=False))
        elif choice == '4' or choice == '':
            print('Exiting. Goodbye!')
            break
        else:
            print('Invalid option, please choose 1-4.')