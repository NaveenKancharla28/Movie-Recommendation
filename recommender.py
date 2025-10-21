import sys
import argparse
from data_preprocessing import load_and_preprocess
from model_builder import build_similarity_model
import pandas as pd

class MovieRecommender:
    def __init__(self, movies_path=None, ratings_path=None):
        self.movies, self.ratings = load_and_preprocess(movies_path, ratings_path)
        self.cosine_sim = None

    def build_model(self):
        self.cosine_sim = build_similarity_model(self.movies)
        print(f"✓ Model built! Similarity matrix shape: {self.cosine_sim.shape}")

    def get_recommendations(self, title, top_n=5, min_ratings=10):
        if self.cosine_sim is None:
            raise ValueError('Model not built!')
        matches = self.movies[self.movies['title'].str.contains(title, case=False, na=False)]
        if len(matches) == 0:
            return f"Movie '{title}' not found."
        idx = matches.index[0]
        sim_scores = list(enumerate(self.cosine_sim[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        recommendations = []
        for i, score in sim_scores[1:]:
            if self.movies.iloc[i]['rating_count'] >= min_ratings:
                recommendations.append((i, score))
            if len(recommendations) >= top_n:
                break
        movie_indices = [i[0] for i in recommendations]
        scores = [i[1] for i in recommendations]
        results = self.movies.iloc[movie_indices][['title', 'genres', 'rating_count']].copy()
        results['similarity'] = scores
        results['similarity'] = results['similarity'].round(3)
        return results

    def search_movies(self, keyword, top_n=10):
        matches = self.movies[self.movies['title'].str.contains(keyword, case=False, na=False)]
        return matches[['title', 'genres', 'rating_count']].head(top_n)

    def top_by_genre(self, genre, top_n=10, min_ratings=0):
        mask = self.movies['genres'].str.contains(genre, case=False, na=False)
        filtered = self.movies[mask].copy()
        filtered = filtered[filtered['rating_count'] >= min_ratings]
        filtered = filtered.sort_values('rating_count', ascending=False)
        return filtered[['title', 'genres', 'rating_count']].head(top_n)


def main():
    parser = argparse.ArgumentParser(description='Movie recommender (interactive or non-interactive).')
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--recommend', '-r', metavar='TITLE', help='Recommend movies similar to TITLE (partial match allowed)')
    group.add_argument('--search', '-s', metavar='KEYWORD', help='Search movies by keyword')
    group.add_argument('--top-genre', '-g', metavar='GENRE', help='Show top movies for GENRE')
    parser.add_argument('--top_n', '-n', type=int, default=5, help='Number of results to return')
    parser.add_argument('--min_ratings', '-m', type=int, default=10, help='Minimum rating count filter')
    args = parser.parse_args()

    print('Loading MovieLens data...')
    recommender = MovieRecommender()
    print('Building recommendation model...')
    recommender.build_model()

    # Non-interactive mode if args provided
    if args.recommend or args.search or args.top_genre:
        if args.recommend:
            recs = recommender.get_recommendations(args.recommend, top_n=args.top_n, min_ratings=args.min_ratings)
            if isinstance(recs, str):
                print(recs)
            else:
                # compact output
                out = recs[['title', 'rating_count', 'similarity']].copy()
                out['title'] = out['title'].str.slice(0,80)
                print(out.to_string(index=False))
        elif args.search:
            res = recommender.search_movies(args.search, top_n=args.top_n)
            out = res[['title', 'rating_count']].copy()
            out['title'] = out['title'].str.slice(0,80)
            print(out.to_string(index=False))
        elif args.top_genre:
            res = recommender.top_by_genre(args.top_genre, top_n=args.top_n, min_ratings=args.min_ratings)
            out = res[['title', 'rating_count']].copy()
            out['title'] = out['title'].str.slice(0,80)
            print(out.to_string(index=False))
        return

    # Simple interactive CLI
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
            recs = recommender.get_recommendations(title, top_n=5)
            if isinstance(recs, str):
                print(recs)
            else:
                print(recs.to_string(index=False))
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


if __name__ == '__main__':
    main()
