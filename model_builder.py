from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def build_similarity_model(movies_df, ngram_range=(1,2)):
    """Given a movies dataframe with a 'genres_clean' column, return cosine similarity matrix."""
    tfidf = TfidfVectorizer(stop_words='english', ngram_range=ngram_range)
    tfidf_matrix = tfidf.fit_transform(movies_df['genres_clean'])
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
    return cosine_sim
