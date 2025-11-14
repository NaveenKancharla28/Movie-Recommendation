Movie Recommendation System

A Python-based Movie Recommendation System that uses data preprocessing, feature extraction, and similarity matching to suggest movies based on user preferences.
This project demonstrates how machine learning and data analysis can be used to build intelligent, content-based recommendation engines.

 Project Structure
File	Description
app.py	Main application entry point. Loads the trained model and runs the recommendation pipeline.
data_preprocessing.py	Cleans, filters, and prepares the raw dataset (movie.csv) for training and feature extraction.
model_builder.py	Builds the feature extraction model using TF-IDF or CountVectorizer and computes cosine similarity between movies.
recommender.py	Contains logic to fetch top N similar movies based on input title.
movie.csv	Dataset containing movie titles, genres, and metadata.
requirements.txt	List of Python dependencies required to run the project.

⚙️ Installation & Setup


Clone the repository

```
git clone https://github.com/NaveenKancharla28/Movie-Recommendation.git 
cd Movie-Recommendation 

```
Create and activate a virtual environment

```
python3 -m venv venv
source venv/bin/activate   # On macOS/Linux
venv\Scripts\activate      # On Windows
 ```

Install dependencies
```
pip install -r requirements.txt

```
Run the app
```
python app.py
```
 How It Works

Data Preprocessing:
The dataset is cleaned, missing values handled, and textual columns like genres and overview are processed for modeling.

Feature Extraction:
The model_builder.py script uses TF-IDF vectorization to convert movie descriptions into numerical feature vectors.

Similarity Computation:
Cosine similarity is computed between all movie vectors to measure how closely they relate.

Recommendation Engine:
Given a movie name, recommender.py retrieves and ranks the top similar movies based on precomputed similarity scores.

 Example Usage
from recommender import recommend_movie

recommend_movie("Inception")

```
Output:

Top 5 similar movies to 'Inception':
1. Interstellar
2. The Matrix
3. The Prestige
4. Shutter Island
5. Memento
```
Requirements

Python 3.8+

pandas

numpy

scikit-learn

Flask (if app.py exposes a web API)

Install with:

pip install -r requirements.txt



👨‍💻 Author

Naveen Kancharla
AI/ML Engineer | Building RAG-powered tools and intelligent automation
🌐 Portfolio https://naveenflix.vercel.app/

💼 LinkedIn https://www.linkedin.com/in/naveen-chaitanya-kancharla-358337238/
