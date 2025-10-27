# Machine Learning Analysis of Spotify Tracks
Predicting Music Popularity Using Audio Features & Metadata
This project explores how audio characteristics and metadata influence song popularity on Spotify. Using machine learning and statistical analysis, the study investigates which features most strongly contribute to success and builds predictive and recommendation models based on those insights.
## Objectives
1. Identify relationships between audio features and popularity
2. Build machine learning models to predict track popularity
3. Determine feature importance
4. Group songs into mood-based clusters
5. Create basic recommendation systems
6. Mood-based filtering
7. Content-based similarity
## Dataset
* Source: Kaggle — Spotify Tracks Dataset
* Total Rows: ~113,549 tracks
* Target Variable: Popularity (0–100)
* Feature Types:
Audio features (Energy, Valence, Tempo, Loudness, etc.)
* Metadata (Artist, Genre, Duration, Explicit, Mode)
## Data Preprocessing Steps
✔ Removed missing values
✔ Corrected out-of-range loudness values
✔ Encoded categorical features using:
Label encoding for “explicit”
Target encoding for “artist” & “genre”
✔ Min-Max scaling for tempo, loudness & duration
✔ Yeo-Johnson power transform for skewed target variable
✔ 80–20 Train-Test split
## Exploratory Data Analysis (Key Insights)
Most tracks show low popularity → right-skewed distribution
Energy positively correlated with Loudness
Acousticness negatively correlated with Energy
Tracks under an artist’s dominant genre tend to be more popular
Mood-related features form clusters but with moderate separation
## Machine Learning Models Used
#### Model                                                  Best R² Score                     Notes
Random Forest Regressor                                    0.7788                         Best performance
Gradient Boosting                                             0.7443                         Good, slower
XGBoost0.                                                     0.7476                         Efficient, good accuracy
LightGBM                                                      0.7462                         Fast
KNN Regressor                                                 0.6638                         Weak for this dataset
✅ Selected Model: Random Forest Regressor
## Feature Importance (Top Predictors)
Artist identity (dominates prediction — 80% importance)
Track genre
Tempo
Speechiness
Acousticness
Valence
✔ Popularity is heavily influenced by artist reputation & genre, not just musical qualities.
## Mood-Based Clustering
Algorithm: K-Means (k=4)
Mood Labels:
Happy
Sad
Chill
Aggressive
Silhouette Score: 0.405 (moderate clustering)
## Recommendation Systems
#### System                            Method                                    Output
Mood-Based                      Cluster filtering                     10 songs from selected mood
Content-Based                   Cosine similarity                     Top 5 similar songs
## Tech Stack
Python, Jupyter Notebook
Pandas, NumPy
Scikit-learn, SciPy
Matplotlib, Seaborn
## Conclusion
Popularity can be predicted reasonably well from the dataset
Artist name is the strongest predictor
Audio features alone cannot fully explain song success
Clustering captures basic emotional categories but is subjective
## Future Improvements
Include real Spotify API data: user behavior, playlist placement
Hyperparameter tuning & deep learning models
Better mood taxonomy or listener-based classification
Hybrid recommendation (Content + Collaborative filtering)
