# MuseRec: Advanced Music Recommendation System

![MuseRec Demo](https://img.shields.io/badge/Status-Active-brightgreen.svg)
![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![Flask](https://img.shields.io/badge/Framework-Flask-black.svg)
![Machine Learning](https://img.shields.io/badge/ML-Scikit--Learn-orange.svg)

MuseRec is a full-stack, machine-learning-powered web application that delivers highly personalized music recommendations. It employs a **Hybrid Recommendation Engine**, combining Content-Based Filtering (audio features and text TF-IDF) and Collaborative Filtering (historical user-item interactions), wrapped in a sleek, premium, Spotify-inspired user interface.

## 🚀 Key Features

### Machine Learning Engine
- **Content-Based Filtering**: Analyzes song acoustic traits (Danceability, Energy, Valence, Acousticness, etc.) and genre tags using TF-IDF Vectorization and MinMax/Standard Scaling.
- **Collaborative Filtering**: Leverages large-scale user interaction data to recommend songs based on similar user listening histories (using Cosine Similarity on sparse matrices).
- **Hybrid System**: Intelligently weights Content-based and Collaborative scores (e.g., 30/70 split) to surface highly accurate and novel recommendations.

### Web Application & UI
- **Premium Interface**: A modern, dark-themed UI featuring glassmorphism, dynamic gradients, and smooth staggered animations.
- **Authentication System**: Secure user registration and login utilizing HTTP-only JWT cookies and Bcrypt password hashing.
- **Interactive Analytics**: Embedded Plotly.js **Radar Charts** visually break down the audio profile of any selected song.
- **Live Audio Playback**: Integrated iTunes API fetches and plays 30-second high-quality audio previews directly in the browser with an interactive bottom player bar.
- **Playlist Management**: Logged-in users can "like" tracks to build their own persistent library.
- **Global Search**: Debounced search functionality allowing instant lookups of songs and artists.

## 📁 Project Structure

This repository follows the industry-standard **Cookiecutter Data Science** layout for reproducibility and clean separation of concerns:

```text
Music_Recommendation_System/
├── README.md               <- The top-level README for developers using this project.
├── data/                   
│   ├── raw/                <- The original, immutable data dump (Music Info.csv, User Listening History.csv).
│   ├── interim/            <- Intermediate data that has been transformed.
│   └── processed/          <- The final, canonical data sets for modeling (cleaned_data.csv, etc.).
├── models/                 <- Trained and serialized models (e.g., transformer.joblib).
├── notebooks/              <- Jupyter notebooks, exploration scripts, or debug JSONs.
├── src/                    <- Source code for machine learning and data processing.
│   ├── __init__.py         <- Makes src a Python module.
│   ├── data/               <- Scripts to download or generate data (data_cleaning.py, etc.)
│   ├── features/           <- Scripts to turn raw data into features for modeling.
│   └── models/             <- Scripts to train ML models and make predictions.
│       ├── collaborative_filtering.py
│       ├── content_based_filtering.py
│       └── hybrid_recommendation.py
├── app/                    <- The core Flask Web Application.
│   ├── __init__.py         <- Flask app factory and extensions setup.
│   ├── auth.py             <- Authentication routes (Register/Login).
│   ├── models.py           <- SQLAlchemy Database models (User, Playlist, PlaylistSong).
│   ├── routes.py           <- Frontend routes and API endpoints.
│   └── utils.py            <- Helper functions linking the app to ML pipelines in `src/`.
├── templates/              <- Flask Jinja2 HTML templates.
├── static/                 <- CSS stylesheets and frontend Vanilla JavaScript.
├── requirements.txt        <- The requirements file for reproducing the analysis environment.
└── app.py                  <- Entry point script to run the Flask web server.
```

## 🛠️ Tech Stack
- **Backend**: Python, Flask, Flask-SQLAlchemy, Flask-JWT-Extended, Flask-Bcrypt
- **Data Science / ML**: Pandas, NumPy, SciPy, Scikit-Learn, Category Encoders, Dask
- **Frontend**: HTML5, Vanilla CSS3 (Custom Variables, CSS Grid/Flexbox), Vanilla JavaScript (ES6+), Jinja2
- **Data Viz**: Plotly.js

## ⚙️ Installation & Local Setup

1. **Clone the repository** and navigate to the project root.
2. **Create a virtual environment**:
   ```bash
   python -m venv venv
   # On Windows:
   venv\Scripts\activate
   # On Mac/Linux:
   source venv/bin/activate
   ```
3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
4. **Prepare the Data / Models (if not already built)**:
   - Place `Music Info.csv` and `User Listening History.csv` into `data/raw/`.
   - Run the preprocessing scripts located in `src/data/` to generate `data/processed/` outputs.
   - Run the pipeline scripts in `src/models/` to generate the `.npz` sparse matrices and `.joblib` objects in `models/`.
5. **Initialize the Database**:
   *(The app utilizes SQLite by default. The database `site.db` will be auto-generated in `instance/` on first run).*
6. **Run the Flask Application**:
   ```bash
   python app.py
   ```
7. **Access the App**: Navigate to `http://127.0.0.1:5000` in your web browser.

## 🧠 Future Enhancements
- Dockerizing the application for single-command deployments.
- Migrating the local SQLite database to PostgreSQL for production scaling.
- Implementing Matrix Factorization (SVD) or Deep Learning (Neural Collaborative Filtering) alongside Cosine Similarity.
- Automating the ML Model retraining pipeline using Airflow or Prefect.

---
*Built with ❤️ for music discovery and robust data science engineering.*
