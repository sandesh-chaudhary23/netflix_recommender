import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import requests
from PIL import Image
from io import BytesIO
import base64
import random
from dotenv import load_dotenv

load_dotenv()

# ─── PAGE CONFIG ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="CineAI – Smart Movie Recommender",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ─── CUSTOM CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Bebas+Neue&family=Roboto:wght@300;400;700&display=swap');

/* Dark Netflix-like theme */
body, .stApp { background-color: #141414; color: #e5e5e5; }
.stApp { font-family: 'Roboto', sans-serif; }

/* Hide streamlit elements */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 0rem; }

/* Hero */
.hero { background: linear-gradient(135deg, #000000 0%, #1a0000 40%, #141414 100%);
        padding: 60px 40px; text-align: center; border-bottom: 3px solid #e50914; }
.hero-title { font-family: 'Bebas Neue', cursive; font-size: 5rem; color: #e50914;
              letter-spacing: 6px; text-shadow: 0 0 30px rgba(229,9,20,0.5); margin: 0; }
.hero-sub { font-size: 1.3rem; color: #b3b3b3; margin-top: 10px; }
.film-reel { font-size: 3rem; animation: spin 4s linear infinite; display: inline-block; }
@keyframes spin { 0%{transform:rotate(0deg)} 100%{transform:rotate(360deg)} }

/* Cards */
.movie-card { background: #1f1f1f; border-radius: 8px; overflow: hidden;
              transition: transform .3s, box-shadow .3s; cursor: pointer; }
.movie-card:hover { transform: scale(1.04); box-shadow: 0 8px 30px rgba(229,9,20,0.3); }

/* Badges */
.badge { display: inline-block; background: #e50914; color: white; padding: 2px 10px;
         border-radius: 12px; font-size: 0.75rem; font-weight: 700; }
.badge-gold { background: #f5c518; color: #000; }

/* Section title */
.section-title { font-family: 'Bebas Neue', cursive; font-size: 2rem; color: #e5e5e5;
                 letter-spacing: 3px; border-left: 4px solid #e50914; padding-left: 12px;
                 margin: 30px 0 15px 0; }

/* Tabs */
.stTabs [data-baseweb="tab-list"] { background: #1f1f1f; border-radius: 8px; }
.stTabs [data-baseweb="tab"] { color: #b3b3b3; }
.stTabs [aria-selected="true"] { color: #e50914 !important; border-bottom-color: #e50914 !important; }

/* Search box */
.stTextInput input { background: #2d2d2d; color: #e5e5e5; border: 2px solid #333;
                     border-radius: 6px; font-size: 1.1rem; }
.stTextInput input:focus { border-color: #e50914; }

/* Selectbox */
.stSelectbox select { background: #2d2d2d; color: #e5e5e5; }

/* Rating stars */
.rating { color: #f5c518; font-size: 1rem; }

/* Detail card */
.detail-card { background: linear-gradient(135deg, #1a0000, #1f1f1f);
               border: 1px solid #333; border-radius: 12px; padding: 24px;
               border-left: 4px solid #e50914; }

/* Metric cards */
.metric-box { background: #1f1f1f; border-radius: 8px; padding: 16px; text-align: center;
              border-top: 3px solid #e50914; }
.metric-val { font-size: 2rem; font-weight: 700; color: #e50914; }
.metric-label { font-size: 0.85rem; color: #b3b3b3; }

button[kind="primary"] { background: #e50914 !important; border-color: #e50914 !important; }

/* SVG poster style */
.svg-poster { border-radius: 6px; overflow: hidden; }
</style>
""", unsafe_allow_html=True)

# ─── API KEYS ──────────────────────────────────────────────────────────────────
TMDB_API_KEY = os.getenv("TMDB_API_KEY", "")
OMDB_API_KEY = os.getenv("OMDB_API_KEY", "")
TMDB_CDN = "https://image.tmdb.org/t/p/w342"

# ─── LOAD DATA ─────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_data():
    try:
        df = pd.read_csv("movies_clean.csv")
        with open("similarity.pkl", "rb") as f:
            sim = pickle.load(f)
        return df, sim
    except FileNotFoundError:
        st.error("❌ Run `python model.py` first to generate movies_clean.csv and similarity.pkl")
        st.stop()

df, similarity = load_data()

# ─── POSTER SYSTEM ─────────────────────────────────────────────────────────────
def generate_svg_poster(title, year="", rating=""):
    colors = ["#e50914","#1a3a5c","#1a4a1a","#4a1a4a","#4a3a1a"]
    color = colors[hash(title) % len(colors)]
    short = title[:2].upper() if title else "??"
    year_str = str(year)[:4] if year else ""
    svg = f"""<svg width="150" height="225" xmlns="http://www.w3.org/2000/svg">
  <defs><linearGradient id="g" x1="0" y1="0" x2="0" y2="1">
    <stop offset="0%" stop-color="{color}"/><stop offset="100%" stop-color="#141414"/>
  </linearGradient></defs>
  <rect width="150" height="225" fill="url(#g)" rx="6"/>
  <text x="75" y="95" font-family="Arial" font-size="38" font-weight="bold"
        fill="white" text-anchor="middle" opacity="0.9">{short}</text>
  <text x="75" y="145" font-family="Arial" font-size="11" fill="#ccc"
        text-anchor="middle">{title[:22]}</text>
  <text x="75" y="165" font-family="Arial" font-size="10" fill="#999"
        text-anchor="middle">{year_str}</text>
  <rect x="20" y="185" width="110" height="2" fill="{color}" opacity="0.6"/>
  <text x="75" y="210" font-family="Arial" font-size="9" fill="#aaa"
        text-anchor="middle">🎬 CineAI</text>
</svg>"""
    return f"data:image/svg+xml;base64,{base64.b64encode(svg.encode()).decode()}"

@st.cache_data(show_spinner=False, ttl=3600)
def get_poster(poster_path, title, year="", tmdb_id=None, rating=""):
    # Tier 1: TMDB CDN from CSV
    if poster_path and str(poster_path) != "nan" and str(poster_path).strip():
        url = f"{TMDB_CDN}{poster_path}"
        try:
            r = requests.get(url, timeout=4)
            if r.status_code == 200:
                return url
        except:
            pass

    # Tier 2a: Live TMDB API
    if TMDB_API_KEY and tmdb_id:
        try:
            url = f"https://api.themoviedb.org/3/movie/{tmdb_id}?api_key={TMDB_API_KEY}"
            r = requests.get(url, timeout=4)
            if r.status_code == 200:
                path = r.json().get("poster_path")
                if path:
                    return f"{TMDB_CDN}{path}"
        except:
            pass

    # Tier 2b: OMDb
    if OMDB_API_KEY:
        try:
            yr = str(year)[:4] if year else ""
            url = f"http://www.omdbapi.com/?t={requests.utils.quote(str(title))}&y={yr}&apikey={OMDB_API_KEY}"
            r = requests.get(url, timeout=4)
            if r.status_code == 200:
                poster = r.json().get("Poster")
                if poster and poster != "N/A":
                    return poster
        except:
            pass

    # Tier 3: SVG fallback
    return generate_svg_poster(str(title), str(year)[:4] if year else "", rating)

def show_poster(poster_path, title, year="", tmdb_id=None, rating="", width=150):
    src = get_poster(poster_path, title, year, tmdb_id, rating)
    if src.startswith("data:image/svg"):
        st.markdown(f'<img src="{src}" width="{width}" style="border-radius:6px"/>', unsafe_allow_html=True)
    else:
        try:
            st.image(src, width=width)
        except:
            st.markdown(f'<img src="{generate_svg_poster(title,year,rating)}" width="{width}" style="border-radius:6px"/>', unsafe_allow_html=True)

# ─── SEARCH FUNCTIONS ──────────────────────────────────────────────────────────
def fuzzy_search(query, df, n=10):
    query = query.strip().lower()
    titles = df['title'].fillna('').str.lower()
    # Tier 1: exact
    exact = df[titles == query]
    if not exact.empty:
        return exact.head(n)
    # Tier 2: prefix
    prefix = df[titles.str.startswith(query)]
    if not prefix.empty:
        return prefix.head(n)
    # Tier 3: contains
    contains = df[titles.str.contains(query, na=False)]
    return contains.head(n)

def get_recommendations(title, df, similarity, n=10):
    matches = df[df['title'].str.lower() == title.strip().lower()]
    if matches.empty:
        matches = df[df['title'].str.lower().str.contains(title.strip().lower(), na=False)]
    if matches.empty:
        return pd.DataFrame()
    idx = matches.index[0]
    # Handle both dense and sparse similarity matrices
    try:
        sim_row = similarity[idx]
        if hasattr(sim_row, 'toarray'):
            sim_row = sim_row.toarray().flatten()
        scores = list(enumerate(sim_row))
    except:
        return pd.DataFrame()
    scores = sorted(scores, key=lambda x: x[1], reverse=True)[1:n+1]
    indices = [i[0] for i in scores]
    return df.iloc[indices].copy()

# ─── STAR RATING ───────────────────────────────────────────────────────────────
def star_rating(score):
    stars = round(score / 2)
    return "★" * stars + "☆" * (5 - stars)

# ─── HERO SECTION ──────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
  <div class="film-reel">🎬</div>
  <div class="hero-title">CineAI</div>
  <div class="hero-sub">Your AI-Powered Netflix-Style Movie Recommender · 20,000+ Movies · Powered by TF-IDF + Cosine Similarity</div>
</div>
""", unsafe_allow_html=True)

# ─── SEARCH SECTION ────────────────────────────────────────────────────────────
st.markdown('<div class="section-title">🔍 Smart Search</div>', unsafe_allow_html=True)
col_search, col_btn = st.columns([5, 1])
with col_search:
    query = st.text_input("", placeholder="Search any movie... (e.g. Inception, Dark Knight, Interstellar)", label_visibility="collapsed")
with col_btn:
    st.markdown("<br>", unsafe_allow_html=True)
    search_clicked = st.button("Search 🔎", use_container_width=True)

selected_movie = None

if query:
    results = fuzzy_search(query, df)
    if not results.empty:
        titles_list = results['title'].tolist()
        selected_movie = st.selectbox("Select a movie:", titles_list, label_visibility="collapsed")
    else:
        st.warning("No movies found. Try a different title.")

# ─── MOVIE DETAIL + RECOMMENDATIONS ───────────────────────────────────────────
if selected_movie:
    movie_row = df[df['title'] == selected_movie].iloc[0]

    st.markdown('<div class="section-title">🎥 Movie Details</div>', unsafe_allow_html=True)

    col_img, col_info = st.columns([1, 3])
    with col_img:
        show_poster(
            movie_row.get('poster_path'),
            movie_row['title'],
            movie_row.get('release_date', '')[:4] if pd.notna(movie_row.get('release_date', '')) else '',
            movie_row.get('id'),
            str(movie_row.get('vote_average', '')),
            width=200
        )

    with col_info:
        st.markdown(f'<div class="detail-card">', unsafe_allow_html=True)
        rating = movie_row.get('vote_average', 0)
        year = str(movie_row.get('release_date', ''))[:4]
        genres_raw = movie_row.get('genres', '')
        st.markdown(f"### {movie_row['title']} ({year})")
        st.markdown(f'<span class="rating">{star_rating(float(rating) if rating else 0)}</span> **{rating}/10** · <span class="badge">{genres_raw[:40] if genres_raw else "N/A"}</span>', unsafe_allow_html=True)

        tagline = movie_row.get('tagline', '')
        if tagline and str(tagline) != 'nan':
            st.markdown(f'*"{tagline}"*')

        overview = movie_row.get('overview', '')
        if overview and str(overview) != 'nan':
            st.markdown(f"**Synopsis:** {str(overview)[:400]}...")

        cast = movie_row.get('cast', '')
        director = movie_row.get('director', '')
        if cast and str(cast) != 'nan':
            st.markdown(f"🎭 **Cast:** {str(cast)[:100]}")
        if director and str(director) != 'nan':
            st.markdown(f"🎬 **Director:** {director}")

        votes = movie_row.get('vote_count', 0)
        popularity = movie_row.get('popularity', 0)
        st.markdown(f"👥 **Votes:** {int(votes):,} · 📈 **Popularity:** {float(popularity):.1f}")
        st.markdown('</div>', unsafe_allow_html=True)

    # AI Recommendations
    st.markdown('<div class="section-title">🤖 AI Recommendations</div>', unsafe_allow_html=True)
    recs = get_recommendations(selected_movie, df, similarity, n=10)

    if not recs.empty:
        cols = st.columns(5)
        for i, (_, row) in enumerate(recs.head(10).iterrows()):
            with cols[i % 5]:
                year_r = str(row.get('release_date', ''))[:4]
                rating_r = row.get('vote_average', 0)
                show_poster(row.get('poster_path'), row['title'], year_r, row.get('id'), str(rating_r), width=140)
                st.markdown(f"**{str(row['title'])[:22]}**")
                st.markdown(f'<span class="rating">{star_rating(float(rating_r) if rating_r else 0)}</span> {rating_r}', unsafe_allow_html=True)
    else:
        st.info("No recommendations found for this title.")

# ─── BROWSE SECTIONS ───────────────────────────────────────────────────────────
st.markdown('<div class="section-title">🎭 Explore Movies</div>', unsafe_allow_html=True)

tab1, tab2, tab3, tab4, tab5 = st.tabs(["🔥 Trending Now", "⭐ Top Rated", "🎭 By Genre", "🆕 New Releases", "💎 Hidden Gems"])

def show_movie_row(movies_df, n=10):
    cols = st.columns(5)
    for i, (_, row) in enumerate(movies_df.head(n).iterrows()):
        with cols[i % 5]:
            year_r = str(row.get('release_date', ''))[:4]
            rating_r = row.get('vote_average', 0)
            show_poster(row.get('poster_path'), row['title'], year_r, row.get('id'), str(rating_r), width=140)
            st.markdown(f"**{str(row['title'])[:22]}**")
            st.markdown(f'⭐ {rating_r} · {year_r}')

with tab1:
    trending = df.sort_values('popularity', ascending=False).head(10)
    show_movie_row(trending)

with tab2:
    top_rated = df[df['vote_count'] >= 200].sort_values('vote_average', ascending=False).head(10)
    show_movie_row(top_rated)

with tab3:
    GENRES = ['Action','Comedy','Drama','Horror','Sci-Fi','Romance','Thriller',
              'Animation','Adventure','Crime','Fantasy','Documentary','Mystery','Family','History']
    genre_choice = st.selectbox("Choose Genre:", GENRES)
    genre_movies = df[df['genres'].str.contains(genre_choice, case=False, na=False)].sort_values('popularity', ascending=False).head(10)
    show_movie_row(genre_movies)

with tab4:
    new_movies = df[df['release_date'].str[:4] >= '2010'].sort_values('release_date', ascending=False).head(10)
    show_movie_row(new_movies)

with tab5:
    hidden = df[(df['vote_average'] >= 7.5) & (df['popularity'] < df['popularity'].quantile(0.4))].sort_values('vote_average', ascending=False).head(10)
    show_movie_row(hidden)

# ─── BROWSE BY DECADE ──────────────────────────────────────────────────────────
st.markdown('<div class="section-title">📅 Browse by Decade</div>', unsafe_allow_html=True)
decades = ['1970s', '1980s', '1990s', '2000s', '2010s', '2020s']
decade_map = {'1970s': ('1970','1979'), '1980s': ('1980','1989'), '1990s': ('1990','1999'),
              '2000s': ('2000','2009'), '2010s': ('2010','2019'), '2020s': ('2020','2029')}

dcols = st.columns(6)
selected_decade = None
for i, d in enumerate(decades):
    with dcols[i]:
        if st.button(d, use_container_width=True):
            selected_decade = d

if selected_decade:
    start, end = decade_map[selected_decade]
    decade_movies = df[(df['release_date'].str[:4] >= start) & (df['release_date'].str[:4] <= end)].sort_values('popularity', ascending=False).head(10)
    st.markdown(f"**Top movies from the {selected_decade}:**")
    show_movie_row(decade_movies)

# ─── FOOTER ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown('<p style="text-align:center;color:#666;font-size:0.85rem">CineAI · Built with Python + Streamlit · TF-IDF + Cosine Similarity · TMDB Dataset · AWS EC2 Deployed</p>', unsafe_allow_html=True)
