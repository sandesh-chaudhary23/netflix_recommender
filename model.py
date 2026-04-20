"""
model.py  ──  CineAI v4 · Advanced ML Pipeline
Run ONCE:  python model.py
Requires:  movies_metadata.csv + credits.csv + keywords.csv  (Kaggle TMDB dataset)
           https://www.kaggle.com/datasets/rounakbanik/the-movies-dataset

What this builds:
  - movies_clean.csv   : cleaned, enriched movie data
  - similarity.pkl     : cosine similarity matrix (TF-IDF + weighted features)
  - tfidf_model.pkl    : saved vectorizer for future use
"""

import pandas as pd
import numpy as np
import pickle, ast, re, os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler

print("=" * 60)
print("  CineAI · ML Pipeline  ")
print("=" * 60)

# ── 1. LOAD MAIN DATASET ─────────────────────────────────────────────────────────
print("\n📂  Step 1: Loading movies_metadata.csv …")
df = pd.read_csv("movies_metadata.csv", low_memory=False)
print(f"     Raw rows: {len(df):,}")

COLS = ["id","title","overview","genres","popularity",
        "vote_average","vote_count","release_date","poster_path",
        "runtime","tagline","spoken_languages","production_countries"]
df = df[[c for c in COLS if c in df.columns]].copy()

# ── 2. LOAD CREDITS + KEYWORDS (if available) ────────────────────────────────────
print("📂  Step 2: Loading credits.csv + keywords.csv (if present) …")

has_credits  = os.path.exists("credits.csv")
has_keywords = os.path.exists("keywords.csv")

if has_credits:
    credits = pd.read_csv("credits.csv")
    credits["id"] = credits["id"].astype(str)
    df["id"]      = df["id"].astype(str)

    def extract_cast(raw, top=5):
        try:
            return " ".join(
                m["name"].replace(" ","_")
                for m in ast.literal_eval(raw)[:top]
            )
        except: return ""

    def extract_director(raw):
        try:
            for m in ast.literal_eval(raw):
                if m.get("job") == "Director":
                    return m["name"].replace(" ","_")
        except: pass
        return ""

    credits["cast_str"]     = credits["cast"].apply(extract_cast)
    credits["director_str"] = credits["crew"].apply(extract_director)
    df = df.merge(credits[["id","cast_str","director_str"]], on="id", how="left")
    print("     ✅  Credits merged")
else:
    df["cast_str"]     = ""
    df["director_str"] = ""
    print("     ⚠️  credits.csv not found — skipping cast/director features")

if has_keywords:
    kw = pd.read_csv("keywords.csv")
    kw["id"] = kw["id"].astype(str)

    def extract_keywords(raw):
        try:
            return " ".join(
                k["name"].replace(" ","_")
                for k in ast.literal_eval(raw)
            )
        except: return ""

    kw["keywords_str"] = kw["keywords"].apply(extract_keywords)
    df = df.merge(kw[["id","keywords_str"]], on="id", how="left")
    print("     ✅  Keywords merged")
else:
    df["keywords_str"] = ""
    print("     ⚠️  keywords.csv not found — skipping keyword features")

# ── 3. CLEAN ──────────────────────────────────────────────────────────────────────
print("\n🧹  Step 3: Cleaning …")

df.dropna(subset=["title","overview"], inplace=True)
df.drop_duplicates(subset=["title"],   inplace=True)

df["popularity"]   = pd.to_numeric(df["popularity"],   errors="coerce").fillna(0)
df["vote_average"] = pd.to_numeric(df["vote_average"], errors="coerce").fillna(0)
df["vote_count"]   = pd.to_numeric(df["vote_count"],   errors="coerce").fillna(0)
df["runtime"]      = pd.to_numeric(df.get("runtime",0),errors="coerce").fillna(0)

# Keep movies with at least 15 votes
df = df[df["vote_count"] >= 15].copy()

# Year
df["year"] = pd.to_datetime(df["release_date"], errors="coerce").dt.year.fillna(0).astype(int)
df["title"] = df["title"].str.strip()

def extract_genres(raw):
    try: return " ".join(g["name"] for g in ast.literal_eval(raw))
    except: return ""

df["genres_str"] = df["genres"].apply(extract_genres)

# Fill NaN text cols
for c in ["cast_str","director_str","keywords_str","overview","genres_str","tagline"]:
    df[c] = df.get(c, pd.Series([""]*len(df))).fillna("")

print(f"     Clean rows: {len(df):,}")

# ── 4. WEIGHTED FEATURE ENGINEERING ──────────────────────────────────────────────
print("\n⚙️   Step 4: Building weighted feature vector …")

def clean_text(t):
    t = str(t).lower()
    t = re.sub(r"[^a-z0-9_ ]", " ", t)
    return re.sub(r"\s+", " ", t).strip()

# Weights: genres > keywords > director > overview > cast > tagline
# Higher repetition = higher weight in TF-IDF
df["tags"] = (
    df["genres_str"].apply(clean_text)   + " " +   # weight ×4
    df["genres_str"].apply(clean_text)   + " " +
    df["genres_str"].apply(clean_text)   + " " +
    df["genres_str"].apply(clean_text)   + " " +
    df["keywords_str"].apply(clean_text) + " " +   # weight ×3
    df["keywords_str"].apply(clean_text) + " " +
    df["keywords_str"].apply(clean_text) + " " +
    df["director_str"].apply(clean_text) + " " +   # weight ×2
    df["director_str"].apply(clean_text) + " " +
    df["overview"].apply(clean_text)     + " " +   # weight ×1
    df["cast_str"].apply(clean_text)     + " " +
    df["tagline"].apply(clean_text)
)

# Limit to top 20k by popularity for speed
df = df.nlargest(20_000, "popularity").reset_index(drop=True)
print(f"     Final dataset: {len(df):,} movies")

# ── 5. TF-IDF VECTORIZATION ───────────────────────────────────────────────────────
print("\n🔢  Step 5: TF-IDF vectorization (max_features=8000) …")

tfidf = TfidfVectorizer(
    max_features=8000,
    stop_words="english",
    ngram_range=(1, 2),       # bigrams help (e.g. "science fiction")
    min_df=2,                  # ignore very rare terms
    sublinear_tf=True,         # log-scaling of term frequency
)
matrix = tfidf.fit_transform(df["tags"])
print(f"     Matrix shape: {matrix.shape}")

# ── 6. COSINE SIMILARITY ──────────────────────────────────────────────────────────
print("\n📐  Step 6: Computing cosine similarity …")
sim = cosine_similarity(matrix, dense_output=False)
print(f"     Similarity matrix: {sim.shape}")

# ── 7. SAVE ───────────────────────────────────────────────────────────────────────
print("\n💾  Step 7: Saving artifacts …")

out_cols = ["id","title","genres_str","overview","tagline",
            "popularity","vote_average","vote_count","year",
            "poster_path","runtime","cast_str","director_str","keywords_str"]
out_cols = [c for c in out_cols if c in df.columns]
df[out_cols].to_csv("movies_clean.csv", index=False)

with open("similarity.pkl","wb") as f:
    pickle.dump(sim, f)

with open("tfidf_model.pkl","wb") as f:
    pickle.dump(tfidf, f)

print(f"\n{'='*60}")
print(f"  ✅  Done!")
print(f"  movies_clean.csv  →  {len(df):,} movies")
print(f"  similarity.pkl    →  {sim.shape}")
print(f"  tfidf_model.pkl   →  vocabulary size: {len(tfidf.vocabulary_):,}")
print(f"{'='*60}\n")
