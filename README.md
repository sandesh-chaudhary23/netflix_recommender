# 🎬 CineAI v4 — Netflix-Style Movie Recommender

## 📁 Folder Structure
```
netflix_recommender/
├── app.py               ← Full Netflix UI (9 sections)
├── model.py             ← ML pipeline (TF-IDF + cosine similarity)
├── requirements.txt
├── .env.example         ← Optional TMDB key
├── AWS_DEPLOYMENT.md    ← AWS EC2 guide
├── movies_metadata.csv  ← Already have ✅
├── credits.csv          ← Download from Kaggle (optional, better accuracy)
├── keywords.csv         ← Download from Kaggle (optional, better accuracy)
├── movies_clean.csv     ← Auto-generated
├── similarity.pkl       ← Auto-generated
└── tfidf_model.pkl      ← Auto-generated
```

## ▶️ Run Steps

```bash
cd ~/Downloads/netflix_recommender
source venv/bin/activate
pip install -r requirements.txt
python model.py           # rebuild model (do this once)
streamlit run app.py      # launch at localhost:8501
```

## 🔑 Posters — No API key needed
`movies_metadata.csv` already has `poster_path` → used with free TMDB CDN.
Missing posters → auto SVG fallback (works offline).

## 🧠 ML Accuracy (v4)
- TF-IDF: 8000 features, bigrams, sublinear TF scaling
- Features: genres×4 + keywords×3 + director×2 + overview + cast
- Add `credits.csv` + `keywords.csv` from Kaggle for best accuracy

## 🚀 AWS: See AWS_DEPLOYMENT.md
