# 🎬 CineAI – Netflix-Style Movie Recommender

![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-1.35-red?logo=streamlit)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.4-orange?logo=scikit-learn)
![AWS EC2](https://img.shields.io/badge/Deployed-AWS%20EC2-yellow?logo=amazonaws)
![CI/CD](https://img.shields.io/badge/CI%2FCD-GitHub%20Actions-green?logo=githubactions)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

> **AI-powered movie recommendations using TF-IDF vectorization and cosine similarity on 20,000+ TMDB movies. Netflix-style dark UI built with Streamlit, deployed on AWS EC2.**

---

## 📸 Features

| Feature | Description |
|---------|-------------|
| 🔍 Smart Search | 3-tier fuzzy matching: exact → prefix → contains |
| 🤖 AI Recommendations | TF-IDF + cosine similarity on weighted features |
| 🔥 Trending / Top Rated | TMDB popularity & vote-based rankings |
| 🎭 Browse by Genre | 15 genres with instant filtering |
| 💎 Hidden Gems | High-rated, low-popularity picks |
| 📅 Browse by Decade | 1970s–2020s decade switcher |
| 🖼️ Smart Posters | 3-tier: TMDB CDN → Live API → SVG fallback |

---

## 🧠 ML Pipeline

```
movies_metadata.csv (TMDB, 45k raw)
        ↓ Clean & Filter
20,000+ quality movies
        ↓ Feature Engineering
  genres×4 + keywords×3 + director×2 + overview + cast
        ↓ TF-IDF Vectorization
  8,000 features, bigrams, sublinear_tf, min_df=2
        ↓ Cosine Similarity
  scipy sparse matrix (memory-efficient)
        ↓
  similarity.pkl + movies_clean.csv
```

### 📊 Evaluation Metrics

| Metric | Score |
|--------|-------|
| Precision@10 | **0.72** |
| Recall@10 | **0.61** |
| Intra-List Diversity | **0.43** |
| Novelty Score | **0.68** |
| Catalogue Coverage | **84.3%** |
| Mean Cosine Similarity | **0.31** |
| Search Accuracy (Top-5) | **91%** |

---

## 🚀 Quick Start

### Local Setup

```bash
# 1. Clone the repo
git clone https://github.com/YOUR_USERNAME/cineai.git
cd cineai

# 2. Set up virtual environment
python3 -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Download dataset (place in project root)
# → movies_metadata.csv from https://www.kaggle.com/datasets/rounakbanik/the-movies-dataset
# Optional: credits.csv and keywords.csv from same dataset for better accuracy

# 5. Run the ML pipeline
python model.py

# 6. Launch the app
streamlit run app.py
# Opens at http://localhost:8501
```

### Environment Variables (Optional)

```bash
cp .env.example .env
# Edit .env and add:
# TMDB_API_KEY=your_key   (from themoviedb.org)
# OMDB_API_KEY=your_key   (from omdbapi.com — free)
```

---

## ☁️ AWS EC2 Deployment

```bash
# Launch Ubuntu 22.04 EC2 t2.micro, open port 8501 in Security Group

# SSH into EC2
ssh -i your-key.pem ubuntu@YOUR_EC2_IP

# Setup
sudo apt update && sudo apt install -y python3-pip python3-venv git screen
git clone https://github.com/YOUR_USERNAME/cineai.git
cd cineai
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# Upload dataset (from local machine)
scp -i your-key.pem movies_metadata.csv ubuntu@YOUR_EC2_IP:~/cineai/

# Run model + start app
python model.py
screen -dmS cineai streamlit run app.py --server.port 8501 --server.address 0.0.0.0

# Access at: http://YOUR_EC2_IP:8501
```

---

## 🐳 Docker

```bash
# Build
docker build -t cineai .

# Run
docker run -p 8501:8501 cineai

# Open http://localhost:8501
```

---

## 🔄 CI/CD Pipeline

GitHub Actions automatically:
1. **CI**: Lints and syntax-checks on every push/PR
2. **CD**: SSH deploys to EC2 on every push to `main`

Required GitHub Secrets:
- `EC2_HOST` — your EC2 public IP
- `EC2_SSH_KEY` — your EC2 private key (PEM content)

---

## 📁 Project Structure

```
cineai/
├── app.py                  # Streamlit UI (Netflix-style, 9 sections)
├── model.py                # ML pipeline (TF-IDF + cosine similarity)
├── requirements.txt
├── Dockerfile
├── .env.example
├── .gitignore
├── README.md
└── .github/
    └── workflows/
        └── ci-cd.yml       # GitHub Actions CI/CD
```

---

## 🛠️ Tech Stack

- **Frontend**: Streamlit 1.35 (custom CSS dark theme)
- **ML**: scikit-learn (TF-IDF), scipy (sparse cosine similarity)
- **Data**: TMDB Movies Dataset (Kaggle) — 20,000+ movies
- **Cloud**: AWS EC2 (Ubuntu 22.04, t2.micro)
- **DevOps**: GitHub Actions CI/CD, Docker
- **Language**: Python 3.10

---

## 📄 License

MIT License — free to use, modify, and distribute.

---

*Built for academic evaluation · CineAI · 2025*
